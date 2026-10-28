"""Train a real-only microscopy DDPM prior for SimCalc additive augmentation.

Python 3.10+, PyTorch 2.x. See ADDITIVE_SIMCALC.md for this integration.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from diffusion_unet import DiffusionUNet
from diffusion_sampling import cosine_schedule, q_sample

ROOT = Path(__file__).resolve().parent
FORMAT = 'simcalc-real-ddpm-addon-v1'
UPDATE_VERSION = 'portable-v3-compatible'


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def tensor_images(images, indices, device, pad=False):
    a = images[np.asarray(indices)].astype(np.float32) / 32767.5 - 1.0
    x = torch.from_numpy(a[:, None]).to(device)
    if pad:
        x = F.pad(x, (0, (-x.shape[-1]) % 8, 0, (-x.shape[-2]) % 8), mode='replicate')
    return x


class CyclingSampler:
    """Checkpointable shuffled sampler, with no worker/prefetch ambiguity."""
    def __init__(self, indices, seed):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.indices)
        self.position = 0
        self.cycles = 0

    def take(self, count):
        result = []
        while len(result) < count:
            n = min(count - len(result), len(self.order) - self.position)
            result.extend(self.order[self.position:self.position + n].tolist())
            self.position += n
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.position = 0
                self.cycles += 1
        return result

    def state_dict(self):
        return dict(order=self.order, position=self.position, cycles=self.cycles,
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.order = state['order']
        self.position = state['position']
        self.cycles = state['cycles']
        self.rng.bit_generator.state = state['rng']


def autocast(enabled):
    return torch.autocast('cuda', dtype=torch.float16) if enabled else nullcontext()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def read_checkpoint(path, allow_legacy=False):
    # Load only checkpoints produced by this package or otherwise trusted by you.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    allowed={FORMAT,'zebrafish-real-ddpm-sdedit-v1'} if allow_legacy else {FORMAT}
    if checkpoint.get('format') not in allowed:
        raise ValueError('Expected this package DDPM checkpoint; GAN checkpoints are incompatible')
    return checkpoint


@torch.no_grad()
def validate(model, images, indices, alphas, device, amp, seed, count=8):
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 91037)
    values = []
    bins = {'low': [], 'mid': [], 'high': []}
    selected = np.asarray(indices)[np.linspace(0, len(indices) - 1, min(count, len(indices))).astype(int)]
    times = np.linspace(50, len(alphas) - 50, len(selected)).round().astype(int)
    for index, timestep in zip(selected, times):
        clean = tensor_images(images, [index], device, pad=True)
        noise = torch.randn(clean.shape, generator=generator, device=device)
        t = torch.tensor([timestep], device=device, dtype=torch.long)
        noisy = q_sample(clean, t, noise, alphas)
        with autocast(amp):
            predicted = model(noisy, t)
        h, w = images.shape[-2:]
        loss = F.mse_loss(predicted.float()[:, :, :h, :w], noise[:, :, :h, :w]).item()
        values.append(loss)
        key = 'low' if timestep < 333 else 'mid' if timestep < 667 else 'high'
        bins[key].append(loss)
    model.train(was_training)
    return {'val_mse': float(np.mean(values)), **{
        f'val_{key}_mse': float(np.mean(v)) if v else float('nan') for key, v in bins.items()}}


@torch.no_grad()
def preview(model, synthetic, real, alphas, device, amp, args, step, run):
    from simcalc_addon import preview_simcalc
    return preview_simcalc(model, synthetic, real, alphas, device, amp, args, step, run)


def session_end_step(start, target, stop_after):
    """Bound one session without changing the original LR schedule horizon."""
    if start < 0 or target < 1 or stop_after < 0:
        raise ValueError('Invalid checkpoint/target/session budget')
    return min(target, start + stop_after) if stop_after else target


def train(args, resume=None, initial=None):
    device = torch.device(args.device)
    amp = device.type == 'cuda' and not args.no_amp
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    from simcalc_data import load_domains
    real, synthetic = load_domains(args)
    real_train = np.flatnonzero(real['split'] == 0)
    real_val = np.flatnonzero(real['split'] == 1)
    if not len(real_train) or not len(real_val):
        raise ValueError('Real data requires nonempty disjoint train and validation splits')
    if synthetic['images'].shape[1:] != real['images'].shape[1:]:
        raise ValueError('Both domains must have the same native full-frame shape')
    data_hashes = dict(real_manifest_sha256=real['fingerprint'],
                       synthetic_npz_sha256=synthetic['simcalc']['source_sha256'])
    if resume and data_hashes != resume['data_sha256']:
        raise ValueError('Dataset differs from the resumed experiment. To start a new run on new '
                         'data with existing weights, use --init-from CHECKPOINT instead of --resume')
    model = DiffusionUNet(args.channels, args.dropout, not args.no_attention,
                          gradient_checkpointing=device.type == 'cuda' and not args.no_checkpointing).to(device)
    if initial:
        model.load_state_dict(initial['ema'])
        print(f'Warm start from EMA at old step {initial["step"]}; new optimizer/sampler/validation history.', flush=True)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                 betas=(0.9, 0.999))
    scaler = (torch.amp.GradScaler('cuda', enabled=amp)
              if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler(enabled=amp))
    alphas = cosine_schedule(args.timesteps).to(device)
    sampler = CyclingSampler(real_train, args.seed + 17)
    start, best, bad_checks = 0, float('inf'), 0
    if resume:
        model.load_state_dict(resume['model'])
        ema.load_state_dict(resume['ema'])
        optimizer.load_state_dict(resume['optimizer'])
        if resume.get('scaler'):
            scaler.load_state_dict(resume['scaler'])
        sampler.load_state_dict(resume['sampler'])
        start, best = resume['step'], resume['best_val_mse']
        bad_checks = resume.get('bad_checks', 0)
        restore_rng(resume['rng'])
        run = Path(args.resume).resolve().parent
    else:
        run = Path(args.output) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        run.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(format=FORMAT, update_version=UPDATE_VERSION, run=str(run), data_sha256=data_hashes,
                  real_train=len(real_train), real_val=len(real_val),
                  real_test=int(np.sum(real['split'] == 2)),
                  real_excluded_gap=int(np.sum(real['split'] == -1)),
                  real_metadata=json.loads(str(real['metadata_json'])),
                  synthetic_count=len(synthetic['images']), native_shape=list(real['images'].shape[1:]),
                  synthetic_metadata=synthetic['simcalc']['metadata'],
                  training_target='noise on REAL training images only',
                  best_metric='fixed-noise real-val epsilon MSE; not refinement realism',
                  parameters=sum(p.numel() for p in model.parameters()))
    json_write(run / 'config.json', config)
    print(f'Device={device}; AMP={amp}; parameters={config["parameters"]:,}; '
          f'full frame={real["images"].shape[1:]}; real train/val={len(real_train)}/{len(real_val)}', flush=True)
    print(f'Output: {run}\nReal-only DDPM training; residual SDEdit previews use SimCalc. '
          f'Early stopping={"off" if args.patience == 0 else args.patience}', flush=True)
    if device.type == 'cpu':
        print('CPU mode: suitable for smoke checks. Use GPU for full training.', flush=True)
    if start >= args.steps:
        print(f'Checkpoint already at step {start}; --steps must exceed it to continue.', flush=True)
        return run
    stop_step = session_end_step(start, args.steps, args.stop_after)
    print(f'Resume step={start}; this session ends at {stop_step}; LR horizon={args.steps}', flush=True)
    if args.preview_on_start:
        print('Preview existing EMA before any new training step...', flush=True)
        preview(ema, synthetic, real, alphas, device, amp, args, start, run)
    log_path = run / 'metrics.csv'
    fields = ['step', 'train_epsilon_mse', 'val_mse', 'val_low_mse', 'val_mid_mse',
              'val_high_mse', 'lr', 'grad_norm', 'ema_decay', 'elapsed_seconds', 'best_val_mse']
    if not log_path.exists():
        with log_path.open('w', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    model.train()
    clock_start = time.perf_counter()
    losses = []
    log_losses = []
    height, width = real['images'].shape[-2:]
    for step in range(start + 1, stop_step + 1):
        # Warm-up then cosine decay; --steps counts optimizer iterations.
        warm = min(1.0, step / max(1, args.warmup))
        progress = max(0, step - args.warmup) / max(1, args.steps - args.warmup)
        lr = args.lr * warm * (0.1 + 0.9 * (1 + math.cos(math.pi * min(progress, 1))) / 2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        for _ in range(args.accumulation):
            clean = tensor_images(real['images'], sampler.take(args.batch_size), device)
            if not args.no_flip and torch.rand(()).item() < 0.5:
                clean = clean.flip(-1)
            clean = F.pad(clean, (0, (-clean.shape[-1]) % 8, 0, (-clean.shape[-2]) % 8), mode='replicate')
            timestep = torch.randint(0, args.timesteps, (clean.shape[0],), device=device)
            noise = torch.randn_like(clean)
            noisy = q_sample(clean, timestep, noise, alphas)
            with autocast(amp):
                predicted = model(noisy, timestep)
                loss = F.mse_loss(predicted.float()[:, :, :height, :width], noise[:, :, :height, :width])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite diffusion loss; inspect input and AMP settings')
            scaler.scale(loss / args.accumulation).backward()
            micro_losses.append(loss.item())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= old_scale
        if not updated:
            print(f'Step {step}: AMP overflow, optimizer update skipped; reduced scale.', flush=True)
        decay = min(args.ema_decay, (1 + step) / (10 + step))
        if updated:
            with torch.no_grad():
                for target, current in zip(ema.parameters(), model.parameters()):
                    target.lerp_(current.detach(), 1 - decay)
                for target, current in zip(ema.buffers(), model.buffers()):
                    target.copy_(current)
        losses.extend(micro_losses)
        log_losses.extend(micro_losses)
        if step == start + 1 or step % args.log_every == 0:
            print(f'Step {step}/{args.steps}; real epsilon MSE={np.mean(log_losses):.5f}; '
                  f'lr={lr:.2g}; sampler cycles={sampler.cycles}; '
                  f'elapsed={time.perf_counter() - clock_start:.1f}s', flush=True)
            log_losses.clear()
        save_due = step % args.save_every == 0 or step == stop_step
        preview_due = args.preview_every > 0 and (step % args.preview_every == 0 or step == stop_step)
        if save_due or preview_due:
            metrics = validate(ema, real['images'], real_val, alphas, device, amp, args.seed, args.val_count)
            improved = metrics['val_mse'] < best - args.min_delta
            if improved:
                best, bad_checks = metrics['val_mse'], 0
            else:
                bad_checks += 1
            state = dict(format=FORMAT, step=step, model=model.state_dict(), ema=ema.state_dict(),
                         model_config=model.config, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         sampler=sampler.state_dict(), rng=rng_state(), args=vars(args),
                         data_sha256=data_hashes, best_val_mse=best, bad_checks=bad_checks,
                         validation=metrics, domain_metadata=json.loads(str(real['metadata_json'])),
                         update_version=UPDATE_VERSION)
            save_checkpoint(run / 'last.pt', state)
            if save_due:
                save_checkpoint(run / f'step_{step:05d}.pt', state)
            if improved:
                save_checkpoint(run / 'best.pt', state)
            row = dict(step=step, train_epsilon_mse=float(np.mean(losses)), **metrics,
                       lr=lr, grad_norm=float(grad_norm), ema_decay=decay,
                       elapsed_seconds=time.perf_counter() - clock_start, best_val_mse=best)
            with log_path.open('a', newline='', encoding='utf-8') as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            print(f'Validation fixed-noise MSE={metrics["val_mse"]:.5f}; '
                  f'low/mid/high={metrics["val_low_mse"]:.4f}/{metrics["val_mid_mse"]:.4f}/'
                  f'{metrics["val_high_mse"]:.4f}; checkpoint={run / "last.pt"}', flush=True)
            losses.clear()
            if preview_due:
                preview(ema, synthetic, real, alphas, device, amp, args, step, run)
            if args.patience > 0 and bad_checks >= args.patience:
                print(f'Early stop: {bad_checks} validation checks without min_delta improvement. '
                      'This monitors denoising MSE, not visual refinement.', flush=True)
                break
    print(f'Done. Resume/refine checkpoint: {run / "last.pt"}', flush=True)
    return run


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--synthetic-npz', default=str(ROOT / 'data' / 'hybrid_highSNR_1000.npz'))
    p.add_argument('--synthetic-cache', default='/content/simcalc_cache')
    p.add_argument('--real-input', help='Optional prepared real directory, uint16 NPY/NPZ/TIFF or grayscale frame directory')
    p.add_argument('--real-images-key', default='')
    p.add_argument('--real-split-file', help='Optional real split NPY with -1/0/1/2 codes')
    p.add_argument('--residual-mix', type=float, default=.75)
    p.add_argument('--max-delta', type=float, default=.15)
    p.add_argument('--cell-mix', type=float, default=.80)
    p.add_argument('--zenodo-cache', default=('/content/zenodo_zebrafish' if os.name != 'nt'
                   else str(ROOT / 'tmp' / 'zenodo_zebrafish')))
    p.add_argument('--temporal-gap', type=int, default=32)
    p.add_argument('--norm-samples', type=int, default=512)
    p.add_argument('--output', default=str(ROOT / 'diffusion_runs'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--stop-after', type=int, default=0, help='New steps in this session; 0 reaches --steps; LR horizon unchanged')
    p.add_argument('--preview-on-start', action='store_true', help='Preview loaded EMA before new training')
    p.add_argument('--preview-strengths', type=float, nargs='+', default=[.15,.35,.55,.75])
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--no-attention', action='store_true')
    p.add_argument('--no-checkpointing', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-flip', action='store_true')
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--ema-decay', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--preview-every', type=int, default=500)
    p.add_argument('--preview-count', type=int, default=2)
    p.add_argument('--val-count', type=int, default=32)
    p.add_argument('--sample-steps', type=int, default=100)
    p.add_argument('--strength', type=float, default=0.35)
    p.add_argument('--structure-guidance', type=float, default=0.0)
    p.add_argument('--dashboard', choices=['inline', 'none'], default='inline')
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--patience', type=int, default=0, help='Validation checks; 0 disables early stopping')
    p.add_argument('--min-delta', type=float, default=1e-5)
    p.add_argument('--resume')
    p.add_argument('--init-from', help='Initialize EMA weights on a new dataset; reset training state')
    return p


def main(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(cli)
    if args.resume and args.init_from:
        raise ValueError('Choose --resume for the same experiment OR --init-from for a new dataset')
    resume = read_checkpoint(args.resume) if args.resume else None
    initial = read_checkpoint(args.init_from,allow_legacy=True) if args.init_from else None
    explicit = {x.split('=')[0].removeprefix('--').replace('-', '_') for x in cli if x.startswith('--')}
    if initial:
        for key in ['channels', 'dropout', 'no_attention', 'timesteps']:
            value = initial['args'][key]
            if key in explicit and getattr(args, key) != value:
                raise ValueError(f'--init-from requires checkpoint architecture: {key}={value}')
            setattr(args, key, value)
    if resume:
        # Keep explicit command-line overrides; recover stored run configuration otherwise.
        for key, value in resume['args'].items():
            if key not in explicit and key not in {'resume', 'mode', 'device', 'init_from'}:
                setattr(args, key, value)
        for key in ['channels', 'dropout', 'no_attention', 'timesteps']:
            if getattr(args, key) != resume['args'][key]:
                raise ValueError(f'Cannot change --{key.replace("_", "-")} when resuming')
    if args.steps < 1 or args.batch_size < 1 or args.accumulation < 1:
        raise ValueError('steps, batch-size and accumulation must be positive')
    if args.save_every < 1 or args.log_every < 1 or args.preview_count < 1:
        raise ValueError('save-every, log-every and preview-count must be positive')
    if args.timesteps < 100 or not 0 <= args.strength <= 1 or args.sample_steps < 1:
        raise ValueError('timesteps>=100, strength in [0,1], sample-steps>=1 required')
    if not 0 <= args.structure_guidance <= 1 or not 0 <= args.ema_decay < 1:
        raise ValueError('structure-guidance in [0,1], ema-decay in [0,1) required')
    if args.temporal_gap < 0 or args.norm_samples < 1 or args.val_count < 1:
        raise ValueError('temporal-gap>=0, norm-samples>=1, val-count>=1 required')
    if args.stop_after < 0 or not args.preview_strengths or any(not 0 <= s <= 1 for s in args.preview_strengths):
        raise ValueError('stop-after>=0 and preview-strengths in [0,1] required')
    if any(not 0 <= value <= 1 for value in [args.residual_mix,args.max_delta,args.cell_mix]):
        raise ValueError('residual-mix/max-delta/cell-mix must be in [0,1]')
    return train(args, resume, initial)


if __name__ == '__main__':
    main()
