"""Unconditional grayscale image diffusion: independent training and DDIM sampling.
Velocity prediction, zero-terminal-SNR cosine schedule and EMA; geometry comes from the dataset."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from .diffusion_unet import DiffusionUNet
from .diffusion_sampling import cosine_schedule
from .zenodo import load_zenodo_full
from .unetpp import (FORMAT as TEACHER_FORMAT, UNetPlusPlus,
                               normalize_image, predict_probability)

FORMAT = 'neural-naomi-static-vdiffusion-v1'
SHAPE = (400, 485)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')


def device_and_amp(args):
    requested = getattr(args, 'device', 'auto')
    selected = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    device = torch.device(selected)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; choose --device cpu or a GPU runtime.')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device, bool(device.type == 'cuda' and not args.no_amp)


def autocast(device, amp):
    return torch.autocast(device.type, dtype=torch.float16) if amp else nullcontext()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    ns = np.random.get_state()
    return dict(python=random.getstate(), numpy=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch'])
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


class Sampler:
    def __init__(self, indices, seed):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.indices)
        self.position = self.cycles = 0

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
        return dict(order=self.order.tolist(), position=self.position, cycles=self.cycles,
                    rng=self.rng.bit_generator.state)

    def load(self, state):
        self.order = np.asarray(state['order'], dtype=np.int64)
        self.position, self.cycles = state['position'], state['cycles']
        self.rng.bit_generator.state = state['rng']


def schedule(timesteps, device):
    alphas = cosine_schedule(timesteps).to(device)
    alphas[-1] = 0.0
    return alphas


def batch_tensor(images, indices, device):
    pixels = images[np.asarray(indices)].astype(np.float32) / 32767.5 - 1.0
    return torch.from_numpy(pixels[:, None]).to(device)


def noisy_target(clean, noise, timesteps, alphas):
    a = alphas[timesteps, None, None, None]
    noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
    velocity = a.sqrt() * noise - (1 - a).sqrt() * clean
    return noisy, velocity


@torch.inference_mode()
def sample(model, alphas, device, amp, seed, steps=100, eta=0., shape=None):
    """Pure-noise DDIM; no real/synthetic image is an inference input."""
    if not 2 <= steps <= len(alphas) or not 0 <= eta <= 1:
        raise ValueError('Need 2 <= sample_steps <= timesteps and 0 <= eta <= 1')
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn((1, 1, *(shape or SHAPE)), generator=generator, device=device)
    grid = torch.linspace(len(alphas) - 1, 0, steps, dtype=torch.float64).round().long().unique_consecutive().tolist()
    previous_mode = model.training
    model.eval()
    try:
        for index, t in enumerate(grid):
            times = torch.full((1,), t, device=device, dtype=torch.long)
            with autocast(device, amp):
                velocity = model(x, times).float()
            if not torch.isfinite(velocity).all():
                raise RuntimeError('Non-finite sampling output; try --no-amp')
            a = alphas[t]
            clean = (a.sqrt() * x - (1 - a).sqrt() * velocity).clamp(-1, 1)
            following = grid[index + 1] if index + 1 < len(grid) else -1
            if following < 0:
                x = clean
                break
            an = alphas[following]
            # Recompute epsilon after x0 clipping, including the zero-SNR start.
            epsilon = (x - a.sqrt() * clean) / (1 - a).clamp_min(1e-12).sqrt()
            sigma = eta * (((1 - an) / (1 - a).clamp_min(1e-12)) *
                           (1 - a / an.clamp_min(1e-12))).clamp_min(0).sqrt()
            x = an.sqrt() * clean + (1 - an - sigma.square()).clamp_min(0).sqrt() * epsilon
            if eta:
                x = x + sigma * torch.randn(x.shape, generator=generator, device=device)
        return ((x[0, 0].clamp(-1, 1) + 1) / 2).cpu().numpy()
    finally:
        model.train(previous_mode)


def load_diffusion(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != FORMAT:
        raise ValueError('Нужны веса neural_naomi этого комплекта; старые SDEdit/CUT checkpoints несовместимы.')
    return state


def teacher_state(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Не найден обученный U-Net++: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    required = {'model', 'base_channels', 'normalization', 'threshold', 'native_shape_hw'}
    if state.get('format') != TEACHER_FORMAT or not required <= state.keys():
        raise ValueError('best.pt должен быть checkpoint нашего train_unetpp_naomi.py '
                         '(naomi-unetpp-fullframe-v1), с normalization и threshold.')
    if len(state['native_shape_hw']) != 2:
        raise ValueError('U-Net++ requires a 2D image size')
    if not 0 < float(state['threshold']) < 1:
        raise ValueError('Invalid U-Net++ threshold')
    for key in ('low_percentile', 'high_percentile'):
        if key not in state['normalization']:
            raise ValueError('Неизвестная нормализация U-Net++: не подставляем её автоматически.')
    return state


@torch.inference_mode()
def segment(model, raw_adu, settings, threshold, device, amp, tta=False):
    image = normalize_image(raw_adu, settings)
    x = torch.from_numpy(image[None, None]).to(device)
    flips = [()] if not tta else [(), (-1,), (-2,), (-2, -1)]
    result = torch.zeros_like(x, dtype=torch.float32)
    for dims in flips:
        value = x.flip(dims) if dims else x
        with autocast(device, amp):
            probability = predict_probability(model(value))
        result += probability.flip(dims) if dims else probability
    probability = (result / len(flips))[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError('Non-finite U-Net++ output')
    return probability, probability >= threshold


@torch.inference_mode()
def validate(model, images, indices, alphas, device, amp, seed, count=18):
    previous_mode = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 91307)
    chosen = np.asarray(indices)[np.linspace(0, len(indices) - 1, min(count, len(indices))).astype(int)]
    times = np.linspace(10, len(alphas) - 1, len(chosen)).round().astype(int)
    values = []
    try:
        for index, t in zip(chosen, times):
            clean = batch_tensor(images, [index], device)
            noise = torch.randn(clean.shape, generator=generator, device=device)
            noisy, target = noisy_target(clean, noise, torch.tensor([t], device=device), alphas)
            with autocast(device, amp):
                predicted = model(noisy, torch.tensor([t], device=device))
            loss = F.mse_loss(predicted.float(), target).item()
            if not math.isfinite(loss):
                raise RuntimeError('Non-finite validation loss')
            values.append(loss)
        return float(np.mean(values))
    finally:
        model.train(previous_mode)


def grid_preview(path, images, titles, mask=None):
    height, width = images[0].shape[:2]
    canvas = Image.new('RGB', (width * len(images), height + 30), 'black')
    draw = ImageDraw.Draw(canvas)
    for i, pixels in enumerate(images):
        plane = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
        if plane.ndim == 2:
            plane = np.repeat(plane[..., None], 3, axis=-1)
        if mask is not None and i == len(images) - 1:
            plane = plane.copy()
            plane[mask] = np.rint(.55 * plane[mask] + .45 * np.array([0, 255, 90])).astype(np.uint8)
        canvas.paste(Image.fromarray(plane), (i * width, 30))
        draw.text((i * width + 7, 7), titles[i], fill='white')
    canvas.save(path)


def train(args):
    global SHAPE
    device, amp = device_and_amp(args)
    seed_all(args.seed)
    state = load_diffusion(args.resume) if args.resume else None
    if state:
        saved = state['args']
        for key in ('channels', 'dropout', 'timesteps', 'batch_size', 'accumulation', 'lr',
                    'weight_decay', 'seed', 'warmup', 'lr_horizon', 'ema_decay', 'gap'):
            setattr(args, key, saved[key])
        if args.steps < state['step']:
            raise ValueError(f'STEPS={args.steps} меньше сохранённого шага {state["step"]}')
    if getattr(args, 'dataset', ''):
        from .data import load_prepared
        data = load_prepared(args.dataset)
    else:
        data = load_zenodo_full(args.cache, gap=args.gap)
    SHAPE = tuple(data['images'].shape[-2:])
    metadata = json.loads(str(data['metadata_json']))
    if state and state['fingerprint'] != data['fingerprint']:
        raise ValueError('Данные/split/нормализация изменились; начните новый запуск.')
    train_ids = np.flatnonzero(data['split'] == 0)
    val_ids = np.flatnonzero(data['split'] == 1)
    model = DiffusionUNet(args.channels, args.dropout, gradient_checkpointing=True).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    sampler = Sampler(train_ids, args.seed)
    alphas = schedule(args.timesteps, device)
    run = Path(args.resume).resolve().parent if state else Path(args.runs) / ('run_' + timestamp())
    run.mkdir(parents=True, exist_ok=True)
    start, best, updates = 0, float('inf'), 0
    if state:
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        scaler.load_state_dict(state['scaler'])
        sampler.load(state['sampler'])
        start, best, updates = state['step'], state['best_val'], state['updates']
        restore_rng(state['rng'])
        print(f'Resume: {args.resume}; saved step={start}', flush=True)
    del state
    config = dict(format=FORMAT, prediction_type='v', zero_terminal_snr=True, args=vars(args),
                  real_metadata=metadata, fingerprint=data['fingerprint'], shape_hw=list(SHAPE),
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  code_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                  selection='fixed-noise validation velocity MSE; not realism or mask quality',
                  labels='U-Net++ pseudo-labels generated after diffusion training')
    write_json(run / ('resume_config.json' if args.resume else 'config.json'), config)
    print(f'Device: {device}; parameters={config["model_parameters"]:,}; '
          f'run={run}; steps={start}->{args.steps}; native shape={SHAPE}', flush=True)
    print(f'RUN_DIR={run}', flush=True)
    if start >= args.steps:
        return Path(args.resume)
    losses, interval_losses = [], []
    begin = time.monotonic()
    step = start
    completed_step = start
    fields = ['step', 'train_velocity_mse', 'val_velocity_mse', 'lr', 'updates', 'elapsed_seconds']

    def save(path, current_step):
        checkpoint = dict(format=FORMAT, step=current_step, updates=updates, model=model.state_dict(),
                          ema=ema.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          sampler=sampler.state_dict(), rng=rng_state(), best_val=best,
                          args=vars(args).copy(), model_config=model.config,
                          real_metadata=metadata, fingerprint=data['fingerprint'], prediction_type='v')
        temporary = run / (path + '.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, run / path)

    try:
        for step in range(start + 1, args.steps + 1):
            progress = min(1., max(0., (step - args.warmup) / max(1, args.lr_horizon - args.warmup)))
            rate = args.lr * min(1., step / max(1, args.warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(args.accumulation):
                clean = batch_tensor(data['images'], sampler.take(args.batch_size), device)
                noise = torch.randn_like(clean)
                timesteps = torch.randint(args.timesteps, (len(clean),), device=device)
                noisy, target = noisy_target(clean, noise, timesteps, alphas)
                with autocast(device, amp):
                    predicted = model(noisy, timesteps)
                    loss = F.mse_loss(predicted.float(), target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss; inspect input or try --no-amp')
                scaler.scale(loss / args.accumulation).backward()
                micro_losses.append(loss.item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                updates += 1
                decay = min(args.ema_decay, (1 + updates) / (10 + updates))
                with torch.no_grad():
                    for e, p in zip(ema.parameters(), model.parameters()):
                        e.lerp_(p.detach(), 1 - decay)
                    for e, p in zip(ema.buffers(), model.buffers()):
                        e.copy_(p)
            optimizer.zero_grad(set_to_none=True)
            del clean, noise, timesteps, noisy, target, predicted, loss
            losses.extend(micro_losses)
            interval_losses.extend(micro_losses)
            completed_step = step
            if step % 25 == 0 or step == start + 1:
                print(f'Step {step}/{args.steps}; v-MSE={np.mean(interval_losses):.5f}; lr={rate:.3g}; '
                      f'updates={updates}; elapsed={time.monotonic()-begin:.0f}s', flush=True)
                interval_losses.clear()
            if step % args.save_every == 0 or step == args.steps:
                val = validate(ema, data['images'], val_ids, alphas, device, amp, args.seed)
                improved = val < best
                best = min(best, val)
                save('last.pt', step)
                if improved:
                    save('best.pt', step)
                if step % args.snapshot_every == 0:
                    save(f'step_{step:06d}.pt', step)
                history = run / 'metrics.csv'
                exists = history.exists()
                with history.open('a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    if not exists:
                        writer.writeheader()
                    writer.writerow(dict(step=step, train_velocity_mse=float(np.mean(losses)),
                                         val_velocity_mse=val, lr=rate, updates=updates,
                                         elapsed_seconds=time.monotonic()-begin))
                losses.clear()
                print(f'Validation v-MSE={val:.6f}; checkpoint={run / "last.pt"}', flush=True)
            if step % args.preview_every == 0 or step == args.steps:
                print('Pure-noise EMA preview...', flush=True)
                real = data['images'][int(val_ids[0])].astype(np.float32) / 65535
                generated = [sample(ema, alphas, device, amp, args.seed + 70000 + k,
                                    args.sample_steps) for k in range(args.preview_count)]
                grid_preview(run / f'preview_{step:06d}.png', [real] + generated,
                             ['Real validation reference'] + [f'Pure noise seed {args.seed+70000+k}' for k in range(len(generated))])
                print(f'PREVIEW_FILE={run / f"preview_{step:06d}.png"}', flush=True)
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        # last.pt is only written at fully completed step boundaries. An interrupt
        # inside a microbatch must not relabel partially updated state as resumable.
        print(f'Interrupted. Resume from the last completed last.pt on Drive; current step={step}.', flush=True)
        raise
    finally:
        frames = data['images']
        if hasattr(frames, 'close'):
            frames.close()
        elif hasattr(frames.raw, 'close'):
            frames.raw.close()
    print(f'Training finished: {run}', flush=True)
    return run / 'last.pt'


def generate(args):
    global SHAPE
    device, amp = device_and_amp(args)
    if args.count < 1:
        raise ValueError('count must be positive')
    state = load_diffusion(args.checkpoint)
    SHAPE = tuple(state['real_metadata']['shape'][-2:])
    teacher = teacher_state(args.unet_checkpoint)
    model = DiffusionUNet(**state['model_config']).to(device).eval()
    model.load_state_dict(state['ema'])
    segmenter = UNetPlusPlus(teacher['base_channels']).to(device).eval().requires_grad_(False)
    segmenter.load_state_dict(teacher['model'])
    threshold = float(teacher['threshold']) if args.threshold is None else args.threshold
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between zero and one')
    settings = teacher['normalization']
    metadata, step = state['real_metadata'], state['step']
    alphas = schedule(state['args']['timesteps'], device)
    normal = metadata['normalization']
    root = Path(args.datasets) / ('dataset_' + timestamp())
    for directory in ('images_adu', 'images_preview', 'masks', 'probabilities', 'overlays', 'qa_previews', 'shards'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    manifest = dict(format='neural-naomi-static-dataset-v1', status='in_progress', requested_count=args.count,
                    generated_count=0, shape_hw=list(SHAPE), diffusion_checkpoint=str(Path(args.checkpoint).resolve()),
                    diffusion_sha256=digest(args.checkpoint), diffusion_step=step, generator_weights='EMA',
                    unet_checkpoint=str(Path(args.unet_checkpoint).resolve()), unet_sha256=digest(args.unet_checkpoint),
                    unet_threshold=threshold, unet_normalization=settings, unet_tta=args.tta,
                    diffusion_normalization=normal, real_source=metadata['source'], seed=args.seed,
                    sampling=dict(method='DDIM', steps=args.sample_steps, eta=args.eta, prediction_type='v',
                                  zero_terminal_snr=True, input='pure Gaussian noise; no source image'),
                    mask_kind='binary semantic U-Net++ pseudo-label; no instance IDs or manual ground truth',
                    filtering='none; empty/full/uncertain results are retained and flagged',
                    limitation='One real recording; independent-fish realism and segmentation quality unverified')
    write_json(root / 'manifest.json', manifest)
    del state, teacher
    gc.collect()
    fields = ['name', 'seed', 'image', 'mask', 'probability', 'foreground_fraction', 'mean_probability',
              'mean_binary_confidence', 'uncertain_fraction', 'saturated_fraction', 'flags']
    shard_images, shard_masks, shard_probs, shard_seeds = [], [], [], []
    rows = []
    with (root / 'pairs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(args.count):
            seed = args.seed + i
            print(f'Generate + segment {i+1}/{args.count}; seed={seed}', flush=True)
            unit = sample(model, alphas, device, amp, seed, args.sample_steps, args.eta)
            raw = np.rint(np.clip(normal['black'] + unit * (normal['white'] - normal['black']), 0, 65535)).astype(np.uint16)
            probability, mask = segment(segmenter, raw, settings, threshold, device, amp, args.tta)
            name = f'sample_{i:06d}_seed{seed}'
            display = normalize_image(raw, settings)
            preview = np.rint(display * 255).astype(np.uint8)
            probability_u16 = np.rint(probability * 65535).astype(np.uint16)
            Image.fromarray(raw).save(root / 'images_adu' / (name + '.tif'))
            Image.fromarray(preview).save(root / 'images_preview' / (name + '.png'))
            Image.fromarray(mask.astype(np.uint8) * 255).save(root / 'masks' / (name + '.png'))
            Image.fromarray(probability_u16).save(root / 'probabilities' / (name + '.png'))
            overlay = np.repeat(preview[..., None], 3, axis=-1)
            overlay[mask] = np.rint(.55 * overlay[mask] + .45 * np.array([0, 255, 90])).astype(np.uint8)
            Image.fromarray(overlay).save(root / 'overlays' / (name + '.png'))
            foreground = float(mask.mean())
            flags = []
            if foreground == 0:
                flags.append('empty_mask')
            if foreground > .9:
                flags.append('foreground_above_90_percent')
            uncertainty = float((np.abs(probability - threshold) < .1).mean())
            if uncertainty > .25:
                flags.append('many_probabilities_near_threshold')
            row = dict(name=name, seed=seed, image=f'images_adu/{name}.tif', mask=f'masks/{name}.png',
                       probability=f'probabilities/{name}.png', foreground_fraction=foreground,
                       mean_probability=float(probability.mean()),
                       mean_binary_confidence=float(np.maximum(probability, 1-probability).mean()),
                       uncertain_fraction=uncertainty, saturated_fraction=float((unit >= .9999).mean()),
                       flags=';'.join(flags))
            rows.append(row)
            writer.writerow(row)
            f.flush()
            shard_images.append(raw)
            shard_masks.append(mask)
            shard_probs.append(probability_u16)
            shard_seeds.append(seed)
            if i < args.qa_count:
                grid_preview(root / 'qa_previews' / (name + '.png'),
                             [display, mask.astype(np.float32), probability, display],
                             ['Generated image (U-Net display)', 'U-Net++ pseudo-mask', 'Foreground probability',
                              f'Overlay; threshold={threshold:.2f}'], mask=mask)
            if len(shard_images) == args.shard_size or i == args.count - 1:
                start = i + 1 - len(shard_images)
                shard_path = root / 'shards' / f'pairs_{start:06d}_{i+1:06d}.npz'
                np.savez_compressed(shard_path, images=np.stack(shard_images), masks=np.stack(shard_masks),
                                    probabilities_u16=np.stack(shard_probs), seeds=np.asarray(shard_seeds, dtype=np.int64),
                                    metadata_json=np.asarray(json.dumps(dict(mask_kind=manifest['mask_kind'],
                                    diffusion_normalization=normal, unet_normalization=settings, threshold=threshold))))
                shard_images.clear(); shard_masks.clear(); shard_probs.clear(); shard_seeds.clear()
            manifest['generated_count'] = i + 1
            write_json(root / 'manifest.json', manifest)
    manifest['status'] = 'complete'
    manifest['flagged_count'] = sum(bool(row['flags']) for row in rows)
    manifest['qa_note'] = 'Confidence/flags describe model outputs, not segmentation accuracy. No samples discarded.'
    write_json(root / 'manifest.json', manifest)
    if not args.no_zip:
        archive = root.with_suffix('.zip')
        print(f'Packing output: {archive}', flush=True)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(root))
    print(f'Dataset complete: {root}', flush=True)
    return root


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['train', 'generate', 'check-unet'], required=True)
    p.add_argument('--unet-checkpoint', default='weights/unetpp/best.pt')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--dataset', default='', help='Prepared dataset directory; otherwise use Zenodo')
    p.add_argument('--cache', default='/content/zenodo_zebrafish' if Path('/content').is_dir() else 'data/cache/zenodo')
    p.add_argument('--runs', default='outputs/diffusion_runs')
    p.add_argument('--datasets', default='outputs/generated')
    p.add_argument('--resume', default='')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--lr-horizon', type=int, default=20000)
    p.add_argument('--channels', type=int, default=48)
    p.add_argument('--dropout', type=float, default=.05)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--gap', type=int, default=32)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--snapshot-every', type=int, default=5000)
    p.add_argument('--preview-every', type=int, default=1000)
    p.add_argument('--preview-count', type=int, default=2)
    p.add_argument('--sample-steps', type=int, default=100)
    p.add_argument('--eta', type=float, default=0.)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--shard-size', type=int, default=25)
    p.add_argument('--qa-count', type=int, default=12)
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--no-zip', action='store_true')
    args = p.parse_args(argv)
    if any(getattr(args, key) < 1 for key in ('steps', 'lr_horizon', 'batch_size', 'accumulation',
           'save_every', 'snapshot_every', 'preview_every', 'shard_size')):
        p.error('Step, batch and interval arguments must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'check-unet':
        state = teacher_state(args.unet_checkpoint)
        print(json.dumps(dict(format=state['format'], base_channels=state['base_channels'],
                              threshold=state['threshold'], normalization=state['normalization'],
                              shape=state['native_shape_hw']), ensure_ascii=False, indent=2))
    elif args.mode == 'train':
        train(args)
    else:
        if not args.checkpoint:
            raise SystemExit('For generation set --checkpoint /.../last.pt')
        generate(args)
