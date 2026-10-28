"""Conservative residual composition, mask QA, fixed controls, sharded export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_sampling import cosine_schedule, sdedit
from diffusion_unet import DiffusionUNet
from simcalc_data import digest, json_write, load_simcalc, synthetic_view


def as_tensor(image, device):
    return torch.from_numpy(image.astype(np.float32) / 32767.5 - 1)[None, None].to(device)


def smooth(x, kernel):
    return F.avg_pool2d(F.pad(x, (kernel//2,) * 4, mode='replicate'), kernel, 1)


def compose(raw, source, edited, labels, normalization, mix=.75, max_delta=.15, cell_mix=.80):
    """Residual in raw ADU; preserve intensities outside the normalization range.

    Exact uint16 identity for zero mix/strength. Mask attenuation is an explicit
    heuristic, not a learned residual distribution or a label guarantee.
    """
    scale = normalization['white'] - normalization['black']
    delta = (edited.float() - source.float()) * (.5 * scale * mix)
    delta = delta.clamp(-max_delta * scale, max_delta * scale)
    mask = torch.from_numpy((labels > 0).astype(np.float32))[None, None].to(source.device)
    mask = smooth(F.max_pool2d(mask, 3, 1, 1), 5).clamp(0, 1)
    delta = delta * (1 - (1-cell_mix) * mask)
    correction = delta[0, 0].cpu().numpy()
    output = np.rint(np.clip(raw.astype(np.float32) + correction, 0, 65535)).astype(np.uint16)
    residual = output.astype(np.int32) - raw.astype(np.int32)
    return output, residual


def correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator > 1e-12 else None


def mask_qa(raw, refined, labels, tissue, normalization, *, labels_available=True, tissue_available=True):
    scale = normalization['white'] - normalization['black']
    x = torch.from_numpy(raw.astype(np.float32))[None,None]
    y = torch.from_numpy(refined.astype(np.float32))[None,None]
    hx, hy = (x-smooth(x,5))[0,0].numpy(), (y-smooth(y,5))[0,0].numpy()
    valid = tissue if np.any(tissue) else np.ones(raw.shape, dtype=bool)
    corr = correlation(hx[valid], hy[valid])
    delta = refined.astype(np.float32)-raw.astype(np.float32)
    if not labels_available:
        return dict(tissue_highpass_correlation=corr,median_instance_contrast_relative_change=None,
                    positive_source_instances=None,nonpositive_output_instance_fraction=None,
                    instance_count=None,mean_abs_delta_adu=float(np.abs(delta).mean()),
                    max_abs_delta_adu=float(np.abs(delta).max()),review_required=True,
                    labels_available=False,tissue_available=tissue_available,
                    note='Instance masks absent: cell attenuation disabled and neuron preservation unassessed')
    # Compact indexing supports sparse uint32/uint64 IDs without allocating max(ID) bins.
    source_ids,ids = np.unique(np.asarray(labels).ravel(),return_inverse=True)
    count = np.bincount(ids)
    # Local-background-subtracted mean for EACH surviving source instance ID.
    cx = np.bincount(ids, weights=(x-smooth(x,9))[0,0].numpy().ravel(), minlength=len(count))
    cy = np.bincount(ids, weights=(y-smooth(y,9))[0,0].numpy().ravel(), minlength=len(count))
    present = (count > 0) & (source_ids > 0)
    means_x, means_y = cx[present]/count[present], cy[present]/count[present]
    # Ignore near-zero source contrast when computing fractional contrast changes.
    bright = means_x > .005 * scale
    drift = float(np.median(np.abs(means_y[bright]-means_x[bright])/np.maximum(means_x[bright], 1))) if bright.any() else None
    vanished = float(np.mean(means_y[bright] <= 0)) if bright.any() else None
    review = corr is None or corr < .98 or drift is None or drift > .2 or (vanished or 0) > .01
    return dict(tissue_highpass_correlation=corr, median_instance_contrast_relative_change=drift,
                positive_source_instances=int(bright.sum()), nonpositive_output_instance_fraction=vanished,
                instance_count=int(present.sum()), mean_abs_delta_adu=float(np.abs(delta).mean()),
                max_abs_delta_adu=float(np.abs(delta).max()), review_required=bool(review),
                labels_available=True,tissue_available=tissue_available,
                note='Heuristic preservation alarms; no realism score or proof of mask validity')


def show(rows, path, title, inline=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), len(rows[0]), squeeze=False, figsize=(3.2*len(rows[0]),3.2*len(rows)))
    delta_bound = max([1.] + [float(np.abs(item[0]).max()) for row in rows for item in row
                              if len(item) == 3 and item[2] == 'delta'])
    for axes_row, row in zip(axes, rows):
        for ax, item in zip(axes_row, row):
            image, caption = item[:2]
            if isinstance(image, torch.Tensor):
                image = image.detach().float().cpu().squeeze().numpy()
            if len(item) == 3 and item[2] == 'delta':
                artist = ax.imshow(image, cmap='coolwarm', vmin=-delta_bound, vmax=delta_bound)
                fig.colorbar(artist, ax=ax, fraction=.046, label='ADU')
            else:
                ax.imshow(image, cmap='gray', vmin=-1, vmax=1)
            ax.set_title(caption, fontsize=8)
            ax.axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    if inline:
        from IPython.display import Image, display
        display(Image(filename=str(path)))


@torch.no_grad()
def preview_simcalc(model, synthetic, real, alphas, device, amp, args, step, run):
    data = synthetic['simcalc']
    normalization = json.loads(str(real['metadata_json']))['normalization']
    indices = np.flatnonzero(synthetic['split'] == 1)[:args.preview_count]
    rows, sweep_rows, raw_rows, qa = [], [], [], []
    strengths = sorted(set(getattr(args, 'preview_strengths', [.15,.35,.55,.75]) + [args.strength]))
    for index in indices:
        source = as_tensor(synthetic['images'][index], device)
        if not torch.equal(sdedit(model, source, alphas, strength=0), source):
            raise AssertionError('SDEdit identity failed')
        sweep = [(source, 'SimCalc source; identity s=0 checked')]
        raw_row = [(source, 'SimCalc source; full frame')]
        for strength in strengths:
            edited = sdedit(model, source, alphas, strength=strength, sample_steps=args.sample_steps,
                            seed=args.seed+int(index), amp=amp, structure_guidance=args.structure_guidance)
            output, residual = compose(data['synthetic_images'][index], source, edited,
                                       data['synthetic_labels'][index], normalization,
                                       args.residual_mix, args.max_delta, args.cell_mix)
            normalized = (output.astype(np.float32)-normalization['black'])/(normalization['white']-normalization['black'])
            final_display = np.clip(normalized,0,1)*2-1
            sweep.append((final_display, f's={strength:.2f}; mix={args.residual_mix:.2f}; cell={args.cell_mix:.2f}'))
            raw_row.append((edited, f'Raw EMA SDEdit; s={strength:.2f}; DDIM {args.sample_steps}'))
            metrics = mask_qa(data['synthetic_images'][index], output, data['synthetic_labels'][index],
                              data['synthetic_tissue_masks'][index], normalization,
                              labels_available=data.get('labels_available',True),
                              tissue_available=data.get('tissue_available',True))
            raw_delta = (edited-source).abs().mean().item() * (normalization['white']-normalization['black'])/2
            qa.append(dict(index=int(index), strength=strength, raw_sdedit_mean_abs_delta_adu=raw_delta,
                           residual_mix=args.residual_mix, cell_mix=args.cell_mix, max_delta_fraction=args.max_delta, **metrics))
            if strength == args.strength:
                rows.append([(source, 'SimCalc source'),
                             (edited, f'Raw EMA SDEdit s={strength:.2f}'),
                             (final_display, f'Final addon; |delta| mean={metrics["mean_abs_delta_adu"]:.1f} ADU'),
                             (residual, 'Final signed residual; shared color scale', 'delta')])
        sweep_rows.append(sweep)
        raw_rows.append(raw_row)
    show(rows, run/f'preview_{step:06d}.png',
         'Source / raw SDEdit / applied addon / exact signed residual; full native frames', args.dashboard=='inline')
    show(sweep_rows, run/f'preview_sweep_{step:06d}.png',
         'Fixed source and seed; stronger appearance changes require neuron/mask review', args.dashboard=='inline')
    show(raw_rows, run/f'preview_raw_sweep_{step:06d}.png',
         f'Raw SDEdit at {args.sample_steps} DDIM steps; high strengths are diagnostics, not accepted labelled data',
         args.dashboard=='inline')
    json_write(run/f'preview_qa_{step:06d}.json', dict(step=step, parameters=dict(strength=args.strength,
               residual_mix=args.residual_mix, cell_mix=args.cell_mix, max_delta=args.max_delta,
               sample_steps=args.sample_steps), results=qa))


@torch.no_grad()
def diagnose(checkpoint, model, synthetic, real, alphas, device, amp, args, run):
    from diagnose_diffusion_refiner import sample_prior
    source_index = int(np.flatnonzero(synthetic['split']==1)[0])
    source = as_tensor(synthetic['images'][source_index], device)
    real_index = int(np.flatnonzero(real['split']==1)[0])
    real_source = as_tensor(real['images'][real_index], device)
    rows, records = [], []
    for kind in ['model', 'ema']:
        model.load_state_dict(checkpoint[kind])
        row = [(source, f'{kind}: SimCalc source')]
        for strength in [.10,.15,.25]:
            output = sdedit(model, source, alphas, strength=strength, sample_steps=args.sample_steps,
                             seed=args.seed+source_index, amp=amp)
            row.append((output, f'{kind}: raw SDEdit s={strength}'))
        self_edit = sdedit(model, real_source, alphas, strength=.15, sample_steps=args.sample_steps, seed=args.seed, amp=amp)
        row.extend([(real_source, 'Real validation input'), (self_edit, 'Real self-edit s=.15'),
                    (sample_prior(model, alphas, synthetic['images'].shape[1:], args.seed, args.sample_steps, amp=amp), 'Pure-noise prior')])
        mse = ((self_edit-real_source).square().mean()/4).item()
        records.append(dict(weights=kind, real_self_edit_psnr_db=float(-10*np.log10(max(mse,1e-12)))))
        rows.append(row)
    show(rows, run/'controls_current_ema.png', 'Same seeds; raw SDEdit controls before residual attenuation; PSNR is preservation only', args.dashboard=='inline')
    json_write(run/'controls_current_ema.json', dict(records=records, synthetic_index=source_index, real_val_index=real_index))


@torch.no_grad()
def export_main(argv=None):
    from train_simcalc_diffusion import read_checkpoint
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--synthetic-npz', required=True)
    p.add_argument('--synthetic-cache', default='/content/simcalc_cache')
    p.add_argument('--output', required=True)
    p.add_argument('--strength', type=float, default=.35)
    p.add_argument('--residual-mix', type=float, default=.75)
    p.add_argument('--max-delta', type=float, default=.15)
    p.add_argument('--cell-mix', type=float, default=.80)
    p.add_argument('--sample-steps', type=int, default=100)
    p.add_argument('--variants', type=int, default=1)
    p.add_argument('--shard-size', type=int, default=25)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--allow-new-synthetic', action='store_true',help='Apply the prior to another calibrated simulator input; training resume remains strict')
    args = p.parse_args(argv)
    if any(not 0<=v<=1 for v in [args.strength,args.residual_mix,args.max_delta,args.cell_mix]):
        p.error('strength/residual-mix/max-delta/cell-mix must be in [0,1]')
    if args.sample_steps<1 or args.variants<1 or args.shard_size<1 or args.limit<0:
        p.error('Invalid steps/count/shard size')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Choose an empty export folder to avoid mixing experiments')
    checkpoint = read_checkpoint(args.checkpoint)
    normalization = checkpoint['domain_metadata']['normalization']
    data = load_simcalc(args.synthetic_npz, args.synthetic_cache)
    same_input=checkpoint['data_sha256']['synthetic_npz_sha256']==data['source_sha256']
    if not same_input and not args.allow_new_synthetic:
        raise ValueError('Synthetic input differs; use --allow-new-synthetic only after explicit calibration/provenance review')
    view = synthetic_view(data, normalization)
    device = torch.device(args.device)
    amp = device.type=='cuda' and not args.no_amp
    model = DiffusionUNet(**checkpoint['model_config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['ema'])
    alphas = cosine_schedule(checkpoint['args']['timesteps']).to(device)
    count = min(args.limit or len(view['images']), len(view['images']))
    export_info = dict(format='simcalc-additive-adu-v1', checkpoint_sha256=digest(args.checkpoint),
                       checkpoint_step=checkpoint['step'], source_sha256=data['source_sha256'], normalization=normalization,
                       same_synthetic_as_training_run=same_input,
                       labels_available=data.get('labels_available',True),tissue_available=data.get('tissue_available',True),
                       parameters=vars(args), equation='refined_images = synthetic_images.astype(int32) + additive_residual_adu',
                       mask_status=('Copied bit-exact. Alignment/visibility after editing is not proven; inspect QA.'
                                    if data.get('labels_available',True)
                                    else 'No instance masks supplied; neuron preservation QA unavailable.'),
                       method='Real-only epsilon DDPM + SDEdit + bounded mask-attenuated residual (heuristic)',
                       source_metadata=data['metadata'])
    records, shard_names = [], []
    for variant in range(args.variants):
        for start in range(0,count,args.shard_size):
            stop = min(count,start+args.shard_size)
            outputs, residuals, seeds, alarms = [], [], [], []
            for index in range(start,stop):
                seed = args.seed+variant*len(view['images'])+index
                source = as_tensor(view['images'][index],device)
                edited = sdedit(model,source,alphas,strength=args.strength,sample_steps=args.sample_steps,seed=seed,amp=amp)
                raw, labels = data['synthetic_images'][index], data['synthetic_labels'][index]
                output, residual = compose(raw, source, edited, labels, normalization,
                                           args.residual_mix,args.max_delta,args.cell_mix)
                if not np.array_equal(raw.astype(np.int32)+residual,output.astype(np.int32)):
                    raise AssertionError('Residual reconstruction failed')
                qa = mask_qa(raw,output,labels,data['synthetic_tissue_masks'][index],normalization,
                             labels_available=data.get('labels_available',True),
                             tissue_available=data.get('tissue_available',True))
                records.append(dict(source_index=index,variant=variant,seed=seed,**qa))
                outputs.append(output); residuals.append(residual); seeds.append(seed); alarms.append(qa['review_required'])
            arrays = {key: np.asarray(value[start:stop] if value.ndim else value)
                      for key,value in data.items() if isinstance(value,np.ndarray)}
            if not data.get('source_profiles_available',True):
                arrays.pop('synthetic_profile_source_frame_indices',None)
            arrays.update(refined_images=np.stack(outputs), additive_residual_adu=np.stack(residuals),
                          source_indices=np.arange(start,stop), refinement_seeds=np.asarray(seeds,dtype=np.uint64),
                          review_required=np.asarray(alarms,dtype=bool), addon_metadata_json=np.asarray(json.dumps(export_info)))
            name=f'addon_v{variant:02d}_{start:05d}_{stop:05d}.npz'
            temporary=out/(name+'.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,**arrays)
            temporary.replace(out/name)
            shard_names.append(name)
            json_write(out/'export_progress.json',dict(complete=False,completed_shards=shard_names,total_completed=len(records)))
            print(f'Export {len(records)}/{count*args.variants}: {name}',flush=True)
    json_write(out/'qa.json',records)
    json_write(out/'manifest.json',dict(**export_info,complete=True,shards=shard_names,
               source_count=count,output_count=len(records),review_required_count=sum(r['review_required'] for r in records)))
    json_write(out/'export_progress.json',dict(complete=True,completed_shards=shard_names,total_completed=len(records)))
    print(f'Export complete: {out}; source and copied labels remain in each shard',flush=True)
    return out


if __name__=='__main__':
    export_main()
