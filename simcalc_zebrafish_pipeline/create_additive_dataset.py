"""Create an exact-size SimCalc dataset with a visible diffusion correction.

Run in PyCharm, or: python create_additive_dataset.py --count 1000
Requires the original SimCalc NPZ and an already trained additive checkpoint.
Existing training, generation, refinement scripts and README are not modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_COUNT = 500                       # PyCharm Run: change to 1000 if needed.
STRENGTH_RANGE = (0.55, 0.75)            # Deliberately a code setting, not a CLI option.
SAMPLE_STEPS = 100
RESIDUAL_MIX = 1.0                      # Apply the full SDEdit correction.
CELL_MIX = 1.0                          # No additional attenuation inside masks.
MAX_DELTA = 1.0                         # No 15%-of-normalization-range clipping.
SHARD_SIZE = 25
PREVIEW_COUNT = 2
SEED = 20261027
DEVICE = 'auto'                         # 'auto', 'cuda' or 'cpu'.


def sampling_plan(source_count, count):
    """One output per distinct selected source; a stable prefix for any count."""
    if count < 1 or source_count < count:
        raise ValueError(f'Requested {count} images, but input contains {source_count}. '
                         'Choose a smaller count or provide more SimCalc source images.')
    low, high = STRENGTH_RANGE
    if not 0 <= low <= high <= 1:
        raise ValueError('STRENGTH_RANGE must satisfy 0 <= low <= high <= 1')
    indices = np.random.default_rng(SEED).permutation(source_count)[:count]
    # Generate the full strength plan so a source keeps the same s when count changes.
    by_source = np.random.default_rng(SEED + 1).uniform(low, high, source_count)
    strengths = by_source[indices]
    seeds = np.asarray([SEED + int(index) for index in indices], dtype=np.uint64)
    return indices, strengths, seeds


def checkpoint_path(explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        folder = ROOT / 'reports/additive_simcalc_training'
        candidates = list(folder.glob('run_*/last.pt'))
        if not candidates:
            raise FileNotFoundError('No trained additive checkpoint found. Supply '
                                    '--checkpoint PATH_TO_ADDITIVE_LAST_PT; the main '
                                    'v-prediction diffusion checkpoint is incompatible.')
        path = max(candidates, key=lambda item: item.stat().st_mtime)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def raw_sdedit_adu(edited, normalization):
    unit = (edited[0, 0].detach().float().cpu().numpy() + 1.0) * 0.5
    adu = normalization['black'] + unit * (normalization['white'] - normalization['black'])
    return np.rint(np.clip(adu, 0, 65535)).astype(np.uint16)


def save_comparison(rows, output, normalization):
    """Same display limits for source/raw/final, one shared signed-residual scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 4, squeeze=False,
                             figsize=(17, 4 * len(rows)), constrained_layout=True)
    bound = max(1.0, max(float(np.abs(row['residual']).max()) for row in rows))
    for axes_row, row in zip(axes, rows):
        images = [row['source'], row['raw_sdedit'], row['refined']]
        titles = ['SimCalc source', f'Raw EMA SDEdit s={row["strength"]:.3f}',
                  f'Final addon; |delta| mean={np.abs(row["residual"]).mean():.1f} ADU']
        for ax, image, title in zip(axes_row[:3], images, titles):
            ax.imshow(image, cmap='gray', vmin=normalization['black'], vmax=normalization['white'])
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        artist = axes_row[3].imshow(row['residual'], cmap='coolwarm', vmin=-bound, vmax=bound)
        axes_row[3].set_title('Final signed residual; shared scale', fontsize=10)
        axes_row[3].axis('off')
    fig.colorbar(artist, ax=list(axes[:, 3]), label='ADU', shrink=0.85)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=IMAGE_COUNT, help='Exact number of refined output images')
    p.add_argument('--synthetic-npz', type=Path,
                   default=ROOT / 'reports/hybrid_highSNR_dataset_1000_mild/hybrid_highSNR_1000.npz')
    p.add_argument('--checkpoint', type=Path, help='Trained SimCalc additive checkpoint; default: newest last.pt')
    p.add_argument('--output', type=Path, help='New output folder; default: a unique folder under reports/')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.count < 1:
        raise ValueError('--count must be positive')
    if SHARD_SIZE < 1 or PREVIEW_COUNT < 1 or SAMPLE_STEPS < 1:
        raise ValueError('SHARD_SIZE, PREVIEW_COUNT and SAMPLE_STEPS must be positive')
    if any(not 0 <= value <= 1 for value in (RESIDUAL_MIX, CELL_MIX, MAX_DELTA)):
        raise ValueError('Composition settings must lie in [0, 1]')
    source_path = args.synthetic_npz.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f'SimCalc input not found: {source_path}. Use --synthetic-npz PATH.')
    checkpoint_file = checkpoint_path(args.checkpoint)
    output = args.output.expanduser().resolve() if args.output else (
        ROOT / 'reports' / f'additive_dataset_{args.count}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new folder: {output}')

    sys.path.insert(0, str(ROOT))
    import torch
    import tifffile
    from diffusion_sampling import cosine_schedule, sdedit
    from diffusion_unet import DiffusionUNet
    from simcalc_addon import as_tensor, compose, mask_qa
    from simcalc_data import digest, json_write, load_simcalc, synthetic_view
    from train_simcalc_diffusion import read_checkpoint

    checkpoint = read_checkpoint(checkpoint_file)
    normalization = checkpoint['domain_metadata']['normalization']
    data = load_simcalc(source_path, ROOT / 'work/additive_dataset_cache')
    if tuple(data['synthetic_images'].shape[1:]) != (400, 485):
        raise ValueError('This export preserves native 400x485 SimCalc images; resizing is disabled')
    if not data.get('labels_available', True) or not data.get('tissue_available', True):
        raise ValueError('This dataset export requires the original instance and tissue masks')
    if checkpoint['data_sha256']['synthetic_npz_sha256'] != data['source_sha256']:
        raise ValueError('Checkpoint and source dataset differ; use the original calibrated SimCalc input')
    indices, strengths, seeds = sampling_plan(len(data['synthetic_images']), args.count)
    view = synthetic_view(data, normalization)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE == 'auto' else DEVICE)
    if device.type == 'cpu':
        torch.set_num_threads(min(8, torch.get_num_threads()))
    amp = device.type == 'cuda'
    model = DiffusionUNet(**checkpoint['model_config']).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['ema'])
    alphas = cosine_schedule(checkpoint['args']['timesteps']).to(device)

    # All persistent output is in a new directory; the input NPZ and checkpoint are read-only.
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('images', 'masks', 'shards'):
        (output / folder).mkdir()
    info = dict(format='simcalc-additive-adu-v1', complete=False, requested_count=args.count,
                source_count=len(data['synthetic_images']), source_sha256=data['source_sha256'],
                source_path=str(source_path), checkpoint_path=str(checkpoint_file),
                checkpoint_sha256=digest(checkpoint_file), checkpoint_step=checkpoint['step'],
                normalization=normalization, shape_yx=[400, 485], strength_range=list(STRENGTH_RANGE),
                sample_steps=SAMPLE_STEPS, residual_mix=RESIDUAL_MIX, cell_mix=CELL_MIX,
                max_delta_fraction=MAX_DELTA, seed=SEED, labels_available=True, tissue_available=True,
                equation='refined_images.astype(int32) = synthetic_images.astype(int32) + additive_residual_adu',
                selection='Seeded selection without replacement; one refined output per source',
                mask_status='Original masks copied exactly; their post-edit alignment requires review',
                composition_note='Full SDEdit delta; original intensities outside the normalization range are retained',
                acceptance_note='QA flags are retained; count includes images requiring review',
                source_metadata=data['metadata'], source_indices=indices.tolist(),
                refinement_strengths=strengths.tolist(), refinement_seeds=seeds.tolist(), shards=[])
    json_write(output / 'manifest.json', info)
    qa_records, preview_rows = [], []

    @torch.no_grad()
    def render(source_index, strength, seed):
        raw = np.asarray(data['synthetic_images'][source_index])
        source = as_tensor(view['images'][source_index], device)
        edited = sdedit(model, source, alphas, strength=float(strength), sample_steps=SAMPLE_STEPS,
                        seed=int(seed), amp=amp)
        final, residual = compose(raw, source, edited, data['synthetic_labels'][source_index],
                                  normalization, RESIDUAL_MIX, MAX_DELTA, CELL_MIX)
        np.testing.assert_array_equal(raw.astype(np.int32) + residual, final.astype(np.int32))
        return dict(source=raw.copy(), raw_sdedit=raw_sdedit_adu(edited, normalization),
                    refined=final, residual=residual, strength=float(strength))

    fields = ['name', 'image', 'mask', 'group', 'source_index', 'seed', 'strength', 'review_required']
    print(f'Creating exactly {args.count} refined images, 400x485; s={STRENGTH_RANGE}; device={device}', flush=True)
    print(f'Output: {output}', flush=True)
    with (output / 'pairs.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for start in range(0, args.count, SHARD_SIZE):
            stop = min(start + SHARD_SIZE, args.count)
            batch_indices = indices[start:stop]
            raw_edits, finals, residuals, alarms = [], [], [], []
            for position in range(start, stop):
                source_index = int(indices[position])
                row = render(source_index, strengths[position], seeds[position])
                labels = np.asarray(data['synthetic_labels'][source_index])
                qa = mask_qa(row['source'], row['refined'], labels,
                             data['synthetic_tissue_masks'][source_index], normalization)
                name = f'frame_{position:06d}'
                image_name, mask_name = f'images/{name}.tif', f'masks/{name}.tif'
                tifffile.imwrite(output / image_name, row['refined'], photometric='minisblack')
                tifffile.imwrite(output / mask_name, labels, photometric='minisblack')
                original_seeds = data.get('synthetic_seeds')
                group = (str(int(original_seeds[source_index])) if original_seeds is not None
                         else f'{data["source_sha256"][:16]}:{source_index}')
                writer.writerow(dict(name=name, image=image_name, mask=mask_name,
                                     group=group,
                                     source_index=source_index, seed=int(seeds[position]),
                                     strength=float(strengths[position]), review_required=qa['review_required']))
                qa_records.append(dict(output_index=position, source_index=source_index,
                                       strength=float(strengths[position]), seed=int(seeds[position]), **qa))
                raw_edits.append(row['raw_sdedit']); finals.append(row['refined'])
                residuals.append(row['residual']); alarms.append(qa['review_required'])
                if len(preview_rows) < PREVIEW_COUNT:
                    preview_rows.append(row)
            stream.flush()
            arrays = {key: np.asarray(value[batch_indices] if value.ndim else value)
                      for key, value in data.items() if isinstance(value, np.ndarray)}
            if not data.get('source_profiles_available', True):
                arrays.pop('synthetic_profile_source_frame_indices', None)
            arrays.update(raw_sdedit_images=np.stack(raw_edits), refined_images=np.stack(finals),
                          additive_residual_adu=np.stack(residuals).astype(np.int32, copy=False),
                          source_indices=batch_indices, refinement_strengths=strengths[start:stop],
                          refinement_seeds=seeds[start:stop], review_required=np.asarray(alarms, dtype=bool),
                          addon_metadata_json=np.asarray(json.dumps({
                              'format': info['format'], 'normalization': normalization,
                              'strength_range': info['strength_range'], 'source_sha256': data['source_sha256'],
                              'checkpoint_sha256': info['checkpoint_sha256'], 'source_metadata': data['metadata']})))
            relative = f'shards/addon_{start:05d}_{stop:05d}.npz'
            temporary = output / (relative + '.partial')
            with temporary.open('wb') as shard:
                np.savez_compressed(shard, **arrays)
            temporary.replace(output / relative)
            info['shards'].append(relative)
            info['output_count'] = stop
            json_write(output / 'manifest.json', info)
            json_write(output / 'qa.json', qa_records)
            print(f'Exported {stop}/{args.count}', flush=True)
            if start == 0:
                save_comparison(preview_rows, output / 'comparison.png', normalization)
                for strength in sorted(set(STRENGTH_RANGE)):
                    rows = [render(int(indices[i]), strength, seeds[i]) for i in range(len(preview_rows))]
                    save_comparison(rows, output / f'comparison_s{strength:.2f}.png', normalization)

    if len(list((output / 'images').glob('*.tif'))) != args.count:
        raise AssertionError('Image count differs from the requested count')
    if len(list((output / 'masks').glob('*.tif'))) != args.count:
        raise AssertionError('Mask count differs from the requested count')
    info.update(complete=True, output_count=args.count,
                review_required_count=sum(bool(row['review_required']) for row in qa_records))
    json_write(output / 'manifest.json', info)
    print(f'Done: {args.count} images in {output / "images"}; comparison.png shows actual saved outputs.', flush=True)
    return output


if __name__ == '__main__':
    main()
