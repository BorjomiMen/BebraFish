"""PyCharm Run: add the trained diffusion correction to SimCalc images.

The source image, refined image, exact signed ADU residual and original labels
are saved in NPZ shards. Edit LIMIT=25 for a short preview; 0 exports all 1000.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train_additive_simcalc import (ROOT, SYNTHETIC_NPZ, SYNTHETIC_CACHE,
                                    OUTPUT_DIR as TRAINING_DIR, selected_device)


OUTPUT_DIR = ROOT / 'reports/additive_simcalc_refined_1000'
CHECKPOINT = None  # None = newest last.pt in TRAINING_DIR; or put an exact local path.
LIMIT = 0  # 0 = all images; 25 = one trial shard.
VARIANTS = 1
SHARD_SIZE = 25
STRENGTH = 0.35
SAMPLE_STEPS = 100
RESIDUAL_MIX = 0.75
CELL_MIX = 0.80
MAX_DELTA = 0.15
SEED = 20261027


def checkpoint_path():
    if CHECKPOINT is not None:
        path = Path(CHECKPOINT)
    else:
        candidates = sorted(TRAINING_DIR.glob('run_*/last.pt'),
                            key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f'No trained checkpoint in {TRAINING_DIR}; run train_additive_simcalc.py first')
        path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_first_comparison(output_dir):
    manifest = json.loads((output_dir / 'manifest.json').read_text())
    first = output_dir / manifest['shards'][0]
    with np.load(first, allow_pickle=False) as data:
        original = data['synthetic_images'][0].astype(np.float32)
        refined = data['refined_images'][0].astype(np.float32)
        residual = data['additive_residual_adu'][0].astype(np.float32)
        labels = data['synthetic_labels'][0] > 0
    norm = manifest['normalization']
    low, high = norm['black'], norm['white']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, image, title in zip(axes, (original, refined, labels),
                                ('SimCalc source', 'With additive correction', 'Original neuron mask')):
        if image.dtype == bool:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(image, cmap='gray', vmin=low, vmax=high)
        ax.set_title(title)
        ax.axis('off')
    bound = max(1, float(np.max(np.abs(residual))))
    axes[3].imshow(residual, cmap='coolwarm', vmin=-bound, vmax=bound)
    axes[3].set_title('Signed correction (ADU)')
    axes[3].axis('off')
    path = output_dir / 'first_comparison.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'Comparison: {path}', flush=True)


def main():
    if not SYNTHETIC_NPZ.is_file():
        raise FileNotFoundError(SYNTHETIC_NPZ)
    try:
        from simcalc_addon import export_main
    except ModuleNotFoundError as error:
        if error.name == 'torch':
            raise RuntimeError('Install requirements-additive.txt in the PyCharm interpreter first') from error
        raise
    checkpoint = checkpoint_path()
    args = [
        '--checkpoint', str(checkpoint),
        '--synthetic-npz', str(SYNTHETIC_NPZ),
        '--synthetic-cache', str(SYNTHETIC_CACHE),
        '--output', str(OUTPUT_DIR),
        '--limit', str(LIMIT),
        '--variants', str(VARIANTS),
        '--shard-size', str(SHARD_SIZE),
        '--strength', str(STRENGTH),
        '--sample-steps', str(SAMPLE_STEPS),
        '--residual-mix', str(RESIDUAL_MIX),
        '--cell-mix', str(CELL_MIX),
        '--max-delta', str(MAX_DELTA),
        '--seed', str(SEED),
        '--device', selected_device(),
    ]
    output = export_main(args)
    save_first_comparison(output)
    return output


if __name__ == '__main__':
    main()
