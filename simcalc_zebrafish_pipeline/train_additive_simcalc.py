"""PyCharm Run: train a real highSNR diffusion prior for the final SimCalc dataset.

The first run downloads and verifies the full 3.9 GB Zenodo highSNR recording.
Training uses real frames only; the SimCalc frames provide fixed previews and
provenance checks. A later Run resumes the newest checkpoint in OUTPUT_DIR.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYNTHETIC_NPZ = ROOT / 'reports/hybrid_highSNR_dataset_1000_mild/hybrid_highSNR_1000.npz'
SYNTHETIC_CACHE = ROOT / 'work/additive_simcalc_cache'
ZENODO_CACHE = ROOT / 'data/zenodo_6293696/full_highSNR'
OUTPUT_DIR = ROOT / 'reports/additive_simcalc_training'

# Edit these settings in PyCharm, then click Run. No terminal arguments needed.
TRAIN_STEPS = 10_000
STEPS_THIS_RUN = 0  # 0 = continue to TRAIN_STEPS; e.g. 500 = stop after 500 new steps.
BATCH_SIZE = 1
ACCUMULATION = 4
CHANNELS = 32
SAVE_EVERY = 500
PREVIEW_EVERY = 500
SAMPLE_STEPS = 100
STRENGTH = 0.35
AUTO_RESUME = True
DEVICE = 'auto'  # auto: CUDA, then Apple MPS, then CPU; or set 'cpu'/'mps'/'cuda'.


def selected_device():
    if DEVICE != 'auto':
        return DEVICE
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def latest_checkpoint():
    candidates = sorted(OUTPUT_DIR.glob('run_*/last.pt'), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def main():
    if not SYNTHETIC_NPZ.is_file():
        raise FileNotFoundError(f'Generate the SimCalc dataset first: {SYNTHETIC_NPZ}')
    try:
        from train_simcalc_diffusion import main as train
    except ModuleNotFoundError as error:
        if error.name == 'torch':
            raise RuntimeError('Install requirements-additive.txt in the PyCharm interpreter first') from error
        raise

    args = [
        '--synthetic-npz', str(SYNTHETIC_NPZ),
        '--synthetic-cache', str(SYNTHETIC_CACHE),
        '--zenodo-cache', str(ZENODO_CACHE),
        '--output', str(OUTPUT_DIR),
        '--steps', str(TRAIN_STEPS),
        '--stop-after', str(STEPS_THIS_RUN),
        '--batch-size', str(BATCH_SIZE),
        '--accumulation', str(ACCUMULATION),
        '--channels', str(CHANNELS),
        '--save-every', str(SAVE_EVERY),
        '--preview-every', str(PREVIEW_EVERY),
        '--sample-steps', str(SAMPLE_STEPS),
        '--strength', str(STRENGTH),
        '--device', selected_device(),
        '--dashboard', 'none',
    ]
    checkpoint = latest_checkpoint() if AUTO_RESUME else None
    if checkpoint:
        args += ['--resume', str(checkpoint)]
        print(f'Resuming {checkpoint}', flush=True)
    else:
        print('Starting a new real-only diffusion training run', flush=True)
    return train(args)


if __name__ == '__main__':
    main()
