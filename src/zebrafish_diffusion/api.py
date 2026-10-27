"""Public inference API. Models are loaded once and reused across seeds."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import tifffile
from . import diffusion as dd
from .unetpp import UNetPlusPlus, normalize_image


def decode(unit, normalization, dtype):
    dtype = np.dtype(dtype)
    raw = normalization['black'] + unit*(normalization['white']-normalization['black'])
    if dtype.kind in 'iu':
        info = np.iinfo(dtype)
        raw = np.rint(np.clip(raw, info.min, info.max))
    return raw.astype(dtype)


class Pipeline:
    def __init__(self, diffusion_checkpoint, unet_checkpoint, device='auto', amp=True, threshold=None, threads=None):
        if threads:
            torch.set_num_threads(int(threads))
        self.device, self.amp = dd.device_and_amp(SimpleNamespace(device=device, no_amp=not amp))
        state = dd.load_diffusion(diffusion_checkpoint)
        self.diffusion_checkpoint = str(Path(diffusion_checkpoint).resolve())
        self.step = int(state['step'])
        self.real_metadata = state['real_metadata']
        self.shape = tuple(self.real_metadata['shape'][-2:])
        self.normalization = self.real_metadata['normalization']
        self.raw_dtype = self.real_metadata.get('raw_dtype', 'uint16')
        self.model = dd.DiffusionUNet(**state['model_config']).to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(state['ema'])
        self.alphas = dd.schedule(state['args']['timesteps'], self.device)
        del state
        teacher = dd.teacher_state(unet_checkpoint)
        self.unet_checkpoint = str(Path(unet_checkpoint).resolve())
        self.segmenter = UNetPlusPlus(teacher['base_channels']).to(self.device).eval().requires_grad_(False)
        self.segmenter.load_state_dict(teacher['model'])
        self.unet_normalization = teacher['normalization']
        self.threshold = float(teacher['threshold']) if threshold is None else float(threshold)
        if not 0 < self.threshold < 1:
            raise ValueError('threshold must be between 0 and 1')
        del teacher

    def generate(self, seed=42, steps=100, eta=0., tta=False, verbose=True):
        calls = [0]
        def progress(module, inputs, output):
            calls[0] += 1
            if calls[0] == 1 or calls[0] % 10 == 0 or calls[0] == steps:
                print(f'DDIM {calls[0]}/{steps}', flush=True)
        hook = self.model.register_forward_hook(progress) if verbose else None
        try:
            unit = dd.sample(self.model, self.alphas, self.device, self.amp, int(seed), steps, eta, self.shape)
        finally:
            if hook:
                hook.remove()
        image = decode(unit, self.normalization, self.raw_dtype)
        probability, mask = dd.segment(self.segmenter, image, self.unet_normalization,
                                       self.threshold, self.device, self.amp, tta)
        return dict(image=image, mask=mask, probability=probability, seed=int(seed), steps=steps,
                    eta=eta, tta=tta)

    def save(self, result, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        stem = f'seed_{result["seed"]}_{dd.timestamp()}'
        paths = save_segmentation(result['image'], result['mask'], result['probability'], output,
                                  stem, self.unet_normalization, self.threshold)
        metadata = dict(seed=result['seed'], steps=result['steps'], eta=result['eta'],
                        device=str(self.device), shape_hw=list(self.shape),
                        diffusion_checkpoint=self.diffusion_checkpoint, diffusion_step=self.step,
                        diffusion_weights='EMA', unet_checkpoint=self.unet_checkpoint,
                        diffusion_normalization=self.normalization, unet_normalization=self.unet_normalization,
                        threshold=self.threshold, unet_tta=result['tta'], files=paths)
        dd.write_json(output/f'{stem}_metadata.json', metadata)
        return paths


def save_segmentation(image, mask, probability, output, stem, normalization, threshold):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key:str((output/f'{stem}_{suffix}').resolve()) for key,suffix in (
        ('image','image.tif'),('mask','mask.png'),('probability','probability.png'),
        ('overlay','overlay.png'),('preview','preview.png'))}
    tifffile.imwrite(paths['image'], image)
    Image.fromarray(mask.astype(np.uint8)*255).save(paths['mask'])
    Image.fromarray(np.rint(np.clip(probability,0,1)*65535).astype(np.uint16)).save(paths['probability'])
    shown = normalize_image(image, normalization)
    rgb = np.repeat(np.rint(shown[...,None]*255).astype(np.uint8),3,axis=-1)
    rgb[mask] = np.rint(.55*rgb[mask]+.45*np.array([0,255,90])).astype(np.uint8)
    Image.fromarray(rgb).save(paths['overlay'])
    dd.grid_preview(paths['preview'], [shown, mask.astype(np.float32), probability, shown],
                    ['Generated / input image', 'Binary mask', 'Foreground probability',
                     f'Overlay; threshold={threshold:.2f}'], mask=mask)
    return paths


def segment_file(image_path, unet_checkpoint, output='outputs/segmentation', device='auto',
                 threshold=None, tta=False, amp=True):
    from .data import plane
    selected, mixed = dd.device_and_amp(SimpleNamespace(device=device, no_amp=not amp))
    state = dd.teacher_state(unet_checkpoint)
    model = UNetPlusPlus(state['base_channels']).to(selected).eval().requires_grad_(False)
    model.load_state_dict(state['model'])
    threshold = float(state['threshold']) if threshold is None else float(threshold)
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    image = plane(image_path)
    probability, mask = dd.segment(model, image, state['normalization'], threshold, selected, mixed, tta)
    stem = Path(image_path).stem+'_'+dd.timestamp()
    paths = save_segmentation(image,mask,probability,output,stem,state['normalization'],threshold)
    dd.write_json(Path(output)/f'{stem}_metadata.json',dict(input=str(Path(image_path).resolve()),
        checkpoint=str(Path(unet_checkpoint).resolve()), threshold=threshold,
        normalization=state['normalization'], device=str(selected), tta=tta, files=paths))
    return paths
