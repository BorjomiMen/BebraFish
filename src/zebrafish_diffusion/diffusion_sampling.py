"""Cosine DDPM schedule and unpaired SDEdit/DDIM image refinement.

All schedule arithmetic is float32. Source tensors use [-1,1], shape [B,1,H,W].
The model predicts epsilon as model(noisy_image, integer_timestep_vector).
No synthetic/real pairs or external diffusion packages are required.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch.nn import functional as F


def cosine_schedule(timesteps: int = 1000) -> torch.Tensor:
    """Return cumulative alphas [T] for Improved-DDPM cosine betas.

    Index 0 is the FIRST noisy state, not clean x0. The clean sentinel is t=-1
    with cumulative alpha exactly 1. Betas are capped at 0.999.
    """
    if not isinstance(timesteps, int) or timesteps < 1:
        raise ValueError("timesteps must be a positive integer")
    points = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    curve = torch.cos((points + .008) / 1.008 * math.pi / 2).square()
    betas = (1 - curve[1:] / curve[:-1]).clamp(min=0., max=.999)
    return torch.cumprod(1 - betas, dim=0).to(torch.float32)


def _schedule(alpha_bar: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(alpha_bar, torch.Tensor) or alpha_bar.ndim != 1 or not len(alpha_bar):
        raise ValueError("alpha_bar must be a nonempty 1-D tensor")
    return alpha_bar.to(device=reference.device, dtype=torch.float32)


def q_sample(x0: torch.Tensor, t: int | torch.Tensor, noise: torch.Tensor,
             alpha_bar: torch.Tensor) -> torch.Tensor:
    """Sample q(x_t|x0) at exact schedule indices, using supplied noise."""
    if x0.shape != noise.shape or x0.ndim != 4:
        raise ValueError("x0 and noise must have equal [B,C,H,W] shapes")
    schedule = _schedule(alpha_bar, x0)
    if isinstance(t, int):
        indices = torch.full((len(x0),), t, device=x0.device, dtype=torch.long)
    elif isinstance(t, torch.Tensor):
        if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("timestep tensor must contain integers")
        indices = t.to(device=x0.device, dtype=torch.long)
        if indices.ndim == 0:
            indices = indices.expand(len(x0))
        if indices.shape != (len(x0),):
            raise ValueError("timestep vector must have one entry per image")
    else:
        raise TypeError("t must be an integer or integer tensor")
    if bool((indices < 0).any()) or bool((indices >= len(schedule)).any()):
        raise ValueError("q_sample timestep outside schedule")
    alpha = schedule[indices].reshape(-1, 1, 1, 1)
    return alpha.sqrt() * x0.float() + (1 - alpha).clamp_min(0).sqrt() * noise.float()


def _lowpass(image: torch.Tensor, kernel: int) -> torch.Tensor:
    if not isinstance(kernel, int) or kernel < 1 or kernel % 2 != 1:
        raise ValueError("structure_kernel must be a positive odd integer")
    padding = kernel // 2
    mode = "reflect" if padding < min(image.shape[-2:]) else "replicate"
    return F.avg_pool2d(F.pad(image, (padding,) * 4, mode=mode), kernel, 1)


def ddim_step(x_t: torch.Tensor, eps: torch.Tensor, t: int, next_t: int,
              alpha_bar: torch.Tensor, source: torch.Tensor | None = None,
              structure_guidance: float = 0., structure_kernel: int = 31) -> torch.Tensor:
    """Deterministic DDIM transition from t to an explicit smaller next_t.

    next_t=-1 is clean output. Clip x0 and recompute epsilon so clipping and
    optional low-frequency x0 guidance remain consistent with the direction.
    Guidance is a declared heuristic; it is not canonical unguided SDEdit.
    """
    if x_t.shape != eps.shape or x_t.ndim != 4:
        raise ValueError("x_t and epsilon must have equal [B,C,H,W] shapes")
    if not isinstance(t, int) or not isinstance(next_t, int):
        raise TypeError("DDIM indices must be scalar integers")
    schedule = _schedule(alpha_bar, x_t)
    if not 0 <= t < len(schedule) or not -1 <= next_t < t:
        raise ValueError("Expected 0<=t<T and -1<=next_t<t")
    if not math.isfinite(structure_guidance) or not 0 <= structure_guidance <= 1:
        raise ValueError("structure_guidance must be in [0,1]")
    x_t, eps = x_t.float(), eps.float()
    alpha = schedule[t]
    next_alpha = schedule[next_t] if next_t >= 0 else x_t.new_tensor(1.)
    sigma = (1 - alpha).clamp_min(1e-12).sqrt()
    x0 = ((x_t - sigma * eps) / alpha.clamp_min(1e-12).sqrt()).clamp(-1, 1)
    if structure_guidance:
        if source is None or source.shape != x_t.shape:
            raise ValueError("Guidance needs the source with matching image geometry")
        x0 = (x0 + structure_guidance *
              (_lowpass(source.float(), structure_kernel) - _lowpass(x0, structure_kernel))).clamp(-1, 1)
    consistent_eps = (x_t - alpha.sqrt() * x0) / sigma
    return next_alpha.sqrt() * x0 + (1 - next_alpha).clamp_min(0).sqrt() * consistent_eps


@torch.no_grad()
def sdedit(model, source: torch.Tensor, alpha_bar: torch.Tensor,
           strength: float = .35, sample_steps: int = 40, seed: int = 42,
           structure_guidance: float = 0., amp: bool = False,
           callback: Callable[[int, int], None] | None = None,
           structure_kernel: int = 31) -> torch.Tensor:
    """Refine full frames through a real-domain DDPM prior using DDIM.

    Padding is only on right/bottom, to multiples of 8; output is unpadded to
    the exact input geometry. Eta=0 is deterministic given the initial noise.
    strength=0 is an exact identity and does not call or change the model.
    """
    if source.ndim != 4 or source.shape[1] != 1 or not len(source):
        raise ValueError("Expected a nonempty grayscale batch [B,1,H,W]")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    if not isinstance(sample_steps, int) or sample_steps < 1:
        raise ValueError("sample_steps must be a positive integer")
    if not math.isfinite(structure_guidance) or not 0 <= structure_guidance <= 1:
        raise ValueError("structure_guidance must be in [0,1]")
    if strength == 0:
        return source.clone()
    original_h, original_w = source.shape[-2:]
    pad_h, pad_w = (-original_h) % 8, (-original_w) % 8
    x0 = source.float()
    if pad_h or pad_w:
        x0 = F.pad(x0, (0, pad_w, 0, pad_h), mode="replicate")
    schedule = _schedule(alpha_bar, x0)
    start = int(round(strength * (len(schedule) - 1)))
    count = min(sample_steps, start + 1)
    grid = torch.linspace(start, 0, count, dtype=torch.float64).round().long().unique_consecutive().tolist()
    generator = torch.Generator(device=x0.device).manual_seed(seed)
    noise = torch.randn(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
    x = q_sample(x0, grid[0], noise, schedule)
    was_training = model.training
    model.eval()
    try:
        for index, timestep in enumerate(grid):
            times = torch.full((len(x),), timestep, device=x.device, dtype=torch.long)
            with torch.autocast(device_type=x.device.type, dtype=torch.float16,
                                enabled=bool(amp and x.device.type == "cuda")):
                prediction = model(x, times)
            following = grid[index + 1] if index + 1 < len(grid) else -1
            x = ddim_step(x, prediction, timestep, following, schedule,
                          source=x0, structure_guidance=structure_guidance,
                          structure_kernel=structure_kernel)
            if callback is not None:
                callback(index + 1, len(grid))
    finally:
        model.train(was_training)
    return x[..., :original_h, :original_w].clamp(-1, 1)
