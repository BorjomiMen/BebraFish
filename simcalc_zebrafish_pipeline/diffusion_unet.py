"""Small grayscale epsilon-prediction U-Net for full-frame microscopy DDPM.

No pretrained natural-image encoder, GAN discriminator, or segmentation model.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class TimeEmbedding(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(nn.Linear(channels, out_channels), nn.SiLU(),
                                 nn.Linear(out_channels, out_channels))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        freq = torch.exp(-math.log(10000) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
        phase = timesteps.float()[:, None] * freq[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        if embedding.shape[1] < self.channels:
            embedding = F.pad(embedding, (0, self.channels - embedding.shape[1]))
        return self.mlp(embedding)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, 2 * out_channels)
        self.norm2 = norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(temb)).to(h.dtype).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (h + self.skip(x)) / math.sqrt(2)


class BottleneckAttention(nn.Module):
    """Attention only at 1/8 resolution; PyTorch SDPA selects its backend."""
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.norm = norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            b, 3, self.heads, c // self.heads, h * w).unbind(1)
        attention = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        attention = attention.transpose(-1, -2).reshape(b, c, h, w)
        return (x + self.proj(attention)) / math.sqrt(2)


class DiffusionUNet(nn.Module):
    def __init__(self, channels: int = 32, dropout: float = 0.0,
                 attention: bool = True, gradient_checkpointing: bool = False):
        super().__init__()
        if channels < 8 or channels % 8:
            raise ValueError('channels must be a positive multiple of 8 (>=8)')
        self.config = dict(channels=channels, dropout=dropout, attention=attention)
        self.gradient_checkpointing = gradient_checkpointing
        widths = [channels, channels * 2, channels * 3, channels * 4]
        time_channels = channels * 4
        self.time = TimeEmbedding(channels, time_channels)
        self.input = nn.Conv2d(1, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        previous = widths[0]
        for i, width in enumerate(widths):
            self.down_blocks.append(nn.ModuleList([
                ResBlock(previous, width, time_channels, dropout),
                ResBlock(width, width, time_channels, dropout)]))
            if i < len(widths) - 1:
                self.downsample.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
            previous = width
        self.middle1 = ResBlock(widths[-1], widths[-1], time_channels, dropout)
        self.attention = BottleneckAttention(widths[-1]) if attention else nn.Identity()
        self.middle2 = ResBlock(widths[-1], widths[-1], time_channels, dropout)
        self.up_blocks = nn.ModuleList()
        for width in reversed(widths):
            self.up_blocks.append(nn.ModuleList([
                ResBlock(previous + width, width, time_channels, dropout),
                ResBlock(width, width, time_channels, dropout)]))
            previous = width
        self.output = nn.Sequential(norm(widths[0]), nn.SiLU(),
                                    nn.Conv2d(widths[0], 1, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _res(self, block: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, t, use_reentrant=False)
        return block(x, t)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('expected B x 1 x H x W grayscale tensor')
        height, width = x.shape[-2:]
        ph, pw = (-height) % 8, (-width) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        temb = self.time(timesteps)
        h = self.input(x)
        skips = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = self._res(block, h, temb)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        h = self._res(self.middle1, h, temb)
        if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
            h = checkpoint(self.attention, h, use_reentrant=False)
        else:
            h = self.attention(h)
        h = self._res(self.middle2, h, temb)
        for blocks in self.up_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = self._res(blocks[0], torch.cat((h, skip), dim=1), temb)
            h = self._res(blocks[1], h, temb)
        return self.output(h)[:, :, :height, :width]
