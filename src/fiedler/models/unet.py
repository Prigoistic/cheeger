"""From-scratch U-Net backbone.

A standard encoder/decoder with skip connections, written from first principles —
the commodity part of the system. It outputs a per-pixel **feature map** (not class
logits); a swappable head (`models/heads.py`) turns that into a segmentation. The
conv/BN/ReLU primitives are torch's optimized kernels (the "use the library where
it's a must for performance" call); the architecture is ours.

The encoder is intentionally small and dependency-free for local/toy work. For
competitive Cityscapes numbers, swap in an ImageNet-pretrained torchvision/timm
encoder later — the decoder and heads are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """(conv 3×3 → BN → ReLU) ×2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool downsample then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transposed-conv upsample, concat skip, DoubleConv. ``ch_in`` from the level
    below, ``ch_out`` = skip channels."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, 2, stride=2)
        self.conv = DoubleConv(ch_out * 2, ch_out)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # pad to the skip's spatial size (handles odd input dims)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """Encoder/decoder U-Net returning a feature map ``(B, out_channels, H, W)``."""

    def __init__(self, in_channels: int = 3, base: int = 32, depth: int = 3,
                 out_channels: int | None = None):
        super().__init__()
        self.out_channels = out_channels or base
        chs = [base * (2 ** i) for i in range(depth + 1)]   # e.g. depth=3 -> [b,2b,4b,8b]
        self.stem = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([Up(chs[i + 1], chs[i]) for i in reversed(range(depth))])
        self.proj = nn.Conv2d(chs[0], self.out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        h = skips[-1]
        for i, up in enumerate(self.ups):
            h = up(h, skips[-(i + 2)])
        return self.proj(h)
