"""Synthetic driving scenes — a tiny dataset for fast local iteration.

Generates layered road scenes (sky, buildings, road) with a few objects (cars,
poles, people, vegetation) drawn at random positions. Labels use Cityscapes
trainIds so the toy set is a drop-in for the real loader, the same palette, metrics
and heads apply. Each sample is deterministic in its index, so the set is
reproducible and a single sample can be overfit for a sanity check.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..metrics.segmentation import CITYSCAPES_CLASSES  # names only

# trainIds we draw with (subset of the 19 classes)
SKY, BUILDING, ROAD, VEG, CAR, POLE, PERSON = 10, 2, 0, 8, 13, 5, 11

# class colours for the rendered image (reuse the canonical palette)
_RGB = torch.tensor([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=torch.float32) / 255.0


def _rect(label, cls, y0, y1, x0, x1):
    label[y0:y1, x0:x1] = cls


def make_toy_scene(H: int = 128, W: int = 256, seed: int = 0,
                   noise: float = 0.12) -> tuple[Tensor, Tensor]:
    """Return (image (3,H,W) float in [0,1], label (H,W) long trainIds)."""
    g = torch.Generator().manual_seed(seed)
    rint = lambda lo, hi: int(torch.randint(lo, hi, (), generator=g))
    label = torch.empty(H, W, dtype=torch.long)

    horizon = int(H * (0.32 + 0.06 * torch.rand((), generator=g)))
    road_top = int(H * (0.58 + 0.06 * torch.rand((), generator=g)))
    label[:horizon] = SKY
    label[horizon:road_top] = BUILDING
    label[road_top:] = ROAD

    # vegetation clumps along the horizon
    for _ in range(rint(2, 5)):
        x = rint(0, W - 30)
        _rect(label, VEG, horizon - rint(6, 18), horizon + rint(2, 10), x, x + rint(16, 40))
    # poles rising from the roadside
    for _ in range(rint(1, 4)):
        x = rint(0, W - 2)
        _rect(label, POLE, horizon, road_top + rint(0, 20), x, x + rint(1, 3))
    # cars on the road
    for _ in range(rint(1, 4)):
        cw, ch = rint(18, 40), rint(10, 22)
        x, y = rint(0, max(1, W - cw)), rint(road_top, max(road_top + 1, H - ch))
        _rect(label, CAR, y, y + ch, x, x + cw)
    # an occasional person
    if torch.rand((), generator=g) < 0.5:
        x = rint(0, W - 6)
        _rect(label, PERSON, road_top - rint(8, 20), road_top + rint(2, 8), x, x + rint(3, 7))

    img = _RGB[label] + noise * torch.randn(H, W, 3, generator=g)
    img = img.clamp(0, 1).permute(2, 0, 1).contiguous()
    return img, label


class ToyDrivingDataset(Dataset):
    """``n`` deterministic synthetic driving scenes."""

    classes = CITYSCAPES_CLASSES

    def __init__(self, n: int = 64, size=(128, 256), seed: int = 0, transform=None):
        self.n = n
        self.size = size
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        img, label = make_toy_scene(self.size[0], self.size[1], seed=self.seed + i)
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label
