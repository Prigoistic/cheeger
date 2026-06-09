"""Cityscapes 19-class palette + label colorization.

The canonical trainId -> RGB mapping used by every Cityscapes leaderboard, so our
visual masks match published qualitative figures.
"""
from __future__ import annotations

import torch
from torch import Tensor

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

_CITYSCAPES_RGB = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
]


def cityscapes_palette(ignore_color=(0, 0, 0)) -> Tensor:
    """(20, 3) uint8 palette; index 19 is the ignore/void colour."""
    return torch.tensor([*_CITYSCAPES_RGB, ignore_color], dtype=torch.uint8)


def colorize(mask: Tensor, palette: Tensor | None = None) -> Tensor:
    """(H,W) int labels -> (H,W,3) uint8 RGB."""
    if palette is None:
        palette = cityscapes_palette()
    mask = mask.long().clamp(0, palette.shape[0] - 1)
    return palette.to(mask.device)[mask]
