"""Segmentation metrics derived from a confusion matrix.

All standard Cityscapes / semantic-segmentation numbers:
  * per-class IoU and mean IoU (mIoU) — the primary benchmark
  * pixel accuracy, mean (per-class) accuracy
  * frequency-weighted IoU

Cityscapes convention: a class absent from both prediction and ground truth
(IoU denominator 0) is excluded from the mean rather than counted as 0.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .confusion import ConfusionMatrix

# canonical 19 Cityscapes eval classes (trainId order)
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def per_class_iou(cm: Tensor) -> Tensor:
    """IoU per class; NaN where the class is absent from both pred and GT."""
    cm = cm.double()
    tp = torch.diagonal(cm)
    fp = cm.sum(dim=0) - tp          # predicted c, not actually c
    fn = cm.sum(dim=1) - tp          # actually c, not predicted c
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))
    return iou


def mean_iou(cm: Tensor) -> float:
    iou = per_class_iou(cm)
    present = ~torch.isnan(iou)
    return float(iou[present].mean()) if present.any() else float("nan")


def pixel_accuracy(cm: Tensor) -> float:
    cm = cm.double()
    total = cm.sum()
    return float(torch.diagonal(cm).sum() / total) if total > 0 else float("nan")


def mean_accuracy(cm: Tensor) -> float:
    """Mean per-class recall = mean_c TP_c / (TP_c + FN_c)."""
    cm = cm.double()
    tp = torch.diagonal(cm)
    gt = cm.sum(dim=1)
    acc = torch.where(gt > 0, tp / gt, torch.full_like(tp, float("nan")))
    present = ~torch.isnan(acc)
    return float(acc[present].mean()) if present.any() else float("nan")


def frequency_weighted_iou(cm: Tensor) -> float:
    cm = cm.double()
    iou = per_class_iou(cm)
    freq = cm.sum(dim=1) / cm.sum().clamp_min(1)
    mask = ~torch.isnan(iou)
    return float((freq[mask] * iou[mask]).sum()) if mask.any() else float("nan")


class SegMetrics:
    """Stateful accumulator: ``.update(pred, target)`` per batch, ``.compute()`` once."""

    def __init__(self, num_classes: int, ignore_index: int = 255,
                 class_names: list[str] | None = None):
        self.cmat = ConfusionMatrix(num_classes, ignore_index)
        self.num_classes = num_classes
        self.class_names = class_names or (
            CITYSCAPES_CLASSES if num_classes == 19 else [f"class_{i}" for i in range(num_classes)]
        )

    def reset(self):
        self.cmat.reset()

    def update(self, pred, target):
        self.cmat.update(pred, target)

    def compute(self) -> dict:
        cm = self.cmat.compute()
        iou = per_class_iou(cm)
        return {
            "mIoU": mean_iou(cm),
            "pixel_acc": pixel_accuracy(cm),
            "mean_acc": mean_accuracy(cm),
            "fwIoU": frequency_weighted_iou(cm),
            "per_class_iou": {
                name: (float(v) if not torch.isnan(v) else float("nan"))
                for name, v in zip(self.class_names, iou)
            },
        }
