"""Boundary-quality metrics — the thesis differentiator.

The spectral head's claim is better *global structure* and crisper class
boundaries. Region mIoU is dominated by interiors and barely moves on thin/edge
structure, so we also measure boundaries directly:

  * ``boundary_iou``  — Cheng et al. 2021: per-class IoU computed only on a thin band
    along each class's contour, averaged (a "Boundary mIoU").
  * ``bf_score``      — contour F1 (Csurka 2013 / DAVIS): precision & recall of
    predicted vs ground-truth semantic edges within a pixel tolerance.
  * ``trimap_accuracy`` — pixel accuracy within a band around GT boundaries.

All torch morphology (max-pool dilation / erosion) — no OpenCV dependency.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _as_bchw(mask01: Tensor) -> Tensor:
    return mask01.float().unsqueeze(0).unsqueeze(0)


def dilate(mask: Tensor, r: int) -> Tensor:
    """Binary dilation by radius r (max-pool with (2r+1) window)."""
    if r <= 0:
        return mask
    out = F.max_pool2d(_as_bchw(mask), kernel_size=2 * r + 1, stride=1, padding=r)
    return out[0, 0] > 0.5


def erode(mask: Tensor, r: int) -> Tensor:
    """Binary erosion by radius r (dual of dilation)."""
    if r <= 0:
        return mask
    return ~dilate(~mask, r)


def semantic_edges(label: Tensor, ignore_index: int = 255) -> Tensor:
    """Boolean map: pixel differs from a 4-neighbour (a class boundary).
    Edges touching ignore pixels are dropped."""
    l = label
    e = torch.zeros_like(l, dtype=torch.bool)
    e[:, :-1] |= l[:, :-1] != l[:, 1:]
    e[:, 1:] |= l[:, 1:] != l[:, :-1]
    e[:-1, :] |= l[:-1, :] != l[1:, :]
    e[1:, :] |= l[1:, :] != l[:-1, :]
    if ignore_index is not None:
        e &= label != ignore_index
    return e


def _diag_radius(shape, ratio: float) -> int:
    h, w = shape[-2], shape[-1]
    return max(1, int(round(ratio * math.sqrt(h * h + w * w))))


def boundary_iou(pred: Tensor, target: Tensor, num_classes: int,
                 dilation_ratio: float = 0.02, ignore_index: int = 255) -> float:
    """Mean over present classes of IoU restricted to each class's boundary band."""
    d = _diag_radius(pred.shape, dilation_ratio)
    ious = []
    for c in range(num_classes):
        gt = (target == c)
        if not gt.any():
            continue
        pr = (pred == c)
        gtb = gt & ~erode(gt, d)
        prb = pr & ~erode(pr, d)
        union = (gtb | prb).sum()
        if union > 0:
            ious.append(((gtb & prb).sum().float() / union.float()).item())
    return float(sum(ious) / len(ious)) if ious else float("nan")


def bf_score(pred: Tensor, target: Tensor, tol: int = 2, ignore_index: int = 255) -> float:
    """Boundary F1: precision/recall of predicted vs GT semantic edges within ``tol``."""
    gt_e = semantic_edges(target, ignore_index)
    pr_e = semantic_edges(pred, ignore_index)
    if gt_e.sum() == 0 and pr_e.sum() == 0:
        return 1.0
    if gt_e.sum() == 0 or pr_e.sum() == 0:
        return 0.0
    gt_d = dilate(gt_e, tol)
    pr_d = dilate(pr_e, tol)
    prec = (pr_e & gt_d).sum().float() / pr_e.sum().float()
    rec = (gt_e & pr_d).sum().float() / gt_e.sum().float()
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def trimap_accuracy(pred: Tensor, target: Tensor, width: int = 3,
                    ignore_index: int = 255) -> float:
    """Pixel accuracy within a band of half-width ``width`` around GT boundaries."""
    band = dilate(semantic_edges(target, ignore_index), width)
    band &= target != ignore_index
    if band.sum() == 0:
        return float("nan")
    correct = (pred == target) & band
    return float(correct.sum().float() / band.sum().float())
