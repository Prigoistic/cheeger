"""Standard dense-segmentation losses (the shared base for both heads).

Controlled-minimal recipe: the conv baseline and the spectral head both train with
``cross_entropy_2d`` (optionally class-weighted). OHEM and Lovász-Softmax are built
here too but are *opt-in* — reserved for the later strong-baseline stage so they
never confound the spectral-vs-conv ablation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def cross_entropy_2d(logits: Tensor, target: Tensor, weight: Tensor | None = None,
                     ignore_index: int = 255, reduction: str = "mean") -> Tensor:
    """Per-pixel softmax cross-entropy. logits (B,K,H,W), target (B,H,W) int."""
    return F.cross_entropy(logits, target.long(), weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def class_weights_median_freq(class_pixel_counts: Tensor) -> Tensor:
    """Median-frequency balancing (Eigen & Fergus): w_c = median(freq) / freq_c.

    Counters Cityscapes imbalance (road ≫ pole). Classes with zero count get weight 0.
    """
    counts = class_pixel_counts.double()
    total = counts.sum().clamp_min(1)
    freq = counts / total
    nonzero = freq[freq > 0]
    med = nonzero.median() if nonzero.numel() > 0 else torch.tensor(1.0)
    w = torch.where(freq > 0, med / freq, torch.zeros_like(freq))
    return w.float()


class OHEMCrossEntropy(nn.Module):
    """Online Hard Example Mining CE — backprop only through hard pixels.

    Opt-in (not in the controlled-minimal recipe). Keeps pixels whose true-class
    probability is below ``thresh``, but always at least ``min_kept`` hardest ones.
    """

    def __init__(self, thresh: float = 0.7, min_kept: int = 100_000,
                 ignore_index: int = 255, weight: Tensor | None = None):
        super().__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target.long()
        pix = F.cross_entropy(logits, target, weight=self.weight,
                              ignore_index=self.ignore_index, reduction="none").reshape(-1)
        with torch.no_grad():
            prob = F.softmax(logits, dim=1)
            tgt = target.clone()
            tgt[tgt == self.ignore_index] = 0
            true_p = prob.gather(1, tgt.unsqueeze(1)).squeeze(1).reshape(-1)
            valid = (target != self.ignore_index).reshape(-1)
            hard = valid & (true_p < self.thresh)
            if hard.sum() < self.min_kept:
                # fall back to the min_kept hardest valid pixels
                masked = torch.where(valid, pix, torch.full_like(pix, -1.0))
                k = min(self.min_kept, int(valid.sum()))
                if k > 0:
                    thr = torch.topk(masked, k).values.min()
                    hard = valid & (pix >= thr)
        sel = pix[hard]
        return sel.mean() if sel.numel() > 0 else pix[valid].mean()


# --------------------------------------------------------------------------- #
# Lovász-Softmax (Berman et al. 2018) — direct IoU surrogate, opt-in.
# --------------------------------------------------------------------------- #
def _lovasz_grad(gt_sorted: Tensor) -> Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - inter / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probs: Tensor, target: Tensor, ignore_index: int = 255) -> Tensor:
    """probs (B,K,H,W) softmax probabilities; target (B,H,W). Mean over classes."""
    B, K, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, K)
    labels = target.reshape(-1)
    valid = labels != ignore_index
    probs, labels = probs[valid], labels[valid]
    if probs.numel() == 0:
        return probs.sum() * 0.0
    losses = []
    for c in range(K):
        fg = (labels == c).to(probs.dtype)
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0
