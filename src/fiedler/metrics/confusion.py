"""Streaming confusion matrix — the backbone every segmentation metric derives from.

Accumulates a K×K integer matrix over an arbitrary number of batches/images via
``bincount``, so the whole validation set costs O(K²) memory regardless of how many
images or how large they are (we never hold predictions in memory). Convention:

    mat[t, p]  =  #pixels with ground-truth class t predicted as class p

so the diagonal is true positives, column sums are predicted counts, row sums are
ground-truth counts.
"""
from __future__ import annotations

import torch
from torch import Tensor


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self) -> None:
        self.mat.zero_()

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor) -> None:
        """pred, target: integer label tensors of identical shape (any dims)."""
        k = self.num_classes
        pred = pred.flatten().to(torch.int64)
        target = target.flatten().to(torch.int64)
        # keep only pixels whose ground truth is a valid class
        valid = (target != self.ignore_index) & (target >= 0) & (target < k)
        p = pred[valid].clamp_(0, k - 1)          # guard stray pred labels
        t = target[valid]
        idx = t * k + p
        binc = torch.bincount(idx, minlength=k * k)
        self.mat += binc.reshape(k, k).to(self.mat.device)

    def compute(self) -> Tensor:
        return self.mat.clone()
