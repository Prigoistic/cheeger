"""Segmentation heads — the swappable component that the whole study compares.

Same U-Net backbone, two heads:
  * ``ConvHead``       — 1×1 conv. The conventional baseline.
  * ``SpectralSegHead``— our differentiable spectral embedding → MLP → logits.

Both consume a feature map ``(B, C, H, W)`` and return a dict with full-resolution
``logits (B, K, H, W)``. The spectral head additionally returns graph-resolution
``prob_graph`` / ``laplacian`` / ``eigvals`` / ``response`` so ``CompositeLoss`` can
add the Rayleigh consistency + MP bulk regularisers. Keeping the heads
interface-compatible is what makes the conv-vs-spectral ablation a clean swap.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..spectral.embedding import SpectralEmbedding


class ConvHead(nn.Module):
    """1×1 conv classifier — the baseline."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, feat: Tensor) -> dict:
        return {"logits": self.classifier(feat)}


class SpectralSegHead(nn.Module):
    """Spectral embedding head.

    Builds the affinity graph at a downsampled ``graph_hw`` resolution (the scale
    knob — dense eigh is O(n³), so we cap N = graph_hw²), embeds, classifies each
    node with a shared MLP, then bilinearly upsamples logits to full resolution.
    Per-image graphs ⇒ we map over the batch.
    """

    def __init__(self, in_channels: int, num_classes: int, k: int = 16,
                 graph_hw: int = 32, knn: int = 16, laplacian_kind: str = "sym",
                 learn_sigma: bool = True, learn_metric: bool = True,
                 sigma_init: float = 1.0, mlp_hidden: int = 64):
        super().__init__()
        self.graph_hw = graph_hw
        self.num_classes = num_classes
        self.embed = SpectralEmbedding(
            k=k, laplacian_kind=laplacian_kind, knn=knn,
            metric_dim=in_channels if learn_metric else None,
            learn_sigma=learn_sigma, sigma_init=sigma_init,
        )
        self.classifier = nn.Sequential(
            nn.Linear(k, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, feat: Tensor) -> dict:
        B, C, H, W = feat.shape
        gh = gw = self.graph_hw
        g = F.adaptive_avg_pool2d(feat, (gh, gw))            # (B, C, gh, gw)

        logit_maps, prob_graph, lap, eig, resp = [], [], [], [], []
        for b in range(B):
            X = g[b].permute(1, 2, 0).reshape(gh * gw, C)    # (N, C)
            out = self.embed(X)
            node_logits = self.classifier(out["phi"])        # (N, K)
            logit_maps.append(node_logits.t().reshape(self.num_classes, gh, gw))
            prob_graph.append(F.softmax(node_logits, dim=1))  # (N, K) for Rayleigh
            lap.append(out["laplacian"]); eig.append(out["eigvals"]); resp.append(out["response"])

        logits_small = torch.stack(logit_maps)               # (B, K, gh, gw)
        logits = F.interpolate(logits_small, size=(H, W), mode="bilinear", align_corners=False)
        return {
            "logits": logits,
            "prob_graph": prob_graph,   # list[B] of (N, K)
            "laplacian": lap,           # list[B] of (N, N)
            "eigvals": eig,             # list[B] of (k,)
            "response": resp,           # list[B] of (k,)
            "graph_hw": (gh, gw),
        }


class SegModel(nn.Module):
    """Backbone + head. ``forward`` returns the head's dict (logits + any aux)."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> dict:
        return self.head(self.backbone(x))
