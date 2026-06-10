"""Spectral regularisers — the novel objective terms.

`rayleigh_consistency` is the contribution: the relaxed Normalized-Cut / Dirichlet
energy of the *predicted* segmentation on the feature graph,

        R(P) = (1/K) Σ_c  (pᶜᵀ L pᶜ) / (pᶜᵀ pᶜ + ε),

where pᶜ is the soft prediction map for class c and L is the graph Laplacian.
Minimising R encourages graph-connected pixels to share labels — a global,
CRF-free smoothness prior that uses the *same* learned graph the embedding is built
on. It runs at **graph resolution** (the downsampled feature map), never full-res.

`bulk_penalty` (re-exported from spectral.response) suppresses the learned spectral
response inside the Marchenko–Pastur noise bulk.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..spectral.response import bulk_penalty  # re-export

__all__ = ["rayleigh_consistency", "bulk_penalty", "CompositeLoss"]


def rayleigh_consistency(prob: Tensor, L: Tensor, eps: float = 1e-8) -> Tensor:
    """prob (N, K) soft predictions on the graph nodes; L (N, N) Laplacian."""
    LP = L @ prob                              # (N, K)
    num = (prob * LP).sum(dim=0)               # (K,)  pᶜᵀ L pᶜ
    den = (prob * prob).sum(dim=0) + eps       # (K,)  pᶜᵀ pᶜ
    return (num / den).mean()


class CompositeLoss(nn.Module):
    """Total training objective = base CE + spectral regularisers.

        L = w_ce·CE + w_ray·Rayleigh + w_bulk·bulk_penalty

    The base CE is what both heads share; the spectral terms only fire when the
    caller supplies graph-resolution predictions + Laplacian (spectral head). Returns
    ``(total, components_dict)`` so the trainer can log each piece.
    """

    def __init__(self, w_ce: float = 1.0, w_rayleigh: float = 0.1, w_bulk: float = 0.0,
                 ignore_index: int = 255, class_weight: Tensor | None = None):
        super().__init__()
        self.w_ce = w_ce
        self.w_rayleigh = w_rayleigh
        self.w_bulk = w_bulk
        self.ignore_index = ignore_index
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        *,
        prob_graph: Tensor | None = None,
        L: Tensor | None = None,
        eigvals: Tensor | None = None,
        response_h: Tensor | None = None,
        mp_n: int | None = None,
        mp_d: int | None = None,
    ) -> tuple[Tensor, dict]:
        from .segmentation import cross_entropy_2d

        ce = cross_entropy_2d(logits, target, weight=self.class_weight,
                              ignore_index=self.ignore_index)
        total = self.w_ce * ce
        comps = {"ce": float(ce.detach())}

        # spectral terms accept a single (prob, L) or per-image lists (batch) and
        # average over the batch — the spectral head emits one graph per image.
        if self.w_rayleigh > 0 and prob_graph is not None and L is not None:
            pairs = list(zip(prob_graph, L)) if isinstance(prob_graph, (list, tuple)) \
                else [(prob_graph, L)]
            ray = torch.stack([rayleigh_consistency(p, l) for p, l in pairs]).mean()
            total = total + self.w_rayleigh * ray
            comps["rayleigh"] = float(ray.detach())

        if self.w_bulk > 0 and response_h is not None and eigvals is not None \
                and mp_n is not None and mp_d is not None:
            items = list(zip(response_h, eigvals)) if isinstance(response_h, (list, tuple)) \
                else [(response_h, eigvals)]
            bp = torch.stack([bulk_penalty(h, e, mp_n, mp_d) for h, e in items]).mean()
            total = total + self.w_bulk * bp
            comps["bulk"] = float(bp.detach())

        comps["total"] = float(total.detach())
        return total, comps
