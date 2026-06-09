"""Learned spectral response h_θ(λ) + Marchenko–Pastur noise floor.

THE theoretical contribution. Prior work (SpecTemp, CLASP, …) all apply some
*fixed, hand-chosen* weighting to the eigenbasis — a scalar temper exponent, a
hard truncation at K. They are special cases of one object: a function h(λ) on the
spectrum. We make that function **learned and label-supervised**:

        Φ = U · diag(h_θ(λ))

`SpectralResponse` is h_θ — a small differentiable filter on eigenvalues. Because
the bottom Laplacian eigenvalues carry segmentation structure and the bulk carries
noise, we give the filter a principled prior on *where* the noise is, via random
matrix theory:

`mp_upper_edge` returns the Marchenko–Pastur bulk edge λ₊ = σ²(1+√γ)², γ = d/n —
the largest eigenvalue a *pure-noise* feature covariance can produce. Eigenvalues
inside the bulk are noise (BBP: signal sits above the edge). `bulk_penalty`
regularises h_θ → 0 inside the bulk, turning SpecTemp's arbitrary "last-10%" noise
floor into an actual generative estimate.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SpectralResponse(nn.Module):
    """Differentiable filter h_θ(λ) ≥ 0 applied elementwise to eigenvalues.

    kind="mlp"  : a small MLP on λ — maximally expressive, the default.
    kind="cheb" : a Chebyshev polynomial Σ θ_m T_m(λ̃) — graph-signal-processing
                  style, ties the filter to the ChebNet lineage, fewer params.
    """

    def __init__(self, kind: str = "mlp", hidden: int = 32, cheb_order: int = 8,
                 lam_max: float = 2.0):
        super().__init__()
        self.kind = kind
        self.lam_max = lam_max
        if kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )
        elif kind == "cheb":
            # initialise near identity-ish low-pass: θ_0 = 1, rest 0
            theta = torch.zeros(cheb_order + 1)
            theta[0] = 1.0
            self.theta = nn.Parameter(theta)
        else:
            raise ValueError(f"unknown response kind {kind!r}")

    def forward(self, lam: Tensor) -> Tensor:
        if self.kind == "mlp":
            h = self.net(lam.unsqueeze(-1)).squeeze(-1)
            return nn.functional.softplus(h)
        # Chebyshev on λ rescaled to [-1, 1]
        x = (2.0 * lam / self.lam_max - 1.0).clamp(-1.0, 1.0)
        Tkm1 = torch.ones_like(x)
        Tk = x
        out = self.theta[0] * Tkm1 + self.theta[1] * Tk if self.theta.numel() > 1 else self.theta[0] * Tkm1
        for m in range(2, self.theta.numel()):
            Tkp1 = 2.0 * x * Tk - Tkm1
            out = out + self.theta[m] * Tkp1
            Tkm1, Tk = Tk, Tkp1
        return nn.functional.softplus(out)


# --------------------------------------------------------------------------- #
# Marchenko–Pastur noise floor
# --------------------------------------------------------------------------- #
def mp_upper_edge(n: int, d: int, var: float = 1.0) -> float:
    """Upper edge of the MP bulk for an ``n``-sample, ``d``-feature noise matrix.

    λ₊ = var · (1 + √(d/n))². Eigenvalues of the feature covariance/affinity below
    this are statistically indistinguishable from noise.
    """
    gamma = d / max(n, 1)
    return float(var) * (1.0 + math.sqrt(gamma)) ** 2


def mp_bulk_mask(eigvals: Tensor, n: int, d: int, var: float = 1.0) -> Tensor:
    """Boolean mask of eigenvalues lying inside the MP bulk (i.e. noise)."""
    return eigvals <= mp_upper_edge(n, d, var)


def bulk_penalty(h: Tensor, eigvals: Tensor, n: int, d: int, var: float = 1.0) -> Tensor:
    """Penalise spectral response mass inside the MP noise bulk.

    Returns mean h(λ)² over eigenvalues judged to be noise — added to the training
    loss so the learned filter suppresses noise eigenvectors without a hand-set
    threshold. Zero if no eigenvalue is in the bulk.
    """
    mask = mp_bulk_mask(eigvals, n, d, var)
    if mask.any():
        return (h[mask] ** 2).mean()
    return h.new_zeros(())
