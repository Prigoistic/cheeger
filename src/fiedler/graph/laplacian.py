"""Graph construction and Laplacian operators — written from scratch in torch.

Everything here is differentiable. No scipy / numpy in the forward path; those
are used only as correctness oracles in tests/.

Conventions
-----------
* A feature map is a tensor ``X`` of shape ``(N, C)`` — ``N`` nodes, ``C`` channels.
* An affinity / weight matrix ``W`` is ``(N, N)``, symmetric, non-negative, zero diagonal.
* The degree vector ``d`` is ``W.sum(dim=1)``; ``D = diag(d)``.
"""
from __future__ import annotations

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Pairwise geometry
# --------------------------------------------------------------------------- #
def pairwise_sq_dists(X: Tensor, metric: Tensor | None = None) -> Tensor:
    """Squared distances ``D2[i,j] = (xi - xj)^T M (xi - xj)``.

    If ``metric`` (M) is ``None`` this is the plain Euclidean squared distance.
    ``metric`` may be:
      * a vector of shape ``(C,)``  -> diagonal (per-channel) metric, M = diag(w**2)
      * a matrix of shape ``(C, C)`` -> full anisotropic metric (should be PSD)

    The learned-metric path (Novelty #1) flows through here: make ``metric``
    a learnable parameter and the graph itself becomes trainable.
    """
    if metric is None:
        # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
        sq = (X * X).sum(dim=1)                       # (N,)
        gram = X @ X.t()                              # (N, N)
        d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * gram
    else:
        if metric.dim() == 1:                         # diagonal metric
            Xm = X * metric.unsqueeze(0)              # scale channels by w
            sq = (Xm * X).sum(dim=1)
            gram = Xm @ X.t()
            d2 = sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * gram
        else:                                         # full M = L L^T factorisation safe
            XM = X @ metric                           # (N, C)
            sq_i = (XM * X).sum(dim=1)
            d2 = sq_i.unsqueeze(1) + sq_i.unsqueeze(0) - 2.0 * (XM @ X.t())
    return d2.clamp_min(0.0)                           # kill tiny negatives from fp error


# --------------------------------------------------------------------------- #
# Affinity
# --------------------------------------------------------------------------- #
def gaussian_affinity(
    X: Tensor,
    sigma: float | Tensor = 1.0,
    k: int | None = None,
    metric: Tensor | None = None,
    self_loops: bool = False,
) -> Tensor:
    """Dense Gaussian affinity ``W = exp(-D2 / (2 sigma^2))``, optionally k-NN sparsified.

    Parameters
    ----------
    sigma : scalar bandwidth (learnable Tensor allowed).
    k     : if given, keep only each node's ``k`` nearest neighbours (symmetrised).
    metric: optional learned metric passed to :func:`pairwise_sq_dists`.
    """
    d2 = pairwise_sq_dists(X, metric=metric)
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(sigma, dtype=X.dtype, device=X.device)
    W = torch.exp(-d2 / (2.0 * sigma * sigma + 1e-12))

    if not self_loops:
        W = W - torch.diag(torch.diag(W))

    if k is not None:
        W = knn_sparsify(W, k)
    return 0.5 * (W + W.t())                           # enforce exact symmetry


def knn_sparsify(W: Tensor, k: int) -> Tensor:
    """Keep the top-``k`` entries per row, zero the rest, then symmetrise by OR.

    Symmetrising by ``max`` keeps an edge if *either* endpoint chose the other —
    the standard mutual-vs-or choice in spectral clustering (we use OR).
    """
    n = W.shape[0]
    k = min(k, n - 1)
    topk_val, topk_idx = W.topk(k, dim=1)
    mask = torch.zeros_like(W)
    mask.scatter_(1, topk_idx, 1.0)
    Wk = W * mask
    return torch.maximum(Wk, Wk.t())


# --------------------------------------------------------------------------- #
# Laplacian variants
# --------------------------------------------------------------------------- #
def degree(W: Tensor) -> Tensor:
    return W.sum(dim=1)


def laplacian(W: Tensor, kind: str = "sym", eps: float = 1e-12) -> Tensor:
    """Graph Laplacian of affinity ``W``.

    kind:
      * ``"comb"`` — combinatorial / unnormalised  L = D - W
      * ``"sym"``  — symmetric normalised           L_sym = I - D^{-1/2} W D^{-1/2}
      * ``"rw"``   — random-walk normalised          L_rw  = I - D^{-1} W

    ``L_sym`` is symmetric PSD with spectrum in ``[0, 2]`` — the right object for
    a numerically stable symmetric eigensolver. ``L_rw`` is not symmetric.
    """
    d = degree(W)
    D = torch.diag(d)
    if kind == "comb":
        return D - W
    if kind == "sym":
        dinv_sqrt = torch.rsqrt(d.clamp_min(eps))
        n = W.shape[0]
        I = torch.eye(n, dtype=W.dtype, device=W.device)
        return I - (dinv_sqrt.unsqueeze(1) * W * dinv_sqrt.unsqueeze(0))
    if kind == "rw":
        dinv = 1.0 / d.clamp_min(eps)
        n = W.shape[0]
        I = torch.eye(n, dtype=W.dtype, device=W.device)
        return I - dinv.unsqueeze(1) * W
    raise ValueError(f"unknown laplacian kind: {kind!r}")
