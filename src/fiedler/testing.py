"""Rigor toolkit — mathematical invariant checks + canonical graph generators.

Shipped inside the package (like ``numpy.testing``) so the *same* invariants guard
every stage: the operators today, the differentiable eigensolver and losses
tomorrow. Tests and the stress fuzzer both import from here — one source of truth
for "what must always be true".

Two groups:
  * ``assert_*`` — raise AssertionError with a descriptive message on violation.
  * graph generators — return ``(W, meta)`` with known ground-truth structure.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Invariant assertions
# --------------------------------------------------------------------------- #
def assert_finite(t: Tensor, name: str = "tensor") -> None:
    if torch.isnan(t).any():
        raise AssertionError(f"{name}: contains NaN")
    if torch.isinf(t).any():
        raise AssertionError(f"{name}: contains Inf")


def assert_symmetric(M: Tensor, tol: float = 1e-9, name: str = "matrix") -> None:
    err = (M - M.t()).abs().max().item()
    if err > tol:
        raise AssertionError(f"{name}: not symmetric (max|M-Mᵀ|={err:.2e} > {tol:.0e})")


def assert_psd(M: Tensor, tol: float = 1e-6, name: str = "matrix") -> None:
    """Positive semidefinite within tolerance (smallest eigenvalue ≥ -tol)."""
    lo = torch.linalg.eigvalsh(M.double()).min().item()
    if lo < -tol:
        raise AssertionError(f"{name}: not PSD (λ_min={lo:.2e} < -{tol:.0e})")


def assert_spectrum_in(M: Tensor, lo: float, hi: float, tol: float = 1e-6,
                       name: str = "matrix") -> None:
    ev = torch.linalg.eigvalsh(M.double())
    if ev.min().item() < lo - tol or ev.max().item() > hi + tol:
        raise AssertionError(
            f"{name}: spectrum [{ev.min():.4f},{ev.max():.4f}] outside [{lo},{hi}]±{tol:.0e}")


def assert_orthonormal(U: Tensor, tol: float = 1e-8, name: str = "eigenvectors") -> None:
    n = U.shape[1]
    err = (U.t() @ U - torch.eye(n, dtype=U.dtype, device=U.device)).abs().max().item()
    if err > tol:
        raise AssertionError(f"{name}: not orthonormal (max|UᵀU-I|={err:.2e} > {tol:.0e})")


def assert_ascending(v: Tensor, tol: float = 1e-9, name: str = "eigenvalues") -> None:
    diffs = (v[1:] - v[:-1]).min().item() if v.numel() > 1 else 0.0
    if diffs < -tol:
        raise AssertionError(f"{name}: not ascending (min step {diffs:.2e})")


def assert_eig_reconstruction(A: Tensor, evals: Tensor, evecs: Tensor,
                              tol: float = 1e-8, name: str = "eig") -> None:
    recon = (evecs * evals.unsqueeze(0)) @ evecs.t()
    err = (recon - A).abs().max().item()
    if err > tol:
        raise AssertionError(f"{name}: reconstruction error {err:.2e} > {tol:.0e}")


def assert_rows_sum_zero(L: Tensor, tol: float = 1e-9, name: str = "comb-laplacian") -> None:
    err = L.sum(dim=1).abs().max().item()
    if err > tol:
        raise AssertionError(f"{name}: rows do not sum to 0 (max|Σrow|={err:.2e})")


def count_zero_eigs(M: Tensor, tol: float = 1e-7) -> int:
    return int((torch.linalg.eigvalsh(M.double()).abs() < tol).sum())


# --------------------------------------------------------------------------- #
# Canonical graph generators  ->  (W, meta)
# --------------------------------------------------------------------------- #
def random_symmetric(n: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    return M + M.t()


def path_graph(n: int) -> tuple[Tensor, dict]:
    W = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W, {"components": 1, "name": f"P{n}"}


def cycle_graph(n: int) -> tuple[Tensor, dict]:
    W, _ = path_graph(n)
    W[0, n - 1] = W[n - 1, 0] = 1.0
    return W, {"components": 1, "name": f"C{n}"}


def complete_graph(n: int) -> tuple[Tensor, dict]:
    W = torch.ones(n, n, dtype=torch.float64) - torch.eye(n, dtype=torch.float64)
    return W, {"components": 1, "name": f"K{n}"}


def disconnected_blocks(sizes: list[int]) -> tuple[Tensor, dict]:
    """Block-diagonal union of complete graphs — #components == len(sizes)."""
    n = sum(sizes)
    W = torch.zeros(n, n, dtype=torch.float64)
    off = 0
    for s in sizes:
        blk, _ = complete_graph(s)
        W[off:off + s, off:off + s] = blk
        off += s
    return W, {"components": len(sizes), "name": f"blocks{sizes}"}


def planted_partition_features(sizes: list[int], sep: float = 6.0, dim: int = 4,
                               seed: int = 0) -> tuple[Tensor, Tensor]:
    """Gaussian blobs, one per cluster, separated along distinct axes.
    Returns (features (N,dim), labels (N,))."""
    g = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for ci, s in enumerate(sizes):
        center = torch.zeros(dim)
        center[ci % dim] = sep * (1 + ci // dim)
        feats.append(torch.randn(s, dim, generator=g, dtype=torch.float64) + center)
        labels.append(torch.full((s,), ci, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels)


def with_isolated(W: Tensor, n_isolated: int) -> Tensor:
    """Append ``n_isolated`` zero-degree nodes to an affinity matrix."""
    n = W.shape[0]
    out = torch.zeros(n + n_isolated, n + n_isolated, dtype=W.dtype)
    out[:n, :n] = W
    return out
