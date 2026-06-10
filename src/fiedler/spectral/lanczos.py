"""From-scratch Lanczos solver for the bottom-k eigenpairs — the scale lift.

Dense ``eigh`` is O(n³): ~3.7 s for one 4096-node graph (measured). Lanczos needs
only the bottom few eigenpairs and costs O(nnz · m) per iteration — it touches the
matrix solely through matrix-vector products, so a k-NN-sparse Laplacian (O(nk)
nonzeros) makes the whole solve near-linear in n. This is what lets the spectral
head run at real feature-map resolution.

Algorithm: build an m-step Krylov basis with the symmetric Lanczos recurrence and
**full reorthogonalisation** (cheap at small m, essential for accurate Ritz pairs),
project onto the resulting tridiagonal T, solve that small dense system, and map the
Ritz vectors back. We take the algebraically smallest k Ritz pairs — the bottom of
the Laplacian spectrum, where segmentation structure lives.

Differentiable: the forward runs without autograd; the backward reuses the
degeneracy-robust broadened-gap formula (`diffeig.eig_backward`) on the k computed
pairs — exact when k = n, a rank-k approximation otherwise (standard for large-scale
spectral methods).
"""
from __future__ import annotations

import torch
from torch import Tensor

from .diffeig import eig_backward


def _make_matvec(A):
    """Unify dense Tensor / sparse Tensor / callable into x -> A x."""
    if callable(A):
        return A
    if A.is_sparse:
        return lambda x: torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    return lambda x: A @ x


def lanczos_tridiag(matvec, n: int, m: int, v0: Tensor, reorth: bool = True):
    """m-step Lanczos: returns Q (n, m'), alpha (m',), beta (m'-1,), m' ≤ m."""
    dtype, device = v0.dtype, v0.device
    Q = torch.zeros(n, m, dtype=dtype, device=device)
    alpha = torch.zeros(m, dtype=dtype, device=device)
    beta = torch.zeros(m, dtype=dtype, device=device)

    q = v0 / v0.norm()
    Q[:, 0] = q
    for j in range(m):
        w = matvec(Q[:, j])
        a = torch.dot(w, Q[:, j]); alpha[j] = a
        w = w - a * Q[:, j]
        if j > 0:
            w = w - beta[j - 1] * Q[:, j - 1]
        if reorth:                                   # full reorth (twice for stability)
            Qj = Q[:, : j + 1]
            w = w - Qj @ (Qj.t() @ w)
            w = w - Qj @ (Qj.t() @ w)
        b = w.norm()
        if j + 1 < m:
            if b < 1e-10:                            # invariant subspace reached
                return Q[:, : j + 1], alpha[: j + 1], beta[:j]
            beta[j] = b
            Q[:, j + 1] = w / b
    return Q, alpha, beta[: m - 1]


def _solve_tridiag(alpha: Tensor, beta: Tensor):
    """Eigendecomp of the small symmetric tridiagonal (dense, ascending)."""
    m = alpha.shape[0]
    T = torch.diag(alpha)
    if m > 1:
        idx = torch.arange(m - 1)
        T[idx, idx + 1] = beta
        T[idx + 1, idx] = beta
    return torch.linalg.eigh(T)                      # tiny m×m solve


def _lanczos_forward(A, k: int, m: int | None, reorth: bool, seed: int):
    matvec = _make_matvec(A)
    n = A.shape[0] if not callable(A) else None
    if n is None:
        raise ValueError("pass matrix shape via a Tensor, not a bare callable")
    # Krylov dim: generous default so the bottom-k converge. Real Laplacians (gapped
    # low spectrum) converge far faster than this; dense random spectra need ~m≈n.
    m = m or min(n, max(4 * k + 40, 60))
    g = torch.Generator(device=A.device).manual_seed(seed)
    v0 = torch.randn(n, generator=g, dtype=A.dtype, device=A.device)

    Q, alpha, beta = lanczos_tridiag(matvec, n, m, v0, reorth)
    ritz_vals, ritz_vecs = _solve_tridiag(alpha, beta)   # (m',), (m', m')
    evals = ritz_vals[:k]                                 # smallest-k Ritz values
    evecs = Q @ ritz_vecs[:, :k]                          # (n, k) Ritz vectors
    return evals.contiguous(), evecs.contiguous()


class _LanczosSmallestK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, k: int, m, reorth: bool, eps: float, seed: int):
        with torch.no_grad():
            evals, evecs = _lanczos_forward(A, k, m, reorth, seed)
        ctx.save_for_backward(evals, evecs)
        ctx.eps = eps
        return evals, evecs

    @staticmethod
    def backward(ctx, g_evals, g_evecs):
        evals, evecs = ctx.saved_tensors
        return eig_backward(evals, evecs, g_evals, g_evecs, ctx.eps), None, None, None, None, None


def lanczos_smallest_k(A: Tensor, k: int, m: int | None = None, reorth: bool = True,
                       eps: float = 1e-8, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Bottom-``k`` eigenpairs of a symmetric ``A`` (dense or sparse), differentiably.

    Parameters
    ----------
    A      : (n, n) symmetric, dense or sparse COO.
    k      : number of smallest eigenpairs.
    m      : Krylov dimension (default ``max(2k+20, 30)``). Larger m → more accurate
             but slower; ``2k`` plus a margin is plenty for the well-separated low end.
    reorth : full reorthogonalisation (recommended; cheap at small m).
    eps    : broadening for the differentiable backward (see ``diffeig.eig_backward``).

    Returns (eigenvalues ascending (k,), eigenvectors (n, k)).
    """
    return _LanczosSmallestK.apply(A, k, m, reorth, eps, seed)
