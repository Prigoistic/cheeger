"""From-scratch symmetric eigensolver — the cyclic Jacobi method.

This is the ground-truth, "I implemented the eigendecomposition myself" solver.
It diagonalises a real symmetric matrix ``A`` by a sequence of Givens rotations
``A <- J^T A J`` each of which zeros one off-diagonal entry. It converges
quadratically and is unconditionally stable for symmetric matrices — no shifts,
no deflation bookkeeping. O(n^3) per sweep, a handful of sweeps total.

For large sparse Laplacians we will switch to Lanczos (top-k only); Jacobi is the
dense reference that every other solver is validated against, and it is fully
correct for the small graphs used in unit tests and toy demos.

Returns eigenvalues in ascending order with matching orthonormal eigenvectors.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _off_norm(A: Tensor) -> Tensor:
    """Frobenius norm of the strict off-diagonal part."""
    return torch.sqrt((A * A).sum() - (torch.diagonal(A) ** 2).sum()).clamp_min(0.0)


def jacobi_eigh(
    A: Tensor,
    max_sweeps: int = 100,
    tol: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Eigendecomposition of a real symmetric matrix via cyclic Jacobi.

    Parameters
    ----------
    A : ``(n, n)`` symmetric. Internally promoted to float64 for accuracy.
    max_sweeps : cap on full N(N-1)/2 sweeps.
    tol : stop when the off-diagonal Frobenius norm falls below this.

    Returns
    -------
    (eigenvalues ``(n,)`` ascending, eigenvectors ``(n, n)`` columns).
    """
    n = A.shape[0]
    work_dtype = torch.float64
    A = A.to(work_dtype).clone()
    A = 0.5 * (A + A.t())                      # symmetrise defensively
    V = torch.eye(n, dtype=work_dtype, device=A.device)

    for _ in range(max_sweeps):
        if _off_norm(A) < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if apq.abs() < 1e-300:
                    continue
                app = A[p, p]
                aqq = A[q, q]
                # symmetric Schur: choose (c, s) zeroing A[p, q]
                tau = (aqq - app) / (2.0 * apq)
                t = torch.sign(tau) / (tau.abs() + torch.sqrt(tau * tau + 1.0))
                if tau == 0:
                    t = torch.ones((), dtype=work_dtype, device=A.device)
                c = 1.0 / torch.sqrt(t * t + 1.0)
                s = t * c

                # A <- J^T A J, J rotating plane (p, q): [[c, s], [-s, c]]
                col_p = A[:, p].clone()
                col_q = A[:, q].clone()
                A[:, p] = c * col_p - s * col_q
                A[:, q] = s * col_p + c * col_q
                row_p = A[p, :].clone()
                row_q = A[q, :].clone()
                A[p, :] = c * row_p - s * row_q
                A[q, :] = s * row_p + c * row_q

                # accumulate eigenvectors: V <- V J
                vp = V[:, p].clone()
                vq = V[:, q].clone()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq

    eigvals = torch.diagonal(A).clone()
    order = torch.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = V[:, order]
    return eigvals.to(torch.get_default_dtype()), eigvecs.to(torch.get_default_dtype())
