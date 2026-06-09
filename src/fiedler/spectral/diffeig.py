"""Differentiable symmetric eigendecomposition with degeneracy-robust gradients.

This is Phase-1's gate. Standard autodiff through an eigendecomposition multiplies
upstream gradients by ``1/(λ_i − λ_j)`` for every pair of eigenvalues. At segment
boundaries — exactly where dense prediction matters — eigenvalues become nearly
degenerate, that term explodes, and training diverges or NaNs. (PyTorch's native
``torch.linalg.eigh`` backward does precisely this and is documented to be unstable
for close eigenvalues.)

The fix (Novelty #2): replace the singular ``1/(λ_i − λ_j)`` with its **Lorentzian
broadening**

        F_ij = (λ_j − λ_i) / ((λ_j − λ_i)² + ε²),

which equals ``1/(λ_j − λ_i)`` when eigenvalues are well separated but stays bounded
by ``1/(2ε)`` as the gap → 0. The forward pass uses ``torch.linalg.eigh`` (fast,
C++); only the backward is replaced, so this composes with the rest of autograd.

Backward derivation (A = U diag(λ) Uᵀ, symmetric):
    Ā = U ( diag(ḡλ) + F ∘ (Uᵀ ḡU) ) Uᵀ ,  then symmetrised,
with F as above. Validated against finite differences via ``torch.autograd.gradcheck``.
"""
from __future__ import annotations

import torch
from torch import Tensor


class _BroadenedEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, eps: float):
        with torch.no_grad():
            evals, evecs = torch.linalg.eigh(A)      # ascending, orthonormal columns
        ctx.save_for_backward(evals, evecs)
        ctx.eps = eps
        return evals, evecs

    @staticmethod
    def backward(ctx, g_evals: Tensor | None, g_evecs: Tensor | None):
        evals, evecs = ctx.saved_tensors
        eps = ctx.eps
        n = evals.shape[0]

        # diff[i, j] = λ_j − λ_i ; broadened reciprocal (0 on the diagonal automatically)
        diff = evals.unsqueeze(0) - evals.unsqueeze(1)
        F = diff / (diff * diff + eps * eps)

        M = evecs.new_zeros((n, n))
        if g_evals is not None:
            M = M + torch.diag(g_evals)
        if g_evecs is not None:
            M = M + F * (evecs.t() @ g_evecs)

        A_bar = evecs @ M @ evecs.t()
        A_bar = 0.5 * (A_bar + A_bar.t())            # input is symmetric
        return A_bar, None


def broadened_eigh(A: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Symmetric eigendecomposition with degeneracy-robust backward.

    Parameters
    ----------
    A   : (n, n) symmetric.
    eps : Lorentzian broadening of the eigenvalue-gap reciprocal in the backward
          pass. ``eps → 0`` recovers exact gradients (use for gradcheck on
          well-separated spectra); larger ``eps`` trades a little gradient bias for
          stability across near-degenerate eigenvalues.

    Returns (eigenvalues ascending (n,), eigenvectors (n, n)).
    """
    return _BroadenedEigh.apply(A, eps)


def smallest_k(A: Tensor, k: int, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Bottom-``k`` eigenpairs of a symmetric matrix, differentiably.

    The bottom of the Laplacian spectrum is where segmentation structure lives
    (λ₁ = 0, the Fiedler vector λ₂, …). Slicing after a full differentiable eigh
    keeps gradients exact — the discarded eigenvectors still contribute their
    cross-terms through the saved full basis in the backward pass.
    """
    evals, evecs = broadened_eigh(A, eps)
    return evals[:k], evecs[:, :k]
