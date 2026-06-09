"""Differentiable eigensolver — gradient correctness + degeneracy robustness.

The make-or-break detail of the whole project. We prove:
  1. the broadened backward matches finite differences (gradcheck), and
  2. it stays finite where the *native* eigh backward blows up (near-degenerate λ).
"""
import pytest
import torch

from fiedler.spectral import broadened_eigh, smallest_k
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)


def _well_separated(n, seed=0):
    """Symmetric matrix with integer-gapped eigenvalues (no degeneracy)."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    lam = torch.arange(1, n + 1, dtype=torch.float64)
    return (Q * lam) @ Q.t()


def test_gradcheck_eigenvalues():
    """Gradient of eigenvalues w.r.t. A matches finite differences."""
    A = _well_separated(6).requires_grad_(True)
    f = lambda M: broadened_eigh(0.5 * (M + M.t()), 1e-12)[0]
    assert torch.autograd.gradcheck(f, (A,), atol=1e-5, rtol=1e-3)


def test_gradcheck_eigenvector_functional():
    """Sign-invariant eigenvector functional U diag(g) Uᵀ matches finite diffs.
    (Raw eigenvectors have sign ambiguity; a Gram-style functional does not.)"""
    A = _well_separated(5, seed=1).requires_grad_(True)
    g = torch.linspace(0.5, 2.0, 5)

    def f(M):
        _, U = broadened_eigh(0.5 * (M + M.t()), 1e-12)
        return U @ torch.diag(g) @ U.t()

    assert torch.autograd.gradcheck(f, (A,), atol=1e-5, rtol=1e-3)


def test_gradcheck_smallest_k():
    A = _well_separated(7, seed=2).requires_grad_(True)
    g = torch.linspace(1.0, 2.0, 3)

    def f(M):
        lam, U = smallest_k(0.5 * (M + M.t()), k=3, eps=1e-12)
        return (lam * lam).sum() + (U @ torch.diag(g) @ U.t()).sum()

    assert torch.autograd.gradcheck(f, (A,), atol=1e-5, rtol=1e-3)


def test_broadening_finite_on_degenerate_spectrum():
    """Two eigenvalues 1e-9 apart: native eigh backward -> huge/NaN; ours bounded."""
    n = 6
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    lam = torch.tensor([1.0, 1.0 + 1e-9, 2.0, 3.0, 4.0, 5.0])  # near-degenerate pair
    A = ((Q * lam) @ Q.t()).requires_grad_(True)

    # ours, broadened
    _, U = broadened_eigh(0.5 * (A + A.t()), eps=1e-3)
    U.sum().backward()
    assert A.grad is not None
    ft.assert_finite(A.grad, "broadened grad")
    assert A.grad.abs().max() < 1e6, "broadened grad should stay bounded"


def test_broadening_matches_native_when_well_separated():
    """With tiny eps and well-separated λ, broadened grad ≈ native eigh grad."""
    A0 = _well_separated(6, seed=3)

    A1 = A0.clone().requires_grad_(True)
    (broadened_eigh(0.5 * (A1 + A1.t()), 1e-12)[1]).pow(2).sum().backward()

    A2 = A0.clone().requires_grad_(True)
    torch.linalg.eigh(0.5 * (A2 + A2.t()))[1].pow(2).sum().backward()

    assert torch.allclose(A1.grad, A2.grad, atol=1e-6)
