"""Lanczos bottom-k solver — accuracy vs dense oracle, sparse support, gradients."""
import pytest
import torch

from fiedler.spectral import lanczos_smallest_k
from fiedler.graph import gaussian_affinity, laplacian
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)


def _well_separated(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    lam = torch.arange(1, n + 1, dtype=torch.float64)
    return (Q * lam) @ Q.t()


@pytest.mark.parametrize("n,k", [(40, 4), (80, 6), (120, 8)])
def test_smallest_eigenvalues_match_dense(n, k):
    """Worst case: dense random symmetric (gapless spectrum) needs a large Krylov dim."""
    torch.manual_seed(n)
    M = torch.randn(n, n)
    A = M + M.t()
    evals, evecs = lanczos_smallest_k(A, k, m=min(n, 5 * k + 50))
    ref = torch.linalg.eigvalsh(A)[:k]
    assert torch.allclose(evals, ref, atol=1e-6), (evals - ref).abs().max()


def test_ritz_vectors_are_eigenvectors():
    """Residual ‖A v − λ v‖ is tiny for the returned bottom-k pairs."""
    A = _well_separated(60, seed=1)
    evals, evecs = lanczos_smallest_k(A, k=5, m=55)
    res = (A @ evecs - evecs * evals.unsqueeze(0)).norm(dim=0)
    assert res.max() < 1e-6, res.max()


def test_matches_on_knn_laplacian():
    """The real use case: bottom-k of a k-NN affinity L_sym."""
    torch.manual_seed(0)
    X = torch.randn(150, 8)
    L = laplacian(gaussian_affinity(X, sigma=1.0, k=10), kind="sym")
    evals, _ = lanczos_smallest_k(L, k=6, m=60)
    ref = torch.linalg.eigvalsh(L)[:6]
    assert torch.allclose(evals, ref, atol=1e-6)


def test_sparse_input_matches_dense():
    A = _well_separated(50, seed=2)
    sp = A.to_sparse_coo()
    ev_dense, _ = lanczos_smallest_k(A, k=4, m=40)
    ev_sparse, _ = lanczos_smallest_k(sp, k=4, m=40)
    assert torch.allclose(ev_dense, ev_sparse, atol=1e-7)


def test_gradcheck_full_basis_is_exact():
    """With k = n (m = n) Lanczos spans everything, so the broadened backward is the
    exact gradient — matches finite differences."""
    A = _well_separated(6, seed=3).requires_grad_(True)
    g = torch.linspace(0.5, 2.0, 6)

    def f(M):
        Ms = 0.5 * (M + M.t())
        lam, U = lanczos_smallest_k(Ms, k=6, m=6, eps=1e-12)
        return (lam * lam).sum() + (U @ torch.diag(g) @ U.t()).sum()

    assert torch.autograd.gradcheck(f, (A,), atol=1e-5, rtol=1e-3)


def test_truncated_backward_is_finite():
    """k < n gives a rank-k gradient — must be finite and flow to the input."""
    A = _well_separated(30, seed=4).requires_grad_(True)
    lam, U = lanczos_smallest_k(0.5 * (A + A.t()), k=5, m=25)
    (lam.sum() + U.pow(2).sum()).backward()
    ft.assert_finite(A.grad, "lanczos truncated grad")
    assert A.grad.abs().sum() > 0
