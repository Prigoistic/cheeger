"""Laplacian operator identities."""
import torch

from fiedler.graph import gaussian_affinity, laplacian, degree
from fiedler.spectral import jacobi_eigh

torch.set_default_dtype(torch.float64)


def _two_triangles():
    W = torch.zeros(6, 6)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W[a, b] = W[b, a] = 1.0
    return W


def test_comb_rows_sum_to_zero():
    L = laplacian(_two_triangles(), kind="comb")
    assert torch.allclose(L.sum(dim=1), torch.zeros(6), atol=1e-10)


def test_components_equal_zero_eigenvalues():
    L = laplacian(_two_triangles(), kind="comb")
    evals, _ = jacobi_eigh(L)
    n_zero = int((evals.abs() < 1e-8).sum())
    assert n_zero == 2


def test_sym_spectrum_in_unit_interval():
    torch.manual_seed(0)
    X = torch.randn(50, 6)
    L = laplacian(gaussian_affinity(X, sigma=1.5, k=8), kind="sym")
    evals, _ = jacobi_eigh(L)
    assert evals.min() > -1e-8
    assert evals.max() < 2.0 + 1e-6


def test_affinity_symmetric_zero_diagonal():
    torch.manual_seed(1)
    X = torch.randn(30, 4)
    W = gaussian_affinity(X, sigma=1.0, k=5)
    assert torch.allclose(W, W.t(), atol=1e-12)
    assert torch.allclose(torch.diag(W), torch.zeros(30), atol=1e-12)


def test_learned_metric_changes_graph():
    """A non-trivial metric must change the affinity — the hook Novelty #1 rides on."""
    torch.manual_seed(2)
    X = torch.randn(20, 4)
    w_iso = gaussian_affinity(X, sigma=1.0, k=5)
    metric = torch.tensor([3.0, 0.1, 0.1, 0.1])  # emphasise channel 0
    w_aniso = gaussian_affinity(X, sigma=1.0, k=5, metric=metric)
    assert not torch.allclose(w_iso, w_aniso, atol=1e-3)
