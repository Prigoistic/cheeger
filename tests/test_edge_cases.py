"""Adversarial / degenerate inputs — the cases that crash naive implementations.

Every one of these previously had the potential to NaN, divide by zero, or return
non-orthonormal eigenvectors. They are pinned here so a regression is caught loud.
"""
import pytest
import torch

from fiedler.graph import gaussian_affinity, laplacian, pairwise_sq_dists
from fiedler.spectral import jacobi_eigh
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)


def test_isolated_nodes_no_nan():
    """Zero-degree nodes must not produce NaN/Inf through degree normalization."""
    W, _ = ft.complete_graph(3)
    W = ft.with_isolated(W, n_isolated=2)
    for kind in ("comb", "sym", "rw"):
        L = laplacian(W, kind=kind)
        ft.assert_finite(L, f"L_{kind} with isolated nodes")


def test_isolated_nodes_sym_documented_behavior():
    """L_sym gives an isolated node eigenvalue 1 (not 0): #zero-eigs != #components
    when isolated nodes exist. Component counting must use the comb Laplacian."""
    W, _ = ft.path_graph(2)
    W = ft.with_isolated(W, n_isolated=2)        # 3 components: {0,1}, {2}, {3}
    assert ft.count_zero_eigs(laplacian(W, "comb")) == 3
    # comb sees 3 components; sym does not (isolated -> eigenvalue 1)
    assert ft.count_zero_eigs(laplacian(W, "sym")) == 1


def test_identical_features_no_nan():
    """All-identical features -> zero pairwise distances -> must stay finite."""
    X = torch.ones(6, 4)
    d2 = pairwise_sq_dists(X)
    ft.assert_finite(d2, "d2")
    assert d2.abs().max() < 1e-9
    W = gaussian_affinity(X, sigma=1.0, k=3)
    ft.assert_finite(W, "W")
    ft.assert_finite(laplacian(W, "sym"), "L_sym")


def test_single_and_two_nodes():
    for n in (1, 2):
        W = gaussian_affinity(torch.randn(n, 3), sigma=1.0, k=3)
        assert W.shape == (n, n)
        ft.assert_finite(laplacian(W, "comb"), f"L n={n}")


def test_k_larger_than_n():
    """Requesting more neighbours than nodes must clamp, not crash."""
    X = torch.randn(5, 3)
    W = gaussian_affinity(X, sigma=1.0, k=50)
    ft.assert_finite(W, "W"); ft.assert_symmetric(W, name="W")


@pytest.mark.parametrize("sigma", [1e-4, 1e-2, 1.0, 1e2, 1e4])
def test_extreme_sigma_finite(sigma):
    X = torch.randn(20, 4) * 10
    W = gaussian_affinity(X, sigma=sigma, k=6)
    ft.assert_finite(W, f"W sigma={sigma}")
    ft.assert_symmetric(W, name="W")


@pytest.mark.parametrize("gen", [ft.complete_graph, ft.cycle_graph])
def test_degenerate_spectrum_orthonormal(gen):
    """Highly symmetric graphs have repeated eigenvalues; eigenvectors must still
    come out orthonormal (the failure mode of naive deflation)."""
    W, _ = gen(8)
    L = laplacian(W, kind="comb")
    evals, evecs = jacobi_eigh(L)
    ft.assert_orthonormal(evecs, tol=1e-8)
    ft.assert_eig_reconstruction(L, evals, evecs, tol=1e-7)


def test_learnable_metric_gradients_flow():
    """The Novelty-#1 hook: distance through a learnable metric must be
    differentiable and produce finite gradients."""
    X = torch.randn(15, 4)
    metric = torch.ones(4, requires_grad=True)
    W = gaussian_affinity(X, sigma=1.0, k=5, metric=metric)
    loss = laplacian(W, "sym").diagonal().sum()
    loss.backward()
    assert metric.grad is not None
    ft.assert_finite(metric.grad, "metric.grad")
