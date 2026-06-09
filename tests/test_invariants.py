"""Mathematical invariants that must hold for ALL inputs.

These are property-style tests: we sweep many random instances and canonical
graphs and assert the defining algebraic properties never break. This is the
backbone the differentiable solver / losses will extend.
"""
import numpy as np
import pytest
import torch

from fiedler.graph import gaussian_affinity, laplacian
from fiedler.spectral import jacobi_eigh
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)

SEEDS = [0, 1, 7, 42, 1337]
SIZES = [3, 8, 20, 45]


# --- affinity ------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", SIZES)
def test_affinity_properties(seed, n):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g)
    W = gaussian_affinity(X, sigma=1.5, k=min(8, n - 1))
    ft.assert_finite(W, "W")
    ft.assert_symmetric(W, name="W")
    assert (W >= 0).all(), "affinity must be non-negative"
    assert (W <= 1.0 + 1e-9).all(), "gaussian affinity must be ≤ 1"
    assert torch.diagonal(W).abs().max() < 1e-9, "zero diagonal (no self-loops)"


# --- combinatorial Laplacian --------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", SIZES)
def test_comb_laplacian_properties(seed, n):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g)
    W = gaussian_affinity(X, sigma=1.5, k=min(8, n - 1))
    L = laplacian(W, kind="comb")
    ft.assert_finite(L, "L")
    ft.assert_symmetric(L, name="L_comb")
    ft.assert_rows_sum_zero(L)
    ft.assert_psd(L, name="L_comb")


# --- symmetric normalized Laplacian -------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n", SIZES)
def test_sym_laplacian_spectrum(seed, n):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 5, generator=g)
    W = gaussian_affinity(X, sigma=1.5, k=min(8, n - 1))
    L = laplacian(W, kind="sym")
    ft.assert_finite(L, "L_sym")
    ft.assert_symmetric(L, name="L_sym")
    ft.assert_spectrum_in(L, 0.0, 2.0, name="L_sym")


# --- connected components == zero eigenvalues (comb) --------------------- #
@pytest.mark.parametrize("sizes", [[4], [3, 5], [2, 2, 2], [6, 1, 3]])
def test_components_equal_zero_eigs(sizes):
    W, meta = ft.disconnected_blocks(sizes)
    L = laplacian(W, kind="comb")
    assert ft.count_zero_eigs(L) == meta["components"]


# --- Jacobi solver vs numpy, over canonical + random graphs -------------- #
@pytest.mark.parametrize("gen", ["path", "cycle", "complete", "random"])
@pytest.mark.parametrize("n", [4, 11, 30])
def test_jacobi_matches_numpy(gen, n):
    if gen == "path":
        A, _ = ft.path_graph(n); A = laplacian(A, kind="comb")
    elif gen == "cycle":
        A, _ = ft.cycle_graph(n); A = laplacian(A, kind="comb")
    elif gen == "complete":
        A, _ = ft.complete_graph(n); A = laplacian(A, kind="comb")
    else:
        A = ft.random_symmetric(n, seed=n)
    evals, evecs = jacobi_eigh(A)
    ft.assert_finite(evals, "evals"); ft.assert_finite(evecs, "evecs")
    ft.assert_ascending(evals)
    ft.assert_orthonormal(evecs, tol=1e-8)
    ft.assert_eig_reconstruction(A, evals, evecs, tol=1e-7)
    np.testing.assert_allclose(
        evals.numpy(), np.linalg.eigvalsh(A.numpy()), atol=1e-8)
