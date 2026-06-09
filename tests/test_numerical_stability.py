"""Numerical stability: precision, conditioning, determinism.

The eigensolver is the project's numerical heart; these tests fix the precision
contract the differentiable layer will depend on.
"""
import pytest
import torch

from fiedler.graph import gaussian_affinity, laplacian
from fiedler.spectral import jacobi_eigh
from fiedler import testing as ft


def test_jacobi_promotes_to_float64_internally():
    """Even from a float32 default, internal work is float64 -> tight reconstruction."""
    torch.set_default_dtype(torch.float32)
    try:
        M = torch.randn(30, 30)
        A = M + M.t()
        evals, evecs = jacobi_eigh(A)
        recon = (evecs * evals.unsqueeze(0)) @ evecs.t()
        assert (recon - A).abs().max().item() < 1e-4   # f32 output tolerance
    finally:
        torch.set_default_dtype(torch.float64)


def test_determinism_same_seed_same_result():
    def run():
        g = torch.Generator().manual_seed(123)
        X = torch.randn(40, 6, generator=g, dtype=torch.float64)
        L = laplacian(gaussian_affinity(X, sigma=1.5, k=8), "sym")
        return jacobi_eigh(L)
    e1, v1 = run()
    e2, v2 = run()
    assert torch.equal(e1, e2)
    assert torch.equal(v1, v2)


def test_ill_conditioned_wide_dynamic_range():
    """Eigenvalues spanning many orders of magnitude must still reconstruct."""
    torch.set_default_dtype(torch.float64)
    D = torch.diag(torch.tensor([1e-6, 1e-3, 1.0, 1e3, 1e6]))
    Q, _ = torch.linalg.qr(torch.randn(5, 5))
    A = Q @ D @ Q.t()
    A = 0.5 * (A + A.t())
    evals, evecs = jacobi_eigh(A)
    # relative reconstruction error (absolute is dominated by the 1e6 mode)
    recon = (evecs * evals.unsqueeze(0)) @ evecs.t()
    rel = (recon - A).abs().max().item() / A.abs().max().item()
    assert rel < 1e-8   # excellent for condition number ~1e12
    ft.assert_orthonormal(evecs, tol=1e-9)


def test_affinity_no_negative_distance_from_fp_error():
    """The squared-distance expansion ||a||²+||b||²-2a·b can go slightly negative
    in fp; pairwise_sq_dists must clamp so sqrt/exp never see negatives."""
    from fiedler.graph import pairwise_sq_dists
    X = torch.randn(100, 3, dtype=torch.float64) * 1e3
    d2 = pairwise_sq_dists(X)
    assert (d2 >= 0).all()
