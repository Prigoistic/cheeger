"""From-scratch Jacobi eigensolver vs numpy oracle."""
import numpy as np
import pytest
import torch

from fiedler.spectral import jacobi_eigh

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("n", [2, 5, 16, 40])
def test_eigenvalues_match_numpy(n):
    torch.manual_seed(n)
    M = torch.randn(n, n)
    A = M + M.t()
    evals, evecs = jacobi_eigh(A)
    ev_np = np.linalg.eigvalsh(A.numpy())
    assert np.allclose(evals.numpy(), ev_np, atol=1e-9)


@pytest.mark.parametrize("n", [3, 12, 30])
def test_reconstruction_and_orthonormality(n):
    torch.manual_seed(n + 1)
    M = torch.randn(n, n)
    A = M + M.t()
    evals, evecs = jacobi_eigh(A)
    recon = (evecs * evals.unsqueeze(0)) @ evecs.t()
    assert torch.allclose(recon, A, atol=1e-8)
    assert torch.allclose(evecs.t() @ evecs, torch.eye(n), atol=1e-9)


def test_ascending_order():
    torch.manual_seed(7)
    M = torch.randn(20, 20)
    evals, _ = jacobi_eigh(M + M.t())
    assert torch.all(evals[1:] >= evals[:-1] - 1e-12)
