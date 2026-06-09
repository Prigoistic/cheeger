"""Validate the from-scratch core against numpy/scipy oracles.

Run:  python scripts/validate_core.py

Checks
------
1. Jacobi eigensolver vs numpy.linalg.eigh on random symmetric matrices.
2. Laplacian spectral facts: smallest eigenvalue == 0, multiplicity == #components.
3. L_sym spectrum lies in [0, 2].
4. Fiedler vector sign-partition recovers a planted 2-cluster split.
"""
import sys
import pathlib

# allow running before `pip install -e .`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from fiedler.graph import gaussian_affinity, laplacian
from fiedler.spectral import jacobi_eigh

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def check(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def test_jacobi_vs_numpy():
    print("1. Jacobi eigensolver vs numpy.linalg.eigh")
    all_ok = True
    for n in (3, 8, 25, 60):
        M = torch.randn(n, n)
        A = M + M.t()                                   # symmetric
        evals, evecs = jacobi_eigh(A)
        ev_np = np.linalg.eigvalsh(A.numpy())
        eval_err = np.abs(evals.numpy() - ev_np).max()
        # reconstruction error: V diag(λ) V^T  ==  A
        recon = (evecs * evals.unsqueeze(0)) @ evecs.t()
        recon_err = (recon - A).abs().max().item()
        ortho_err = (evecs.t() @ evecs - torch.eye(n)).abs().max().item()
        ok = eval_err < 1e-9 and recon_err < 1e-9 and ortho_err < 1e-9
        all_ok &= check(
            f"n={n:3d}",
            ok,
            f"λ_err={eval_err:.1e}  recon={recon_err:.1e}  ortho={ortho_err:.1e}",
        )
    return all_ok


def test_laplacian_spectrum():
    print("2. Laplacian spectral identities")
    all_ok = True

    # two disconnected triangles -> exactly 2 zero eigenvalues (comb Laplacian)
    W = torch.zeros(6, 6)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W[a, b] = W[b, a] = 1.0
    L = laplacian(W, kind="comb")
    evals, _ = jacobi_eigh(L)
    n_zero = int((evals.abs() < 1e-8).sum())
    all_ok &= check("2 components -> 2 zero eigenvalues", n_zero == 2, f"got {n_zero}")
    all_ok &= check("λ1 == 0", evals[0].abs().item() < 1e-9, f"λ1={evals[0]:.2e}")

    # L_sym spectrum in [0, 2]
    X = torch.randn(40, 5)
    Ls = laplacian(gaussian_affinity(X, sigma=1.5, k=8), kind="sym")
    evals, _ = jacobi_eigh(Ls)
    in_range = bool((evals.min() > -1e-8) and (evals.max() < 2.0 + 1e-6))
    all_ok &= check(
        "L_sym spectrum in [0, 2]", in_range, f"[{evals.min():.3f}, {evals.max():.3f}]"
    )
    return all_ok


def test_fiedler_partition():
    print("3. Fiedler vector recovers a planted 2-cluster split")
    # two gaussian blobs in feature space
    g0 = torch.randn(30, 4) + torch.tensor([5.0, 0, 0, 0])
    g1 = torch.randn(30, 4) + torch.tensor([-5.0, 0, 0, 0])
    X = torch.cat([g0, g1], dim=0)
    true = torch.cat([torch.zeros(30), torch.ones(30)])

    W = gaussian_affinity(X, sigma=2.0, k=10)
    L = laplacian(W, kind="sym")
    evals, evecs = jacobi_eigh(L)
    fiedler = evecs[:, 1]                                # second-smallest eigenvector
    pred = (fiedler > 0).double()
    # label sign is arbitrary -> take best of pred / 1-pred
    acc = max((pred == true).float().mean().item(), ((1 - pred) == true).float().mean().item())
    return check("partition accuracy == 1.0", acc > 0.99, f"acc={acc:.3f}  λ2={evals[1]:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("fiedler core validation")
    print("=" * 60)
    results = [test_jacobi_vs_numpy(), test_laplacian_spectrum(), test_fiedler_partition()]
    print("=" * 60)
    if all(results):
        print(f"{GREEN}ALL CHECKS PASSED{RESET}")
        sys.exit(0)
    print(f"{RED}SOME CHECKS FAILED{RESET}")
    sys.exit(1)
