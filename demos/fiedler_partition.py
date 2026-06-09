"""Demo: spectral partitioning of two feature-space blobs, end to end.

Builds a learned-metric-ready Gaussian affinity graph over synthetic features,
forms the symmetric normalised Laplacian, solves it with our from-scratch Jacobi
eigensolver, and visualises (a) the spectrum and (b) the 2-D spectral embedding
coloured by the Fiedler sign-partition.

Run:  python demos/fiedler_partition.py   ->   writes results/fiedler_partition.png
"""
import sys
import pathlib

# allow running before `pip install -e .`
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt

from fiedler.graph import gaussian_affinity, laplacian
from fiedler.spectral import jacobi_eigh
from fiedler.utils import seed_everything
from visualizers.spectral_viz import plot_spectrum, plot_embedding_2d, savefig


def main():
    seed_everything(0)
    torch.set_default_dtype(torch.float64)

    # two well-separated gaussian blobs in 4-D feature space
    n = 40
    g0 = torch.randn(n, 4) + torch.tensor([5.0, 0, 0, 0])
    g1 = torch.randn(n, 4) + torch.tensor([-5.0, 0, 0, 0])
    X = torch.cat([g0, g1], dim=0)
    labels = torch.cat([torch.zeros(n), torch.ones(n)])

    W = gaussian_affinity(X, sigma=2.0, k=10)
    L = laplacian(W, kind="sym")
    eigvals, eigvecs = jacobi_eigh(L)

    fiedler = eigvecs[:, 1]
    partition = (fiedler > 0).double()
    acc = max((partition == labels).float().mean().item(),
              ((1 - partition) == labels).float().mean().item())
    print(f"spectral gap λ2={eigvals[1]:.4f}  λ3={eigvals[2]:.4f}  partition acc={acc:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_spectrum(eigvals[:12], ax=ax1, title="L_sym spectrum (first 12)")
    plot_embedding_2d(eigvecs[:, 1:3], labels=labels, ax=ax2,
                      title=f"spectral embedding (acc={acc:.2f})")
    path = savefig(fig, "fiedler_partition.png")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
