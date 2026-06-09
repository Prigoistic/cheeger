"""Plotting helpers for spectral structure.

Pure presentation layer — imports from the installed ``fiedler`` package and
renders. Kept outside ``src/`` because it is tooling, not library code.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: write files, never block on a window
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_spectrum(eigvals: Tensor, ax=None, title: str = "Laplacian spectrum"):
    """Bar plot of the eigenvalue ladder — the spectral gap is visible as the
    jump after the near-zero eigenvalues."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ev = eigvals.detach().cpu().numpy()
    ax.bar(range(len(ev)), ev, width=0.8)
    ax.set_xlabel("index")
    ax.set_ylabel("λ")
    ax.set_title(title)
    return ax


def plot_embedding_2d(coords: Tensor, labels: Tensor | None = None, ax=None,
                      title: str = "spectral embedding"):
    """Scatter nodes in their first two non-trivial spectral coordinates
    (v2, v3). Semantically similar nodes collapse together here."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    xy = coords.detach().cpu().numpy()
    c = None if labels is None else labels.detach().cpu().numpy()
    ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="coolwarm", s=24, edgecolors="none")
    ax.set_xlabel("v2 (Fiedler)")
    ax.set_ylabel("v3")
    ax.set_title(title)
    return ax


def savefig(fig, name: str, outdir: str = "results") -> str:
    """Save under results/ and return the path."""
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = str(out / name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
