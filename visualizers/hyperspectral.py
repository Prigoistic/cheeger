"""Hyperspectral datacube visualisation (e.g. 33-band Hyper-Drive).

A monitor renders 3 channels; a datacube has tens. This maps an (H, W, B) cube
into something viewable four ways:
  * true/false-colour band selection  (pick 3 bands -> RGB)
  * per-band grayscale montage          (inspect every band)
  * PCA-to-RGB                          (first 3 principal components -> RGB)
  * per-pixel spectral signature        (the spectrum at one location)

Uses Spectral Python (SPY) if installed for fast band mapping, otherwise a
self-contained numpy/torch path — no hard dependency.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

try:
    import spectral as _spy
    _HAS_SPY = True
except Exception:
    _HAS_SPY = False


def _cube_np(cube: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(cube, Tensor):
        cube = cube.detach().cpu().numpy()
    return np.asarray(cube, dtype=np.float32)


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)  # robust contrast stretch
    return ((x - lo) / (hi - lo + 1e-8)).clip(0, 1)


def band_rgb(cube, bands=(0, 1, 2)) -> np.ndarray:
    """Map three chosen bands to an RGB image (true- or false-colour)."""
    c = _cube_np(cube)
    rgb = np.stack([_norm(c[..., b]) for b in bands], axis=-1)
    return rgb


def band_montage(cube, max_bands: int = 36) -> plt.Figure:
    """Grayscale grid of every band — see what each wavelength captures."""
    c = _cube_np(cube)
    B = min(c.shape[-1], max_bands)
    cols = int(np.ceil(np.sqrt(B)))
    rows = int(np.ceil(B / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.7 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < B:
            ax.imshow(_norm(c[..., i]), cmap="gray")
            ax.set_title(f"band {i}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    return fig


def pca_rgb(cube) -> np.ndarray:
    """First 3 principal components mapped to RGB — best single false-colour view
    of a high-band cube. SPY's principal_components if available, else torch SVD."""
    c = _cube_np(cube)
    H, W, B = c.shape
    if _HAS_SPY:
        pc = _spy.principal_components(c)
        proj = pc.transform(c)[..., :3]
    else:
        X = torch.from_numpy(c.reshape(-1, B))
        X = X - X.mean(0, keepdim=True)
        # top-3 right singular vectors
        _, _, V = torch.pca_lowrank(X, q=min(3, B))
        proj = (X @ V[:, :3]).reshape(H, W, 3).numpy()
    return np.stack([_norm(proj[..., i]) for i in range(3)], axis=-1)


def spectral_signature(cube, points: list[tuple[int, int]], ax=None) -> plt.Axes:
    """Plot the spectrum (intensity vs band) at one or more pixels."""
    c = _cube_np(cube)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    for (y, x) in points:
        ax.plot(c[y, x, :], label=f"({y},{x})")
    ax.set_xlabel("band"); ax.set_ylabel("intensity"); ax.set_title("spectral signature")
    ax.legend(fontsize=8)
    return ax


def savefig(fig: plt.Figure, name: str, outdir: str = "results") -> str:
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    path = str(out / name)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path
