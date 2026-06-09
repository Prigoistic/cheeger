"""Static segmentation debugging panels (matplotlib).

The uniform across all workflows: side-by-side [Raw | Ground Truth | Prediction],
plus a translucent overlay and a per-class isolation montage (the "hide roads to
inspect trees" view from interactive dashboards, rendered statically).
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .palette import colorize, cityscapes_palette, CITYSCAPES_CLASSES


def _np_img(image: Tensor) -> np.ndarray:
    img = image.detach().cpu()
    if img.dim() == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
        img = img.permute(1, 2, 0)
    img = img.float().numpy()
    if img.max() > 1.5:
        img = img / 255.0
    return img.clip(0, 1)


def triptych(image: Tensor, gt: Tensor, pred: Tensor, palette: Tensor | None = None,
             title: str = "") -> plt.Figure:
    """[Raw | GT | Pred] in one row — the standard debug view."""
    palette = cityscapes_palette() if palette is None else palette
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(_np_img(image));                     axes[0].set_title("Raw")
    axes[1].imshow(colorize(gt, palette).cpu().numpy()); axes[1].set_title("Ground Truth")
    axes[2].imshow(colorize(pred, palette).cpu().numpy()); axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    return fig


def overlay(image: Tensor, mask: Tensor, palette: Tensor | None = None,
            alpha: float = 0.5, title: str = "overlay") -> plt.Figure:
    """Translucent colorized mask over the raw image."""
    palette = cityscapes_palette() if palette is None else palette
    base = _np_img(image)
    col = colorize(mask, palette).float().cpu().numpy() / 255.0
    blended = (1 - alpha) * base + alpha * col
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(blended.clip(0, 1)); ax.set_title(title); ax.axis("off")
    return fig


def class_montage(pred: Tensor, class_ids: list[int] | None = None,
                  names: list[str] | None = None) -> plt.Figure:
    """One binary panel per class — the 'isolate a single class' inspection view."""
    names = names or CITYSCAPES_CLASSES
    present = torch.unique(pred).tolist()
    class_ids = class_ids or [c for c in present if 0 <= c < len(names)]
    if not class_ids:
        class_ids = [0]
    n = len(class_ids)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), squeeze=False)
    for i, cid in enumerate(class_ids):
        ax = axes[i // cols][i % cols]
        ax.imshow((pred == cid).float().cpu().numpy(), cmap="magma", vmin=0, vmax=1)
        ax.set_title(names[cid] if cid < len(names) else f"class {cid}", fontsize=9)
        ax.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    return fig


def savefig(fig: plt.Figure, name: str, outdir: str = "results") -> str:
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    path = str(out / name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
