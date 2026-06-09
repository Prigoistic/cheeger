"""Exercise the full visualizer suite on synthetic data (no trained model needed).

Run:  python demos/viz_demo.py
Writes to results/:
  seg_triptych.png, seg_overlay.png, seg_class_montage.png,
  hsi_pca_rgb.png, hsi_band_montage.png, hsi_signature.png
and a TensorBoard run under results/runs/viz_demo/  (view: tensorboard --logdir results/runs)
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fiedler.utils import seed_everything
from fiedler.engine.logging import ExperimentLogger
from visualizers import seg_panels, hyperspectral
from visualizers.palette import cityscapes_palette


def fake_scene(H=128, W=256, K=19):
    """A crude driving-like label map: sky / building / road bands + a 'car' blob."""
    gt = torch.zeros(H, W, dtype=torch.long)
    gt[: H // 3] = 10            # sky
    gt[H // 3 : 2 * H // 3] = 2  # building
    gt[2 * H // 3 :] = 0         # road
    gt[H // 2 : H // 2 + 25, W // 2 : W // 2 + 50] = 13  # car
    # raw image = palette colours + noise; pred = gt with errors
    palette = cityscapes_palette()
    img = palette[gt].float() / 255.0 + 0.06 * torch.randn(H, W, 3)
    pred = gt.clone()
    noise = torch.rand(H, W) < 0.08
    pred[noise] = torch.randint(0, K, (int(noise.sum()),))
    return img.clamp(0, 1), gt, pred, palette


def fake_cube(H=96, W=128, B=33):
    """A 33-band Hyper-Drive-like cube: smooth spectral gradients + structure."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    bands = []
    for b in range(B):
        phase = b / B
        bands.append(torch.sin(6 * (xx + phase)) * torch.cos(4 * (yy + phase)) + 0.3 * torch.randn(H, W))
    return torch.stack(bands, dim=-1)


def main():
    seed_everything(0)
    img, gt, pred, palette = fake_scene()

    # --- segmentation panels ---
    seg_panels.savefig(seg_panels.triptych(img, gt, pred, palette, "synthetic scene"),
                       "seg_triptych.png")
    seg_panels.savefig(seg_panels.overlay(img, pred, palette, alpha=0.55, title="pred overlay"),
                       "seg_overlay.png")
    seg_panels.savefig(seg_panels.class_montage(pred), "seg_class_montage.png")

    # --- hyperspectral cube ---
    cube = fake_cube()
    hyperspectral.savefig(
        _imshow(hyperspectral.pca_rgb(cube), "PCA -> RGB (33 bands)"), "hsi_pca_rgb.png")
    hyperspectral.savefig(hyperspectral.band_montage(cube, max_bands=33), "hsi_band_montage.png")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    hyperspectral.spectral_signature(cube, [(20, 30), (70, 100)], ax=ax)
    hyperspectral.savefig(fig, "hsi_signature.png")

    # --- training dashboard logger (TensorBoard) with a fake learning curve ---
    logger = ExperimentLogger(str(ROOT / "results/runs/viz_demo"))
    for step in range(20):
        loss = float(np.exp(-step / 6) + 0.05 * np.random.rand())
        miou = float(0.3 + 0.5 * (1 - np.exp(-step / 6)))
        logger.log_scalars(step, **{"train/loss": loss, "val/mIoU": miou})
    logger.log_seg("val/sample", img, gt, pred, palette, step=19)
    logger.close()

    print("wrote panels + TensorBoard run to results/  (tensorboard --logdir results/runs)")


def _imshow(rgb, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(rgb); ax.set_title(title); ax.axis("off")
    return fig


if __name__ == "__main__":
    main()
