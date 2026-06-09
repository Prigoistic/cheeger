"""Unified experiment logger — one API over TensorBoard and (optional) Weights & Biases.

The training loop calls this; it never imports a dashboard backend directly. Both
backends are optional: TensorBoard is used if installed, W&B only if explicitly
enabled *and* importable. With neither, it degrades to console scalar prints, so
code that logs never crashes on a bare environment.

    logger = ExperimentLogger("results/runs/exp1", use_wandb=False)
    logger.log_scalars(step, loss=0.3, miou=0.51)
    logger.log_seg("val/sample", image, gt, pred, palette, step)   # image overlay grid
    logger.close()
"""
from __future__ import annotations

import pathlib

import torch
from torch import Tensor

try:  # TensorBoard ships with torch
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:  # pragma: no cover
    _HAS_TB = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


class ExperimentLogger:
    def __init__(
        self,
        logdir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        project: str | None = "fiedler",
        run_name: str | None = None,
        config: dict | None = None,
    ):
        self.logdir = pathlib.Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._tb = None
        self._wandb = None

        if use_tensorboard and _HAS_TB:
            self._tb = SummaryWriter(str(self.logdir))
        if use_wandb:
            if _HAS_WANDB:
                self._wandb = wandb.init(
                    project=project, name=run_name, dir=str(self.logdir), config=config
                )
            else:
                print("[ExperimentLogger] wandb requested but not installed — skipping.")

    # --- scalars ---------------------------------------------------------- #
    def log_scalars(self, step: int, **values: float) -> None:
        for k, v in values.items():
            v = float(v)
            if self._tb:
                self._tb.add_scalar(k, v, step)
        if self._wandb:
            self._wandb.log({**{k: float(v) for k, v in values.items()}, "step": step})
        if not self._tb and not self._wandb:
            msg = "  ".join(f"{k}={float(v):.4f}" for k, v in values.items())
            print(f"[step {step}] {msg}")

    # --- images ----------------------------------------------------------- #
    def log_image(self, tag: str, img: Tensor, step: int) -> None:
        """img: CHW or HWC float[0,1] / uint8."""
        chw = _to_chw(img)
        if self._tb:
            self._tb.add_image(tag, chw, step)
        if self._wandb:
            self._wandb.log({tag: wandb.Image(chw.permute(1, 2, 0).cpu().numpy()), "step": step})

    def log_seg(
        self,
        tag: str,
        image: Tensor,
        gt: Tensor,
        pred: Tensor,
        palette: Tensor,
        step: int,
    ) -> None:
        """Log a [image | colorized GT | colorized pred] strip in one panel."""
        from_palette = lambda m: _colorize(m, palette)
        img = _to_chw(image).float()
        if img.max() > 1.5:
            img = img / 255.0
        strip = torch.cat([img, from_palette(gt), from_palette(pred)], dim=2)  # along width
        self.log_image(tag, strip, step)

    def close(self) -> None:
        if self._tb:
            self._tb.flush()
            self._tb.close()
        if self._wandb:
            self._wandb.finish()


# --------------------------------------------------------------------------- #
def _to_chw(img: Tensor) -> Tensor:
    if img.dim() == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)
    elif img.dim() == 3 and img.shape[0] not in (1, 3):  # HWC -> CHW
        img = img.permute(2, 0, 1)
    return img


def _colorize(mask: Tensor, palette: Tensor) -> Tensor:
    """mask: (H,W) int labels -> (3,H,W) float[0,1] using palette (K,3) uint8."""
    mask = mask.long().clamp_min(0)
    rgb = palette.to(mask.device)[mask]          # (H, W, 3)
    return rgb.permute(2, 0, 1).float() / 255.0
