"""Device-agnostic training / evaluation loop, plus builders that turn a Config
into a model, dataset and loss.

The loop is head-blind. A conv head returns only ``logits`` and trains on plain
cross-entropy; a spectral head also returns graph-resolution ``prob_graph`` /
``laplacian`` / ``response``, which the loop forwards to the composite loss so the
Rayleigh and bulk terms fire. The Rayleigh weight is warmed up over a configurable
step window (it is harmful before the features organise).
"""
from __future__ import annotations

import pathlib

import torch
from torch import nn

from ..models import UNet, ConvHead, SpectralSegHead, SegModel
from ..losses import CompositeLoss
from ..metrics import SegMetrics, boundary_iou
from ..data import ToyDrivingDataset, CityscapesDataset
from ..data import transforms as T
from .config import Config


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_model(cfg: Config) -> SegModel:
    m, s = cfg.model, cfg.model.spectral
    backbone = UNet(3, base=m.base, depth=m.depth, out_channels=m.feat_dim)
    if m.head == "conv":
        head = ConvHead(m.feat_dim, cfg.data.num_classes)
    elif m.head == "spectral":
        head = SpectralSegHead(
            m.feat_dim, cfg.data.num_classes, k=s.k_eigenvecs, graph_hw=s.graph_resolution,
            knn=s.knn, laplacian_kind=s.laplacian, learn_metric=(s.affinity == "learned"),
            mlp_hidden=s.mlp_hidden, solver=s.solver, lanczos_m=s.lanczos_m,
        )
    else:
        raise ValueError(f"unknown head {m.head!r}")
    return SegModel(backbone, head)


def build_dataset(cfg: Config, split: str = "train", transform=None):
    if cfg.data.name == "toy":
        seed = cfg.seed + (0 if split == "train" else 10_000)
        return ToyDrivingDataset(n=cfg.data.n_toy, size=cfg.data.crop_size, seed=seed,
                                 transform=transform)
    if cfg.data.name == "cityscapes":
        return CityscapesDataset(cfg.data.root, split=split, transform=transform)
    raise ValueError(f"unknown dataset {cfg.data.name!r}")


def build_criterion(cfg: Config, class_weight=None) -> CompositeLoss:
    return CompositeLoss(
        w_ce=cfg.loss.cross_entropy, w_rayleigh=cfg.loss.spectral_consistency,
        w_bulk=cfg.loss.bulk, ignore_index=255, class_weight=class_weight,
    )


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model, criterion, optimizer, *, device="cpu", num_classes=19,
                 feat_dim=32, graph_resolution=32, ignore_index=255, logger=None,
                 warmup=(0, 0), ckpt_dir="results/ckpt", experiment="exp"):
        self.model = model.to(device)
        self.criterion = criterion
        self.opt = optimizer
        self.device = device
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.graph_res = graph_resolution
        self.ignore_index = ignore_index
        self.logger = logger
        self.warmup_start, self.warmup_end = warmup
        self.w_rayleigh_base = criterion.w_rayleigh
        self.want_bulk = criterion.w_bulk > 0
        self.ckpt_dir = pathlib.Path(ckpt_dir)
        self.experiment = experiment
        self.global_step = 0

    def _warmup_weight(self) -> float:
        s, e = self.warmup_start, self.warmup_end
        if e <= s:
            return self.w_rayleigh_base
        if self.global_step <= s:
            return 0.0
        return self.w_rayleigh_base * min(1.0, (self.global_step - s) / (e - s))

    def _spectral_kwargs(self, out: dict) -> dict:
        kw = {}
        if "laplacian" in out and "prob_graph" in out:
            kw["prob_graph"] = out["prob_graph"]
            kw["L"] = out["laplacian"]
        if self.want_bulk and "response" in out:
            kw["response_h"] = out["response"]
            kw["eigvals"] = out["eigvals"]
            kw["mp_n"] = self.graph_res * self.graph_res
            kw["mp_d"] = self.feat_dim
        return kw

    def train_one_epoch(self, loader, log_every: int = 20) -> float:
        self.model.train()
        running = 0.0
        for img, label in loader:
            img, label = img.to(self.device), label.to(self.device)
            self.opt.zero_grad()
            out = self.model(img)
            self.criterion.w_rayleigh = self._warmup_weight()
            loss, comps = self.criterion(out["logits"], label, **self._spectral_kwargs(out))
            loss.backward()
            self.opt.step()
            running += float(loss.detach())
            if self.logger is not None and self.global_step % log_every == 0:
                self.logger.log_scalars(self.global_step,
                                        **{f"train/{k}": v for k, v in comps.items()},
                                        **{"train/w_rayleigh": self.criterion.w_rayleigh})
            self.global_step += 1
        return running / max(1, len(loader))

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        sm = SegMetrics(self.num_classes, self.ignore_index)
        biou = []
        for img, label in loader:
            img, label = img.to(self.device), label.to(self.device)
            pred = self.model(img)["logits"].argmax(1)
            sm.update(pred, label)
            for b in range(pred.shape[0]):
                v = boundary_iou(pred[b], label[b], self.num_classes, ignore_index=self.ignore_index)
                if v == v:  # skip NaN
                    biou.append(v)
        self.model.train()
        out = sm.compute()
        out["boundary_iou"] = sum(biou) / len(biou) if biou else float("nan")
        return out

    def fit(self, train_loader, val_loader=None, epochs: int = 10, log_every: int = 20):
        best = -1.0
        history = []
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, log_every=log_every)
            row = {"epoch": epoch, "train_loss": train_loss}
            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                row.update({f"val_{k}": metrics[k] for k in ("mIoU", "pixel_acc", "boundary_iou")})
                if self.logger is not None:
                    self.logger.log_scalars(self.global_step, **{
                        "val/mIoU": metrics["mIoU"], "val/pixel_acc": metrics["pixel_acc"],
                        "val/boundary_iou": metrics["boundary_iou"]})
                if metrics["mIoU"] == metrics["mIoU"] and metrics["mIoU"] > best:
                    best = metrics["mIoU"]
                    self.save_checkpoint(epoch, best)
            history.append(row)
        return history, best

    def save_checkpoint(self, epoch: int, best: float) -> str:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = str(self.ckpt_dir / f"{self.experiment}_best.pt")
        torch.save({"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                    "epoch": epoch, "best_mIoU": best}, path)
        return path
