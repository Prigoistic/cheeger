"""Controlled head comparison: conv vs spectral, everything else identical.

Trains both heads on the same data with the same backbone, optimiser and schedule,
then reports final validation mIoU and boundary-IoU side by side. This is the core
experiment of the project in miniature.

    python scripts/benchmark.py --epochs 15 --data toy
"""
import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from fiedler.engine import Config, Trainer, build_model, build_dataset, build_criterion
from fiedler.data import transforms as T
from fiedler.utils import seed_everything


def run_head(cfg: Config, head: str, train_loader, val_loader, device):
    cfg.model.head = head
    cfg.experiment = f"bench-{head}"
    seed_everything(cfg.seed)                       # identical init conditions per head
    model = build_model(cfg)
    criterion = build_criterion(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    trainer = Trainer(model, criterion, opt, device=device, num_classes=cfg.data.num_classes,
                      feat_dim=cfg.model.feat_dim,
                      graph_resolution=cfg.model.spectral.graph_resolution,
                      warmup=(cfg.loss.rayleigh_warmup_start, cfg.loss.rayleigh_warmup_end))
    trainer.fit(train_loader, val_loader, epochs=cfg.optim.epochs)
    return trainer.evaluate(val_loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="toy")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.data.name = args.data
    cfg.data.num_classes = 19
    cfg.optim.epochs = args.epochs

    seed_everything(cfg.seed)
    tf = T.Compose([T.RandomHorizontalFlip(0.5), T.Normalize()])
    train_loader = DataLoader(build_dataset(cfg, "train", tf), batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(build_dataset(cfg, "val", T.Normalize()), batch_size=cfg.data.batch_size)

    print(f"benchmark on {args.data}  ({args.epochs} epochs, device={args.device})\n" + "=" * 56)
    results = {}
    for head in ("conv", "spectral"):
        m = run_head(Config.from_dict(cfg.to_dict()), head, train_loader, val_loader, args.device)
        results[head] = m
        print(f"{head:9s}  mIoU={m['mIoU']:.4f}  pixel_acc={m['pixel_acc']:.4f}  "
              f"boundary_iou={m['boundary_iou']:.4f}")
    print("=" * 56)
    d = results["spectral"]["mIoU"] - results["conv"]["mIoU"]
    print(f"spectral - conv  ΔmIoU = {d:+.4f}")


if __name__ == "__main__":
    main()
