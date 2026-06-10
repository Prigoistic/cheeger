"""Train a segmentation model from a config.

    python scripts/train.py                          # toy data, conv head, defaults
    python scripts/train.py --head spectral --epochs 20
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --data cityscapes --root datasets/cityscapes

The spectral head's eigensolver runs on CPU/CUDA (not MPS), so spectral runs default
to CPU locally; pass --device cuda on a GPU box.
"""
import sys
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from fiedler.engine import Config, Trainer, build_model, build_dataset, build_criterion
from fiedler.engine import ExperimentLogger
from fiedler.data import transforms as T
from fiedler.utils import seed_everything, get_device


def pick_device(requested, head):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # eigh unsupported on MPS; CPU is the safe local default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--head", default=None, choices=["conv", "spectral"])
    ap.add_argument("--data", default=None, choices=["toy", "cityscapes"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.head: cfg.model.head = args.head
    if args.data: cfg.data.name = args.data
    if args.root: cfg.data.root = args.root
    if args.epochs is not None: cfg.optim.epochs = args.epochs
    if cfg.data.name == "toy":
        cfg.data.num_classes = 19

    seed_everything(cfg.seed)
    device = pick_device(args.device, cfg.model.head)
    print(f"experiment={cfg.experiment}  head={cfg.model.head}  data={cfg.data.name}  device={device}")

    tf = T.Compose([T.RandomHorizontalFlip(0.5), T.Normalize()])
    train_ds = build_dataset(cfg, "train", transform=tf)
    val_ds = build_dataset(cfg, "val", transform=T.Normalize())
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)

    model = build_model(cfg)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr,
                                 weight_decay=cfg.optim.weight_decay)
    logger = ExperimentLogger(str(pathlib.Path("results/runs") / cfg.experiment))

    trainer = Trainer(
        model, criterion, optimizer, device=device, num_classes=cfg.data.num_classes,
        feat_dim=cfg.model.feat_dim, graph_resolution=cfg.model.spectral.graph_resolution,
        logger=logger, warmup=(cfg.loss.rayleigh_warmup_start, cfg.loss.rayleigh_warmup_end),
        experiment=cfg.experiment,
    )
    history, best = trainer.fit(train_loader, val_loader, epochs=cfg.optim.epochs)
    logger.close()
    print(f"done. best val mIoU = {best:.4f}")


if __name__ == "__main__":
    main()
