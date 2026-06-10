"""Config round-trip + trainer loop (builders, fit, evaluate, warmup)."""
import pathlib

import pytest
import torch
from torch.utils.data import DataLoader

from fiedler.engine import Config, Trainer, build_model, build_dataset, build_criterion
from fiedler.engine.trainer import Trainer as _Trainer

pytestmark = pytest.mark.float32


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_defaults_and_nesting():
    cfg = Config()
    assert cfg.model.spectral.k_eigenvecs == 16
    assert cfg.data.num_classes == 19


def test_config_from_dict_ignores_unknown_keys():
    d = {"experiment": "x", "model": {"head": "spectral", "encoder": "resnet34",
         "spectral": {"k_eigenvecs": 8, "solver": "lanczos", "bogus": 1}}}
    cfg = Config.from_dict(d)
    assert cfg.experiment == "x"
    assert cfg.model.head == "spectral"
    assert cfg.model.spectral.k_eigenvecs == 8 and cfg.model.spectral.solver == "lanczos"


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.model.head = "spectral"; cfg.optim.epochs = 7
    p = tmp_path / "c.yaml"
    cfg.to_yaml(str(p))
    back = Config.from_yaml(str(p))
    assert back.model.head == "spectral" and back.optim.epochs == 7


def test_loads_repo_default_yaml():
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(str(root / "configs/default.yaml"))
    assert cfg.data.num_classes == 19
    assert cfg.model.spectral.k_eigenvecs == 16


# --------------------------------------------------------------------------- #
# builders + trainer
# --------------------------------------------------------------------------- #
def _toy_cfg(head="conv"):
    cfg = Config()
    cfg.data.name = "toy"; cfg.data.n_toy = 4; cfg.data.crop_size = (48, 64)
    cfg.data.batch_size = 2; cfg.data.num_classes = 19
    cfg.model.head = head; cfg.model.base = 8; cfg.model.depth = 2; cfg.model.feat_dim = 8
    cfg.model.spectral.graph_resolution = 10; cfg.model.spectral.k_eigenvecs = 6
    cfg.model.spectral.knn = 8; cfg.model.spectral.mlp_hidden = 16
    cfg.optim.epochs = 2
    return cfg


def _loaders(cfg):
    tr = DataLoader(build_dataset(cfg, "train"), batch_size=cfg.data.batch_size)
    va = DataLoader(build_dataset(cfg, "val"), batch_size=cfg.data.batch_size)
    return tr, va


def test_build_model_conv_and_spectral():
    conv = build_model(_toy_cfg("conv"))
    spec = build_model(_toy_cfg("spectral"))
    x = torch.randn(1, 3, 48, 64)
    assert conv(x)["logits"].shape == (1, 19, 48, 64)
    assert spec(x)["logits"].shape == (1, 19, 48, 64)
    assert "laplacian" in spec(x)


def test_trainer_conv_runs_and_loss_finite():
    cfg = _toy_cfg("conv")
    model, crit = build_model(cfg), build_criterion(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    tr, va = _loaders(cfg)
    trainer = Trainer(model, crit, opt, device="cpu", num_classes=19,
                      feat_dim=cfg.model.feat_dim, graph_resolution=10)
    history, best = trainer.fit(tr, va, epochs=2)
    assert len(history) == 2
    assert all(r["train_loss"] == r["train_loss"] for r in history)   # finite
    metrics = trainer.evaluate(va)
    assert "mIoU" in metrics and "boundary_iou" in metrics


def test_trainer_spectral_forwards_spectral_loss_terms():
    cfg = _toy_cfg("spectral")
    cfg.loss.spectral_consistency = 0.1
    model, crit = build_model(cfg), build_criterion(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    tr, _ = _loaders(cfg)
    trainer = Trainer(model, crit, opt, device="cpu", num_classes=19,
                      feat_dim=cfg.model.feat_dim, graph_resolution=10)
    loss = trainer.train_one_epoch(tr)
    assert loss == loss and loss > 0           # finite, positive


def test_warmup_schedule():
    cfg = _toy_cfg("spectral")
    model, crit = build_model(cfg), build_criterion(cfg)
    crit.w_rayleigh = 0.05
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = Trainer(model, crit, opt, device="cpu", warmup=(10, 30))
    trainer.global_step = 0;  assert trainer._warmup_weight() == 0.0
    trainer.global_step = 20; assert abs(trainer._warmup_weight() - 0.025) < 1e-6
    trainer.global_step = 40; assert abs(trainer._warmup_weight() - 0.05) < 1e-6


def test_trainer_conv_overfits_toy():
    """Sanity: a few epochs on a tiny toy set should push training loss down."""
    cfg = _toy_cfg("conv"); cfg.data.n_toy = 2; cfg.optim.epochs = 6
    model, crit = build_model(cfg), build_criterion(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    tr, va = _loaders(cfg)
    trainer = Trainer(model, crit, opt, device="cpu", num_classes=19, feat_dim=8, graph_resolution=10)
    first = trainer.train_one_epoch(tr)
    for _ in range(5):
        last = trainer.train_one_epoch(tr)
    assert last < first
