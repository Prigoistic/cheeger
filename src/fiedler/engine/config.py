"""Typed experiment configuration.

Dataclasses mirror configs/*.yaml. ``Config.from_yaml`` loads a file, ``from_dict``
builds from a plain dict (and ignores unknown keys, so a yaml can carry extra notes),
and ``to_yaml`` serialises back. The trainer and scripts are driven entirely by a
``Config``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict


def _keep_known(cls, d: dict) -> dict:
    """Drop keys that are not fields of the dataclass (robust to extra yaml keys)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names}


@dataclass
class DataConfig:
    name: str = "toy"                 # toy | cityscapes
    root: str = "datasets/cityscapes"
    num_classes: int = 19
    crop_size: tuple = (128, 256)
    batch_size: int = 4
    n_toy: int = 64                   # number of synthetic scenes


@dataclass
class SpectralConfig:
    k_eigenvecs: int = 16
    laplacian: str = "sym"
    affinity: str = "learned"         # learned | gaussian
    knn: int = 16
    graph_resolution: int = 32
    solver: str = "dense"             # dense | lanczos
    lanczos_m: int | None = None
    mlp_hidden: int = 64


@dataclass
class ModelConfig:
    backbone: str = "unet"
    base: int = 32
    depth: int = 3
    feat_dim: int = 32                # U-Net output channels = head input channels
    head: str = "conv"               # conv | spectral
    spectral: SpectralConfig = field(default_factory=SpectralConfig)


@dataclass
class LossConfig:
    cross_entropy: float = 1.0
    spectral_consistency: float = 0.1
    bulk: float = 0.0
    rayleigh_warmup_start: int = 0    # global steps; 0,0 means no warmup
    rayleigh_warmup_end: int = 0


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50


@dataclass
class Config:
    experiment: str = "exp"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d or {})
        m = dict(d.get("model", {}))
        spec = SpectralConfig(**_keep_known(SpectralConfig, m.pop("spectral", {})))
        model = ModelConfig(spectral=spec, **_keep_known(ModelConfig, m))
        data = DataConfig(**_keep_known(DataConfig, d.get("data", {})))
        if isinstance(data.crop_size, list):
            data.crop_size = tuple(data.crop_size)
        loss = LossConfig(**_keep_known(LossConfig, d.get("loss", {})))
        optim = OptimConfig(**_keep_known(OptimConfig, d.get("optim", {})))
        top = _keep_known(cls, d)
        top.pop("data", None); top.pop("model", None); top.pop("loss", None); top.pop("optim", None)
        return cls(data=data, model=model, loss=loss, optim=optim, **top)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
