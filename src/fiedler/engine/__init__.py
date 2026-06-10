"""Training / evaluation orchestration.

  config.py   typed experiment config (dataclasses <-> configs/*.yaml)
  trainer.py  device-agnostic train/eval loop + builders (model, dataset, loss)
  logging.py  experiment logger (TensorBoard / optional W&B)

The trainer is backend-blind: the same loop runs on CPU/MPS locally or CUDA in the
cloud. Note that the spectral head's eigensolver needs CPU or CUDA (eigh is not
implemented on MPS), so spectral runs go on CPU locally.
"""
from .config import Config, DataConfig, ModelConfig, SpectralConfig, LossConfig, OptimConfig
from .trainer import Trainer, build_model, build_dataset, build_criterion
from .logging import ExperimentLogger

__all__ = [
    "Config", "DataConfig", "ModelConfig", "SpectralConfig", "LossConfig", "OptimConfig",
    "Trainer", "build_model", "build_dataset", "build_criterion", "ExperimentLogger",
]
