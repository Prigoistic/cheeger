"""Device + reproducibility helpers.

One source of truth for device selection so every module — and every notebook on
Colab/AWS — picks the right backend automatically: CUDA if present, else Apple
MPS, else CPU. Nothing in the library hard-codes ``cuda``.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Best available device. ``prefer`` (e.g. "cpu") forces a choice if valid."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    """Seed python / numpy / torch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
