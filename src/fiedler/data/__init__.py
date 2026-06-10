"""Dataset adapters.

  toy.py         synthetic driving scenes (Cityscapes trainIds) for fast iteration
  cityscapes.py  Cityscapes loader + the official 34 -> 19 trainId remap
  transforms.py  joint image/label transforms (crop, flip, resize, normalize)
"""
from .toy import ToyDrivingDataset, make_toy_scene
from .cityscapes import (
    CityscapesDataset,
    CITYSCAPES_ID_TO_TRAINID,
    build_trainid_lut,
    remap_to_trainids,
)
from . import transforms

__all__ = [
    "ToyDrivingDataset",
    "make_toy_scene",
    "CityscapesDataset",
    "CITYSCAPES_ID_TO_TRAINID",
    "build_trainid_lut",
    "remap_to_trainids",
    "transforms",
]
