"""Cityscapes loader with the official 34 -> 19 trainId remap.

Cityscapes annotations store 34 raw label ids; the benchmark evaluates 19 of them
and ignores the rest. This module owns that mapping and a thin Dataset that pairs
``leftImg8bit`` images with ``gtFine`` label maps. Image data lives under
``datasets/`` (gitignored); see datasets/README.md for the download.
"""
from __future__ import annotations

import pathlib

import torch
from torch import Tensor
from torch.utils.data import Dataset

# official id -> trainId for the 19 evaluation classes; everything else is ignore
CITYSCAPES_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}


def build_trainid_lut(ignore_index: int = 255) -> Tensor:
    """A (256,) lookup table: labelId -> trainId, ignore for unmapped ids."""
    lut = torch.full((256,), ignore_index, dtype=torch.long)
    for label_id, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        lut[label_id] = train_id
    return lut


_LUT = build_trainid_lut()


def remap_to_trainids(label_ids: Tensor, ignore_index: int = 255) -> Tensor:
    """Map raw labelId tensor -> trainId tensor via the lookup table."""
    lut = _LUT if ignore_index == 255 else build_trainid_lut(ignore_index)
    return lut.to(label_ids.device)[label_ids.long().clamp(0, 255)]


class CityscapesDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None,
                 ignore_index: int = 255):
        self.root = pathlib.Path(root)
        self.split = split
        self.transform = transform
        self.ignore_index = ignore_index

        img_dir = self.root / "leftImg8bit" / split
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Cityscapes images not found at {img_dir}. "
                "Download leftImg8bit + gtFine and unzip under datasets/cityscapes/ "
                "(see datasets/README.md)."
            )
        self.pairs = self._discover(img_dir)
        if not self.pairs:
            raise FileNotFoundError(f"no image/label pairs found under {img_dir}")

    def _discover(self, img_dir: pathlib.Path):
        pairs = []
        for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
            rel = img_path.relative_to(self.root / "leftImg8bit")
            label_path = (self.root / "gtFine" / rel).with_name(
                img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            )
            if label_path.exists():
                pairs.append((img_path, label_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        import numpy as np
        from PIL import Image  # lazy: only needed when actually loading data

        img_path, label_path = self.pairs[i]
        img = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        label_ids = torch.from_numpy(np.array(Image.open(label_path)))
        label = remap_to_trainids(label_ids, self.ignore_index)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
