"""Joint image/label transforms.

Each transform takes ``(image, label)`` and returns ``(image, label)``, keeping the
two aligned. ``image`` is a float CHW tensor, ``label`` an integer HW tensor. Label
interpolation is always nearest neighbour so class ids stay intact; padding uses the
ignore index so introduced pixels are excluded from the loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Tensor, label: Tensor):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, label):
        if torch.rand(()) < self.p:
            image = torch.flip(image, dims=[-1])
            label = torch.flip(label, dims=[-1])
        return image, label


class RandomCrop:
    def __init__(self, size, ignore_index: int = 255):
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.ignore_index = ignore_index

    def __call__(self, image, label):
        _, H, W = image.shape
        # pad if the image is smaller than the crop (image with 0, label with ignore)
        ph, pw = max(self.h - H, 0), max(self.w - W, 0)
        if ph or pw:
            image = F.pad(image, [0, pw, 0, ph], value=0.0)
            label = F.pad(label, [0, pw, 0, ph], value=self.ignore_index)
            _, H, W = image.shape
        top = int(torch.randint(0, H - self.h + 1, ()))
        left = int(torch.randint(0, W - self.w + 1, ()))
        image = image[:, top:top + self.h, left:left + self.w]
        label = label[top:top + self.h, left:left + self.w]
        return image, label


class Resize:
    """Resize to a fixed (H, W): bilinear for the image, nearest for the label."""

    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, label):
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode="bilinear",
                              align_corners=False).squeeze(0)
        label = F.interpolate(label[None, None].float(), size=self.size,
                              mode="nearest").squeeze(0).squeeze(0).long()
        return image, label


class Normalize:
    """Per-channel normalisation of the image only (ImageNet stats by default)."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image, label):
        return (image - self.mean.to(image)) / self.std.to(image), label
