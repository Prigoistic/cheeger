"""Network architectures.

  unet.py   — from-scratch U-Net backbone (pluggable encoder; pretrained later).
  heads.py  — ConvHead (1×1 conv baseline) | SpectralSegHead (our spectral head),
              composed with the backbone by SegModel. The head is the swapped
              component in the controlled conv-vs-spectral comparison.
"""
from .unet import UNet, DoubleConv, Down, Up
from .heads import ConvHead, SpectralSegHead, SegModel

__all__ = [
    "UNet", "DoubleConv", "Down", "Up",
    "ConvHead", "SpectralSegHead", "SegModel",
]
