"""Loss functions.

Controlled-minimal recipe: both the conv baseline and the spectral head train with
the same base ``cross_entropy_2d``; the spectral head adds the novel spectral
regularisers (``rayleigh_consistency`` + ``bulk_penalty``) via ``CompositeLoss``.
OHEM / Lovász are built but opt-in (strong-baseline stage only).
"""
from .segmentation import (
    cross_entropy_2d,
    class_weights_median_freq,
    OHEMCrossEntropy,
    lovasz_softmax,
)
from .spectral import rayleigh_consistency, bulk_penalty, CompositeLoss

__all__ = [
    "cross_entropy_2d",
    "class_weights_median_freq",
    "OHEMCrossEntropy",
    "lovasz_softmax",
    "rayleigh_consistency",
    "bulk_penalty",
    "CompositeLoss",
]
