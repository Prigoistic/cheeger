"""Evaluation metrics — mIoU / accuracies / boundary quality.

All derived from a streaming confusion matrix; the numbers the spectral head must
beat the conv baseline on. ``sklearn`` appears only as a test oracle, never here.
"""
from .confusion import ConfusionMatrix
from .segmentation import (
    SegMetrics,
    per_class_iou,
    mean_iou,
    pixel_accuracy,
    mean_accuracy,
    frequency_weighted_iou,
    CITYSCAPES_CLASSES,
)
from .boundary import boundary_iou, bf_score, trimap_accuracy, semantic_edges, dilate, erode

__all__ = [
    "ConfusionMatrix",
    "SegMetrics",
    "per_class_iou",
    "mean_iou",
    "pixel_accuracy",
    "mean_accuracy",
    "frequency_weighted_iou",
    "CITYSCAPES_CLASSES",
    "boundary_iou",
    "bf_score",
    "trimap_accuracy",
    "semantic_edges",
    "dilate",
    "erode",
]
