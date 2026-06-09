"""fiedler — a from-scratch differentiable spectral embedding library for dense
semantic segmentation.

All spectral mathematics (affinity graphs, Laplacians, eigensolvers, the
differentiable backward pass) is implemented from first principles in torch.
External libraries (scipy/numpy) appear only as correctness oracles in tests/.

Package map
-----------
    graph/      affinity kernels + Laplacian operators
    spectral/   eigensolvers + the differentiable spectral embedding layer
    models/     U-Net backbone + segmentation heads (conv baseline, spectral)
    losses/     spectral-consistency regulariser + segmentation losses
    data/       toy graphs + Cityscapes adapters
    metrics/    mIoU / boundary-F segmentation metrics
    engine/     config + device-agnostic train/eval loop
    utils/      device selection, seeding
"""
from . import graph, spectral, utils

__all__ = ["graph", "spectral", "utils"]
__version__ = "0.0.1"
