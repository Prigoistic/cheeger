# fiedler

A from-scratch, differentiable **spectral embedding module** for dense semantic
segmentation. Every spectral operator — affinity graph, Laplacian, eigensolver,
and the backward pass — is implemented from first principles in PyTorch. The
module plugs into a U-Net as a learnable segmentation head and is benchmarked
against the conventional 1×1-conv head on a driving dataset (Cityscapes).

External libraries (scipy/numpy) are used **only as correctness oracles in
tests** — never in the forward path.

## Layout

```
src/fiedler/          the installable library (pure spectral math)
  graph/              affinity kernels + Laplacian variants
  spectral/           eigensolvers + differentiable embedding layer
  models/             U-Net backbone + heads (conv baseline | spectral)
  losses/             Rayleigh-quotient spectral-consistency loss
  data/               toy graphs + Cityscapes adapters
  metrics/            mIoU / boundary-F
  engine/             config + device-agnostic train/eval loop
  utils/              device selection, seeding

tests/                pytest suite (math validated vs numpy)
scripts/              CLI runners (validate, train, benchmark)
demos/                runnable illustrations (e.g. Fiedler partition)
visualizers/          plotting tools (spectrum, embedding scatter)
notebooks/            thin Colab/Jupyter training drivers
datasets/             data root (gitignored payloads)
extensions/           C++/CUDA hot-path kernels (added after math is proven)
configs/              experiment configs (yaml)
results/              checkpoints, logs, figures (gitignored)
```

## Quickstart

```bash
pip install -e ".[dev,data]"   # or: make install
make validate                  # prove the core against numpy oracles
make test                      # unit tests
make demo                      # writes results/fiedler_partition.png
```

## Compute

Device-agnostic by construction (`fiedler.utils.get_device`): runs on Apple MPS
locally, a Colab T4, or an AWS/Azure CUDA box with no code changes. Develop
locally on CPU/MPS; move to a GPU only for full Cityscapes training.

## Status

- [x] Graph operators: Gaussian + learned-metric affinity, k-NN sparsify, 3 Laplacian variants
- [x] From-scratch symmetric eigensolver (cyclic Jacobi), validated vs numpy
- [ ] Differentiable spectral embedding layer (implicit-diff backward)
- [ ] Sparse Lanczos top-k solver
- [ ] U-Net integration + Cityscapes training + benchmarks
```
