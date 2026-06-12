# cheeger

**Differentiable spectral embedding for dense semantic segmentation — built from scratch.**

`cheeger` learns to segment by *cutting graphs*. It treats a network's feature map
as a graph, builds a learned affinity, and runs a fully differentiable spectral
embedding as a U-Net segmentation head — trained end-to-end and benchmarked against
a conventional 1×1-conv head on a driving dataset (Cityscapes).

> **Why the name.** The Cheeger inequality, `h(G)²/2 ≤ λ₂ ≤ 2·h(G)`, bounds the
> *Cheeger constant* `h(G)` — the conductance of the best cut in a graph — by the
> Fiedler value `λ₂`. Dense segmentation *is* finding good cuts, so the spectrum of
> the graph Laplacian is the natural object to optimise. The core spectral library
> is the **`fiedler`** package (after the Fiedler vector, `λ₂`'s eigenvector).

Every spectral operator — affinity graph, Laplacian, eigensolver, **and the
backward pass** — is implemented from first principles in PyTorch. External
libraries (scipy/numpy) are used **only as correctness oracles in tests**, never in
the forward path.

## What makes it novel

Prior spectral-segmentation work applies a *fixed, hand-chosen* weighting to the
eigenbasis (a temper exponent, a hard truncation at K). `cheeger` replaces that
scalar with a **learned, label-supervised spectral response** `h_θ(λ)`:

```
Φ = U · diag(h_θ(λ))
```

trained through a **degeneracy-robust differentiable eigensolver** (Lorentzian-
broadened eigenvalue-gap gradients, stable where eigenvalues collapse at segment
boundaries), with the noise floor derived from random-matrix theory (the
**Marchenko–Pastur bulk edge**) rather than a hand-set threshold.

## Layout

```
src/fiedler/          the installable spectral library (pure math)
  graph/              affinity kernels (+learned metric) + Laplacian variants
  spectral/           jacobi · diffeig (differentiable) · response (h_θ, MP) · embedding
  models/             U-Net backbone + heads (conv baseline | spectral)   [Phase 3]
  losses/             Rayleigh-quotient spectral-consistency loss          [Phase 2]
  data/  metrics/  engine/  utils/  testing/
tests/                pytest suite (math validated vs numpy, gradcheck)
scripts/              validate_core · stress_test (P0 fuzzer)
demos/  visualizers/  notebooks/  datasets/  extensions/  configs/  results/
```

## Quickstart

```bash
pip install -e ".[dev,data]"   # or: make install
make validate                  # prove the core against numpy oracles
make test                      # full unit suite (incl. gradcheck)
make stress                    # P0 randomized fuzz gate (~27s)
make demo                      # writes results/fiedler_partition.png
```

```python
from fiedler.spectral import SpectralEmbedding
emb = SpectralEmbedding(k=16, knn=16, metric_dim=256)   # learned affinity + h_θ
out = emb(features)            # features (N, C) -> out["phi"] (N, 16), differentiable
```

## Compute

Device-agnostic by construction (`fiedler.utils.get_device`): runs on Apple MPS
locally, a Colab T4, or an AWS/Azure CUDA box with no code changes. Develop locally
on CPU/MPS; move to a GPU only for full Cityscapes training.

## Status

- [x] Graph operators: Gaussian + learned-metric affinity, k-NN sparsify, 3 Laplacian variants
- [x] From-scratch symmetric eigensolver (cyclic Jacobi), validated vs numpy
- [x] P0 rigor: invariant toolkit, hardened edge/numerical tests, parallel stress fuzzer
- [x] **Phase 1** — differentiable eigensolver (broadened gradients, gradcheck-verified),
      learned `h_θ(λ)` + Marchenko–Pastur noise floor, `SpectralEmbedding` module
- [x] **Phase 2** — segmentation metrics (mIoU/per-class/accuracies + Boundary-IoU/BF/trimap,
      sklearn-validated) and losses (CE + class weights + optional OHEM/Lovász;
      Rayleigh spectral-consistency + MP `bulk_penalty` via `CompositeLoss`)
- [x] **Phase 3** — U-Net backbone + ConvHead/SpectralSegHead + Cityscapes loader + toy dataset
- [x] **Phase 4** — typed Config, device-agnostic Trainer, `scripts/train.py` + `scripts/benchmark.py` (smoke-tested end-to-end)
