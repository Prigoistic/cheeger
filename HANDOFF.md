# fiedler — Project Handoff & Research Notes

> A from-scratch, differentiable **spectral embedding module** for dense semantic
> segmentation. Every spectral operator (affinity graph, Laplacian, eigensolver,
> backward pass) is implemented from first principles in PyTorch and plugged into
> a U-Net as a learnable segmentation head, benchmarked against a 1×1-conv head on
> Cityscapes. scipy/numpy appear **only as correctness oracles in tests**, never in
> the forward path.

This file is the single pickup point for a new session. It captures (1) the
research thesis, (2) what is actually built vs stubbed, and (3) the ordered
roadmap to completion. Source of truth for code is the repo; source of truth for
*direction* is this file.

---

## 0. Context: where this came from

The project name was changed from `specseg` → **`fiedler`** (after the Fiedler
vector, the 2nd Laplacian eigenvector used for graph partitioning). The rename is
complete across `pyproject.toml`, `src/fiedler/`, tests, scripts, demos, README.

Four papers were analyzed (full critiques in
`.noteweave/sessions/ext_1781011818335/papers/*/analysis.md`):

| Paper | arXiv | Core idea | Key flaw (per analysis) |
|---|---|---|---|
| **Spectral Tempering (SpecTemp)** | 2603.19339 (Li 2026) | power-scale eigenvalues by an adaptive γ(k) governed by an SNR-knee | SNR is admittedly "not a generative estimate, only a proxy"; linear SNR→γ map "chosen by parsimony"; noise floor = last 10% of eigenvalues (arbitrary); validated on **one** model/dataset |
| **Deep Spectral Improvement** | 2402.02474 (Arefi 2024) | channel pruning (NCR entropy + DCR std) before affinity; Bray-Curtis-over-Chebyshev affinity | DCR *hurts* when stacked on NCR (32.92→32.70); affinity justified only by a toy example; eval on filtered easy subset |
| **CLASP** | 2509.25016 (Curie 2025) | training-free per-image spectral clustering of DINOv2 features; eigengap+silhouette picks K | uses eigvecs of affinity **A** and of Laplacian **L** interchangeably (not equivalent); no cluster→label mapping protocol |
| **Spectral U-Net** | 2409.09216 (Peng 2024) | DTCWT/iDTCWT wavelet down/up-sampling in U-Net to preserve high-freq boundaries | "lossless/invertible" claim is false once a learned conv mixes coefficients; pseudo-code non-executable |

An experiment plan ("ASE-UNet") already exists at
`.noteweave/sessions/ext_1781011818335/instructions.md` — it is the *engineering*
synthesis (prune + Bray-Curtis + temper + wavelet decoder). The thesis below is
the *theoretical* core that goes beyond that synthesis.

---

## 1. The research thesis (what makes this novel)

**Through-line across all four papers:** each manipulates a *spectrum* and then
makes an **ad-hoc, fixed, unsupervised decision about how to weight or truncate
it**. They are all special cases of one operator — a weighting function `h(λ)`
applied to the eigenbasis of a graph operator on per-image features:

- SpecTemp tempering = `h(λ) = λ^{γ(k)/2}`
- CLASP truncation at K = `h(λ) = 𝟙[λ < λ_K]`
- Deep Spectral channel pruning = reshapes *which* λ exist
- Spectral U-Net wavelet bands = the spatial-domain dual of the same filter

**None of them makes the spectral weighting both learnable AND supervised by the
dense labels.** SpecTemp's whole selling point (learning-free γ) is also its fatal
flaw.

### Thesis statement

> Replace SpecTemp's scalar, hand-chosen γ(k) with a **differentiable spectral
> response `h_θ(λ)`** — a learned filter on the bottom eigenpairs of the per-image
> symmetric normalized Laplacian `L_sym = I − D^{-1/2} W D^{-1/2}` — trained
> **end-to-end against dense (Cityscapes) labels** through a differentiable
> eigensolver. The embedding becomes `Φ = U · diag(h_θ(λ))`. The Rayleigh-quotient
> spectral-consistency loss supplies the supervisory bridge.

Why this is a real theory, not a re-skin:

1. **Generalizes a scalar to a function.** `h_θ(λ)` is a graph-signal-processing
   spectral filter (cf. ChebNet) but applied to a *content-adaptive, per-image*
   Laplacian for dense prediction — an object that exists in none of the 4 papers.
2. **Falsifiable test of SpecTemp's central claim.** Train `h_θ` freely, then ask:
   *does the learned filter recover the SNR-knee shape?* Yes → you supply the
   justification SpecTemp lacked. No → you falsify "optimal tempering is
   SNR-governed." Either is publishable. It is exactly the oracle-across-settings
   experiment the SpecTemp analysis flags as missing.
3. **Collapses 4 heuristics into 1 trained operator.** Affinity choice (paper 2)
   defines L; channel pruning (paper 2) defines the node features; `h_θ(λ)`
   replaces both γ(k) (paper 1) and eigengap-K truncation (paper 3); wavelet band
   weighting (paper 4) is its spatial dual.

### The rigor that makes it theory, not another heuristic

SpecTemp's weakest point — "noise floor = mean of last 10% of eigenvalues" — has a
principled replacement via **random matrix theory**. The **Marchenko–Pastur (MP)
bulk edge** gives the exact noise-eigenvalue cutoff for a random feature matrix;
the **BBP transition** says signal eigenvalues are those above the MP edge. So:

> Derive the noise floor as the **MP bulk edge** of the affinity/feature spectrum.
> Regularize `h_θ(λ) → 0` inside the bulk, leave it free above. This turns the
> "monotonic proxy" into an actual generative estimate — fixing the exact sentence
> the analysis pinned as SpecTemp's weakest claim.

### Two supporting threads the papers hand you for free

- **CLASP's A-vs-L contradiction is a real theorem.** Eigenvectors of `A` and of
  `L_sym` coincide *only* on a degree-regular graph; for a kNN graph they diverge,
  and the divergence is exactly the degree normalization. A clean statement of
  *when segmentation signal lives in A's top eigenvectors vs L's bottom ones*
  justifies the repo's existing commitment to `L_sym`.
- **DCR-hurts-NCR (32.92→32.70) is predictable.** Std-pruning likely deletes
  low-variance channels carrying the *smooth, low-frequency* component — precisely
  the small-λ eigenvectors that define segments. Reframe: **channel selection
  should happen in the spectral domain of the resulting Laplacian, not on raw
  channel statistics.**

---

## 2. Build status (audited)

Math core is real, from-scratch, and verified (13 pytest tests + `validate_core.py`
all green). The entire learning stack is stubbed.

### Built & verified ✅

| Module | Status | Contents |
|---|---|---|
| `src/fiedler/graph/laplacian.py` | ✅ | `pairwise_sq_dists` (**learnable metric** — diag or full M, "Novelty #1" hook), `gaussian_affinity` (dense, kNN-sparsifiable, learnable σ), `knn_sparsify` (top-k + OR-symmetrize), `degree`, `laplacian` (comb / sym / rw) |
| `src/fiedler/spectral/jacobi.py` | ✅ | Cyclic Jacobi symmetric eigensolver, float64, ascending order; matches numpy to ~1e-13 |
| `src/fiedler/utils/device.py` | ✅ | MPS/CUDA/CPU selection |
| `src/fiedler/engine/logging.py` | ✅ | experiment logger |
| `tests/`, `scripts/validate_core.py` | ✅ pass | Jacobi vs numpy; Laplacian identities (rows→0, #components = #zero-eigvals, L_sym∈[0,2]); Fiedler recovers planted 2-cluster split |
| `visualizers/`, `demos/` | ✅ | presentation only; produce PNGs in `results/` |

### Stubbed (docstring-only `__init__.py`, zero implementation) ⬜

- ⬜ `losses/` — Rayleigh-quotient spectral-consistency loss ("Novelty #5")
- ⬜ `metrics/` — mIoU, per-class IoU, pixel-acc, boundary-F → **no way to score yet**
- ⬜ `models/` — UNet backbone, ConvHead baseline, SpectralSegHead
- ⬜ `data/` — toy graphs + Cityscapes adapter
- ⬜ `engine/` — `config.py` + `trainer.py` (only `logging.py` exists; `configs/default.yaml` written but no loader)

---

## 3. Roadmap to completion

### ⚠️ The blocking risk, before anything else

The Jacobi solver is **dense `O(n³)` with a Python double-loop** — correct but only
usable for n ≤ ~60. The config builds the graph at 64×64 = **4096 nodes**. Jacobi
will never run at that size. The whole project hinges on a **differentiable top-k
eigensolver that does not exist yet.** This is both the central technical risk and
where the genuine difficulty/novelty lives.

### Critical path

```
PHASE 1 — make spectral differentiation real (the gate)
  [ ] spectral/lanczos.py     top-k eigenpairs for large sparse L
                              (or torch.linalg.eigh wrapper + custom backward via
                              the eigenvector-gradient formula dV = Σ_{j≠i} (v_jᵀ dA v_i)/(λ_i−λ_j) v_j)
  [ ] spectral/embedding.py   SpectralEmbedding nn.Module: features → graph → L_sym
                              → top-k eigvecs → Φ = U·diag(h_θ(λ)); differentiable end-to-end
  [ ] spectral/response.py    h_θ(λ) learned spectral filter  +  MP-edge noise-floor estimator   ← THE THEORY
  [ ] test: torch.autograd.gradcheck the backward on small graphs;
            sanity: free-trained h_θ vs SpecTemp's SNR-knee shape

PHASE 2 — losses & metrics (so you can score anything)
  [ ] losses/spectral_loss.py   Rayleigh-quotient regulariser  vᵀLv / vᵀv  (Novelty #5)
  [ ] losses wiring             cross-entropy + λ·spectral_consistency
  [ ] metrics/segmentation.py   mIoU, per-class IoU, pixel-acc, boundary-F (2px)

PHASE 3 — model & data
  [ ] models/unet.py + heads.py  UNet backbone; ConvHead (baseline) vs SpectralSegHead (swappable)
  [ ] data/toy.py                planted-partition graphs for fast iteration
  [ ] data/cityscapes.py         loader + 34→19 label remap

PHASE 4 — orchestration
  [ ] engine/config.py           dataclass ↔ configs/*.yaml
  [ ] engine/trainer.py          device-agnostic train/eval loop + checkpointing
  [ ] scripts/train.py, scripts/benchmark.py

PHASE 5 — the experiment
  [ ] toy first: spectral head beats conv head on synthetic structured data
  [ ] Cityscapes: mIoU(spectral) vs mIoU(conv), boundary-F, 3 seeds (42,1337,2024), paired t-test
  [ ] theory result: does learned h_θ(λ) recover the MP/SNR knee?
```

The single highest-leverage next move is **Phase 1's eigensolver +
`spectral/embedding.py`** — everything downstream is blocked on it, and it is where
`h_θ(λ)` (the theoretical contribution) plugs in.

---

## 4. Conventions & gotchas for the next session

- **Python is `python3`** on this machine (`python` is not on PATH).
- Package installs editable: `pip install -e ".[dev,data]"` or `make install`.
- Quick checks: `make validate` (numpy oracles), `make test` (pytest), `make demo`.
- Tests run in float64 (`torch.set_default_dtype(torch.float64)`); the production
  forward path should run in float32 but the eigensolver may need float64 for
  gradient stability — verify with gradcheck.
- **No scipy/numpy in the forward path** — that is the project's defining
  constraint. They are oracles in `tests/` only.
- `L_sym` (spectrum in [0,2], symmetric PSD) is the chosen operator — the right
  object for a stable symmetric solver. `L_rw` is not symmetric; avoid in the solver.
- "Novelty #1" = learnable metric in the affinity (already hooked in
  `pairwise_sq_dists`); "Novelty #5" = Rayleigh-quotient loss (not yet written).
  The full novelty list is not centrally documented — infer from code comments and
  `configs/default.yaml` (`affinity: learned`, `k_eigenvecs: 16`, `laplacian: sym`).
- The differentiable backward through the eigensolver is the make-or-break detail:
  eigenvector gradients blow up when eigenvalues are near-degenerate (small
  eigengap). Plan for spectral-gap regularization or a tolerance floor in the
  `1/(λ_i − λ_j)` term.
