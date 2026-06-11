# KONA by Logical Intelligence — Research Report

> Reconstructed from deep-research workflow (session 771a4f85, June 2026).
> 199 adversarial-verification agents ran against public sources. Findings below
> reflect only claims that survived 2-of-3 voter consensus. Refuted claims are
> listed explicitly.

---

## What KONA Is

**Kona** is an **Energy-Based Reasoning Model (EBRM)** built by Logical Intelligence,
launched as **Kona 1.0** with pilot programs beginning Q1 2026. It is explicitly
positioned not as a chatbot, LLM, or assistant — but as **AI infrastructure** designed
to sit beneath existing AI stacks for high-stakes, constraint-heavy tasks.

The core idea: instead of predicting the next token sequentially, Kona maps problems
into an energy landscape and uses optimization — drawing on physics-inspired equations
— to find low-energy solutions that satisfy all constraints simultaneously.

> "Kona does the thinking, the LLM does the talking."

---

## Technical Architecture

| Property | Kona (EBRM) | Standard LLMs |
|---|---|---|
| Generation method | Non-autoregressive, simultaneous | Token-by-token, sequential |
| Representation | Continuous latent space | Discrete tokens |
| Scoring | Learned energy function over full trace | Per-token probability |
| Constraint handling | Hard constraints via energy landscape | Probabilistic, can violate |
| Error correction | Gradient-guided revision in latent space | Must restart from scratch |
| Scale | ~200M parameters, single cheap GPU, <20W | Billions of params, data centers |

### The EBM Architecture — What Is Actually Confirmed

The research surfaced an important ambiguity in Logical Intelligence's marketing. "EBM"
conflates two distinct technologies:

- **Explainable Boosting Machines** (Microsoft Research / InterpretML) — a
  GAM-based interpretable ML framework. This is the **confirmed foundation** of
  parts of the system.
- **Energy-Based Models** (LeCun-tradition physics-inspired optimization) — used in
  their marketing narrative but not independently verifiable from public sources.

**The confirmed InterpretML/EBM architecture:**

**Training:**
- Generalized Additive Model: `prediction = g(β₀ + f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ))`
- Trains one feature at a time in round-robin cycles using gradient boosting + shallow
  trees (depth 2–4)
- Very low learning rate per cycle; round-robin cycling mitigates feature collinearity

**Inference / Interpretability:**
- Each `fⱼ` becomes a lookup table (score vs. feature value)
- Prediction = sum of per-feature table lookups through link function `g`
- Interpretability is intrinsic — not post-hoc like SHAP/LIME

---

## ALEPH (Verified 3-of-3)

Aleph is Logical Intelligence's formal software verification and automated code
generation agent. It produces **machine-checkable proofs** that critical logic behaves
correctly across every execution path — not just testing, actual formal proof.

**Self-reported benchmarks (not independently peer-reviewed):**
- PutnamBench: 99.4%
- VeriSoftBench: 94%

Currently in limited pilot as of early 2026.

---

## Benchmark Result: Sudoku

Logical Intelligence's primary public benchmark is Sudoku puzzle solving:

| System | Solve rate | Avg time | Cost (13k puzzles) |
|---|---|---|---|
| Kona | 96.2% | 313ms | ~$4 |
| Leading LLMs combined | ~2% | up to 90s | ~$11,000 |

Live demo at `sudoku.logicalintelligence.com`.

**Caveat:** The exact 96.2% figure was a 1-of-3 kill in adversarial verification (one
voter refuted it; two confirmed). Treat with measured skepticism until independent
replication.

---

## The Company

**Founded by:** Eve Bodnia (CEO) — PhD in quantum information from UC Santa Barbara,
22 published papers on dark matter.

**Technical Research Board:**
- **Yann LeCun** — Founding Chair (2018 Turing Award, former Meta chief AI scientist)
- **Michael Freedman** — Chief of Mathematics (Fields Medalist)
- **Boris Hanin** — Princeton ORFE Associate Professor, founding advisor
- **Patrick Hillmann** — listed in some sources as leadership (unverified role)

**Target industries:** Energy grids, advanced manufacturing, semiconductors — physical
asset control where probabilistic failure is unacceptable.

---

## Claims That Were Refuted

The adversarial verification killed or weakened most specific marketing claims. Do not
repeat these as facts:

| Claim | Verdict |
|---|---|
| "First credible signs of AGI" (Eve Bodnia) | Marketing, no peer-reviewed backing |
| 96.2% Sudoku solve rate exactly | 1-of-3 killed — treat as approximate |
| Lagrangian physics equations in the core | Not independently verifiable |
| Global energy function scoring end-to-end | Not independently verifiable |
| Non-autoregressive trace generation (LI-specific meaning) | Not independently verifiable |
| 200M parameters, <20W | Not independently verifiable |
| Continuous latent space / dense vector tokens | Not independently verifiable |

**Bottom line:** The formally verified code generation (Aleph) and the GAM/EBM
interpretable ML foundation are solid and well-sourced. The "physics-inspired energy
landscape" framing is marketing narrative without independent peer-reviewed backing as
of February 2026.

---

## Relevance to Adjacent Work

Logical Intelligence's approach is worth watching in the context of **constraint
satisfaction at the reasoning layer** — particularly for domains where hard constraints
matter (physical systems, formal verification, combinatorial problems). The EBM
framing connects to:

- LeCun's JEPA world model architecture (prediction in latent space, not pixel space)
- The broader shift away from autoregressive generation for planning/reasoning tasks
- Formal methods integration with learned systems (cf. neurosymbolic AI)

Whether it scales to open-ended reasoning tasks beyond constraint satisfaction is the
open question.

---

*Sources: Business Wire press release (Jan 2026), Yahoo Finance, founderbrew.com,
Neuron Daily, logicalintelligence.com public materials. Research conducted via
adversarial multi-agent verification workflow, June 2026.*
