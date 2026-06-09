"""Randomized stress / fuzz harness — P0 rigor gate.

Two complementary sweeps, because the harness has two distinct jobs:

  A. INVARIANT sweep (broad, fast) — verify the algebraic properties of affinity
     and Laplacian operators over a huge config space. Uses batched
     ``torch.linalg.eigvalsh`` (C++) so thousands of graphs cost one call per
     size-bucket. This is where breadth comes from.

  B. SOLVER sweep (focused, parallel) — verify our *from-scratch* ``jacobi_eigh``
     matches the numpy oracle (orthonormality, reconstruction, eigenvalues).
     Only meaningful at the sizes Jacobi targets (n ≤ 40); parallelised across
     CPU cores. This is where solver-correctness confidence comes from.

Splitting them removes the pure-Python O(n³) Jacobi loop from the breadth sweep
(~100× faster) while keeping every guarantee the old single-loop gave.

    python scripts/stress_test.py                 # 5000 invariant + 400 solver
    python scripts/stress_test.py 20000 1000      # more of each
    python scripts/stress_test.py 5000 400 1      # --jobs=1 (disable multiproc)
    make stress
"""
import os
import sys
import time
import pathlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from fiedler.graph import gaussian_affinity, laplacian
from fiedler.spectral import jacobi_eigh
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)
GREEN, RED, YEL, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


# --------------------------------------------------------------------------- #
# config generation (seeded by index -> fully reproducible, order-independent)
# --------------------------------------------------------------------------- #
def make_configs(n_trials, n_lo, n_hi, seed=0):
    rng = np.random.default_rng(seed)
    cfgs = []
    for _ in range(n_trials):
        cfgs.append(dict(
            n=int(rng.integers(n_lo, n_hi)),
            c=int(rng.integers(1, 12)),
            sigma=float(10 ** rng.uniform(-2, 2)),
            k=int(rng.integers(1, 20)),
            kind=str(rng.choice(["comb", "sym"])),
            scale=float(10 ** rng.uniform(-2, 3)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return cfgs


def _build_laplacian(cfg):
    """Build one Laplacian and check the cheap affinity invariants inline."""
    g = torch.Generator().manual_seed(cfg["seed"])
    X = torch.randn(cfg["n"], cfg["c"], generator=g, dtype=torch.float64) * cfg["scale"]
    W = gaussian_affinity(X, sigma=cfg["sigma"], k=cfg["k"])
    ft.assert_finite(W, "W"); ft.assert_symmetric(W, name="W")
    if not ((W >= 0).all() and (W <= 1 + 1e-9).all()):
        raise AssertionError("affinity out of [0,1]")
    return laplacian(W, kind=cfg["kind"])


# --------------------------------------------------------------------------- #
# Sweep A — batched invariant verification
# --------------------------------------------------------------------------- #
def invariant_sweep(cfgs):
    """Bucket by n, batch the eigendecomposition, vectorize invariant checks."""
    buckets: dict[int, list] = {}
    for c in cfgs:
        buckets.setdefault(c["n"], []).append(c)

    worst = {"sym": 0.0, "psd_margin": 0.0, "range": 0.0, "rows": 0.0}
    failures = []
    t0 = time.time()
    for n, group in sorted(buckets.items()):
        Ls, kinds = [], []
        for c in group:
            try:
                Ls.append(_build_laplacian(c)); kinds.append(c["kind"])
            except Exception as e:
                failures.append((c, repr(e)))
        if not Ls:
            continue
        L = torch.stack(Ls)                                  # (B, n, n)
        kinds = np.array(kinds)
        # batched, vectorized invariants
        sym_err = (L - L.transpose(-1, -2)).abs().amax(dim=(-1, -2))      # (B,)
        ev = torch.linalg.eigvalsh(L)                        # (B, n)  one call/bucket
        if not torch.isfinite(L).all() or not torch.isfinite(ev).all():
            failures.append(({"n": n}, "non-finite L or spectrum"))
        worst["sym"] = max(worst["sym"], sym_err.max().item())
        # comb: PSD + rows sum to zero ; sym: spectrum in [0,2]
        is_comb = torch.tensor(kinds == "comb")
        if is_comb.any():
            psd_margin = (-ev[is_comb].amin(dim=-1)).clamp_min(0).max().item()
            rows = L[is_comb].sum(dim=-1).abs().amax().item()
            worst["psd_margin"] = max(worst["psd_margin"], psd_margin)
            worst["rows"] = max(worst["rows"], rows)
            if psd_margin > 1e-6:
                failures.append(({"n": n, "kind": "comb"}, f"not PSD (margin {psd_margin:.1e})"))
            if rows > 1e-8:
                failures.append(({"n": n, "kind": "comb"}, f"rows sum {rows:.1e}"))
        is_sym = ~is_comb
        if is_sym.any():
            over = (ev[is_sym].amax() - 2.0).clamp_min(0).item()
            under = (-ev[is_sym].amin()).clamp_min(0).item()
            rng_err = max(over, under)
            worst["range"] = max(worst["range"], rng_err)
            if rng_err > 1e-6:
                failures.append(({"n": n, "kind": "sym"}, f"spectrum out of [0,2] by {rng_err:.1e}"))
    return worst, failures, time.time() - t0


# --------------------------------------------------------------------------- #
# Sweep B — parallel solver correctness (jacobi vs numpy)
# --------------------------------------------------------------------------- #
def _solver_worker(cfg):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    try:
        # mix random-symmetric and real-Laplacian inputs
        if cfg["seed"] % 2 == 0:
            A = ft.random_symmetric(cfg["n"], seed=cfg["seed"])
        else:
            A = _build_laplacian(cfg)
        evals, evecs = jacobi_eigh(A)
        if not (torch.isfinite(evals).all() and torch.isfinite(evecs).all()):
            return ("FAIL", cfg, "non-finite output")
        I = torch.eye(cfg["n"], dtype=torch.float64)
        ortho = (evecs.t() @ evecs - I).abs().max().item()
        recon = ((evecs * evals.unsqueeze(0)) @ evecs.t() - A).abs().max().item()
        lam = float(np.abs(evals.numpy() - np.linalg.eigvalsh(A.numpy())).max())
        if ortho > 1e-7 or recon > 1e-6 or lam > 1e-7:
            return ("FAIL", cfg, f"ortho={ortho:.1e} recon={recon:.1e} lam={lam:.1e}")
        return ("OK", ortho, recon, lam)
    except Exception as e:  # pragma: no cover
        return ("FAIL", cfg, repr(e))


def solver_sweep(cfgs, jobs):
    worst = {"ortho": 0.0, "recon": 0.0, "lam": 0.0}
    failures = []
    t0 = time.time()
    if jobs == 1:
        results = map(_solver_worker, cfgs)
    else:
        ex = ProcessPoolExecutor(max_workers=jobs)
        results = ex.map(_solver_worker, cfgs, chunksize=8)
    for r in results:
        if r[0] == "FAIL":
            failures.append((r[1], r[2]))
        else:
            _, o, rc, l = r
            worst["ortho"] = max(worst["ortho"], o)
            worst["recon"] = max(worst["recon"], rc)
            worst["lam"] = max(worst["lam"], l)
    if jobs != 1:
        ex.shutdown()
    return worst, failures, time.time() - t0


# --------------------------------------------------------------------------- #
def main(inv_trials, solver_trials, jobs):
    print(f"{'='*66}\nfiedler stress harness   "
          f"(invariant={inv_trials}, solver={solver_trials}, jobs={jobs})\n{'='*66}")

    inv_cfgs = make_configs(inv_trials, n_lo=2, n_hi=80, seed=0)
    sol_cfgs = make_configs(solver_trials, n_lo=2, n_hi=40, seed=1)

    wi, fi, ti = invariant_sweep(inv_cfgs)
    print(f"A. invariant sweep  {inv_trials} graphs in {ti:5.1f}s  "
          f"({inv_trials/ti:,.0f}/s)")
    print(f"   worst: sym={wi['sym']:.1e} psd_margin={wi['psd_margin']:.1e} "
          f"range={wi['range']:.1e} rows={wi['rows']:.1e}  fails={len(fi)}")

    ws, fs, ts = solver_sweep(sol_cfgs, jobs)
    print(f"B. solver sweep     {solver_trials} jacobi vs numpy in {ts:5.1f}s  "
          f"({solver_trials/ts:,.0f}/s)")
    print(f"   worst: ortho={ws['ortho']:.1e} recon={ws['recon']:.1e} "
          f"lam_err={ws['lam']:.1e}  fails={len(fs)}")

    print("=" * 66)
    fails = fi + fs
    for cfg, msg in fails[:10]:
        print(f"{RED}FAIL{RESET} {cfg} -> {msg}")
    if not fails:
        print(f"{GREEN}STRESS PASSED — {inv_trials+solver_trials} trials, "
              f"0 invariant violations, {ti+ts:.1f}s total{RESET}")
        return 0
    print(f"{RED}STRESS FAILED — {len(fails)} violations{RESET}")
    return 1


if __name__ == "__main__":
    inv = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    sol = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    jb = int(sys.argv[3]) if len(sys.argv) > 3 else min(8, (os.cpu_count() or 2))
    sys.exit(main(inv, sol, jb))
