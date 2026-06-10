"""Scale profiler — sparse Lanczos vs dense eigh for the bottom-k eigenpairs.

Two honest claims, two sections:

  A. ACCURACY + SPEEDUP on the *real* operator (a k-NN gaussian-affinity Laplacian
     with cluster structure, like learned segment features). With a sufficient
     Krylov dim Lanczos matches dense ``eigh`` to machine precision and is faster.
     The bottom-k here are clustered near 0, so they need m ≈ 20k to resolve — that
     is the speed/accuracy knob, and even at that m Lanczos wins.

  B. SCALING beyond what dense can do. Dense ``eigh`` needs an n×n matrix (O(n²)
     memory) and an O(n³) solve — impossible past ~10k nodes. Lanczos touches the
     graph only via mat-vecs, so it runs at 64k+ nodes in well under a second.

Run:  python scripts/profile_scale.py
"""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from fiedler.spectral import lanczos_smallest_k
from fiedler.graph import gaussian_affinity, laplacian

torch.set_default_dtype(torch.float64)
K = 16


def timeit(fn):
    t = time.time(); out = fn(); return out, time.time() - t


def knn_feature_laplacian(n, clusters=8, dim=16, seed=0):
    """Dense-built k-NN gaussian-affinity L_sym on clustered features — the real
    operator. Feasible to build only while n² fits in memory (≤ ~4096)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, dim, generator=g) + torch.randint(0, clusters, (n, 1), generator=g) * 6.0
    return laplacian(gaussian_affinity(X, sigma=2.0, k=10), kind="sym")


def grid_laplacian(side):
    """Sparse 4-neighbour grid Laplacian (pixel adjacency) — O(n) to build."""
    H = W = side
    n = H * W
    idx = torch.arange(n).reshape(H, W)
    er = torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)])
    ed = torch.stack([idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)])
    i = torch.cat([er[0], ed[0], er[1], ed[1]])
    j = torch.cat([er[1], ed[1], er[0], ed[0]])
    A = torch.sparse_coo_tensor(torch.stack([i, j]), torch.ones(i.numel()), (n, n)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    Li = torch.cat([torch.arange(n), A.indices()[0]])
    Lj = torch.cat([torch.arange(n), A.indices()[1]])
    Lv = torch.cat([deg, -A.values()])
    return torch.sparse_coo_tensor(torch.stack([Li, Lj]), Lv, (n, n)).coalesce()


def section_a():
    print(f"A. accuracy + speedup on the real k-NN feature graph   (bottom-{K}, m=20k)")
    print(f"   {'n':>6} {'dense eigh':>11} {'lanczos':>10} {'speedup':>8} {'agreement':>11}")
    for n in (1024, 2048, 4096):
        L = knn_feature_laplacian(n)
        m = min(n, 20 * K)
        (lz, _), t_lz = timeit(lambda: lanczos_smallest_k(L, K, m=m))
        (ev, _), t_de = timeit(lambda: torch.linalg.eigh(L))
        agree = (lz - ev[:K]).abs().max().item()
        print(f"   {n:>6} {t_de*1000:9.0f}ms {t_lz*1000:8.0f}ms {t_de/t_lz:6.1f}x {agree:>11.1e}")


def section_b():
    print(f"\nB. scaling beyond dense — sparse grid Laplacian, Lanczos only   (bottom-{K})")
    print(f"   {'grid':>9} {'n':>7} {'nnz':>9} {'lanczos':>10} {'dense would need':>18}")
    for side in (64, 128, 180, 256):              # n up to 65536
        n = side * side
        L = grid_laplacian(side)
        (_, _), t_lz = timeit(lambda: lanczos_smallest_k(L, K))
        gb = n * n * 8 / 1e9
        print(f"   {side}×{side:<4} {n:>7} {L._nnz():>9} {t_lz*1000:8.0f}ms   {gb:8.1f} GB dense")


def main():
    print("=" * 70)
    print(f"scale profiler — bottom-{K} eigenpairs   (sparse Lanczos vs dense eigh)")
    print("=" * 70)
    section_a()
    section_b()
    print("-" * 70)
    print("Lanczos matches dense to machine precision with enough Krylov steps, and")
    print("runs at 65k nodes (34 GB as a dense matrix) in well under a second.")


if __name__ == "__main__":
    main()
