"""SpectralEmbedding + SpectralResponse — end-to-end differentiability & behaviour."""
import torch

from fiedler.spectral import (
    SpectralEmbedding, SpectralResponse, mp_upper_edge, mp_bulk_mask, bulk_penalty,
)
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)


def test_response_nonnegative_and_differentiable():
    for kind in ("mlp", "cheb"):
        r = SpectralResponse(kind=kind)
        lam = torch.linspace(0, 2, 16, requires_grad=True)
        h = r(lam)
        assert h.shape == lam.shape
        assert (h >= 0).all(), "response must be non-negative"
        h.sum().backward()
        ft.assert_finite(lam.grad, f"{kind} response grad")


def test_mp_edge_monotonic_in_aspect_ratio():
    # more features per sample -> higher noise edge
    assert mp_upper_edge(100, 5) < mp_upper_edge(100, 50)
    assert mp_upper_edge(1000, 10) < mp_upper_edge(100, 10)


def test_bulk_penalty_runs_and_is_finite():
    eig = torch.linspace(0, 3, 20)
    h = torch.ones_like(eig)
    p = bulk_penalty(h, eig, n=50, d=10)
    ft.assert_finite(p, "bulk_penalty")
    assert p.item() >= 0


def test_embedding_forward_shapes():
    X = torch.randn(40, 6)
    emb = SpectralEmbedding(k=8, knn=10)
    out = emb(X)
    assert out["phi"].shape == (40, 8)
    assert out["eigvals"].shape == (8,)
    assert out["eigvecs"].shape == (40, 8)
    for key, v in out.items():
        ft.assert_finite(v, key)


def test_embedding_gradients_flow_to_all_params():
    torch.manual_seed(0)
    X = torch.randn(30, 5, requires_grad=True)
    emb = SpectralEmbedding(k=6, knn=8, metric_dim=5, learn_sigma=True)
    out = emb(X)
    loss = out["phi"].pow(2).sum()
    loss.backward()

    # gradient reaches the input features
    ft.assert_finite(X.grad, "X.grad")
    assert X.grad.abs().sum() > 0
    # ... the affinity bandwidth
    assert emb.log_sigma.grad is not None and torch.isfinite(emb.log_sigma.grad).all()
    # ... the learned metric (Novelty #1)
    assert emb.metric.grad is not None and emb.metric.grad.abs().sum() > 0
    # ... the spectral response h_θ (the theory)
    resp_grads = [p.grad for p in emb.response.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in resp_grads)


def _kmeans2(X, iters=50):
    """Deterministic 2-means (farthest-point init) — enough to score separability."""
    c0 = X[0]
    c1 = X[(X - c0).pow(2).sum(1).argmax()]
    c = torch.stack([c0, c1])
    assign = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(X, c).argmin(dim=1)
        new = torch.stack([X[assign == j].mean(0) if (assign == j).any() else c[j]
                           for j in range(2)])
        if torch.allclose(new, c):
            break
        c = new
    return assign


def test_lanczos_solver_matches_dense_embedding():
    """On a connected graph (distinct bottom eigenvalues — the realistic head case)
    the lanczos solver path matches dense eigh on the bottom-k eigenvalues."""
    torch.manual_seed(0)
    X = torch.randn(120, 8)                                   # connected k-NN graph
    common = dict(k=6, knn=12, learn_sigma=False, sigma_init=1.5)
    dense = SpectralEmbedding(**common, solver="dense")
    lanc = SpectralEmbedding(**common, solver="lanczos", lanczos_m=90)
    lanc.load_state_dict(dense.state_dict())                  # only the solver differs
    od, ol = dense(X), lanc(X)
    assert torch.allclose(od["eigvals"], ol["eigvals"], atol=1e-5), \
        (od["eigvals"] - ol["eigvals"]).abs().max()


def test_lanczos_solver_is_differentiable():
    torch.manual_seed(0)
    X = torch.randn(60, 5, requires_grad=True)
    emb = SpectralEmbedding(k=4, knn=8, metric_dim=5, solver="lanczos", lanczos_m=40)
    out = emb(X)
    out["phi"].pow(2).sum().backward()
    ft.assert_finite(X.grad, "X.grad (lanczos)")
    assert X.grad.abs().sum() > 0
    assert emb.metric.grad is not None and emb.metric.grad.abs().sum() > 0


def test_embedding_separates_planted_clusters():
    """The learned embedding places two well-separated blobs into two clusters in
    spectral-coordinate space (k-means on the embedding rows — the correct use of a
    spectral embedding, robust to null-space rotation when components disconnect)."""
    feats, labels = ft.planted_partition_features([30, 30], sep=8.0, dim=4, seed=0)
    # K eigenvectors for K=2 clusters; row-normalised (Ng-Jordan-Weiss)
    emb = SpectralEmbedding(k=2, knn=10, learn_sigma=False, sigma_init=2.0)
    out = emb(feats)
    rows = torch.nn.functional.normalize(out["phi"].detach(), dim=1, eps=1e-8)
    pred = _kmeans2(rows)
    acc = max((pred == labels).float().mean().item(),
              ((1 - pred) == labels).float().mean().item())
    assert acc > 0.95, f"planted clusters not separated (acc={acc})"
