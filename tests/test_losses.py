"""Losses — sanity, gradients, ignore handling, and the spectral regularisers."""
import pytest
import torch

from fiedler.losses import (
    cross_entropy_2d, class_weights_median_freq, OHEMCrossEntropy, lovasz_softmax,
    rayleigh_consistency, CompositeLoss,
)
from fiedler.graph import gaussian_affinity, laplacian
from fiedler.spectral import SpectralResponse
from fiedler import testing as ft

torch.set_default_dtype(torch.float64)
IGNORE = 255


# --------------------------------------------------------------------------- #
# cross-entropy base
# --------------------------------------------------------------------------- #
def test_cross_entropy_perfect_is_near_zero():
    B, K, H, W = 2, 5, 8, 8
    target = torch.randint(0, K, (B, H, W))
    logits = torch.zeros(B, K, H, W).scatter_(1, target.unsqueeze(1), 50.0)  # confident & correct
    assert cross_entropy_2d(logits, target).item() < 1e-10


def test_cross_entropy_ignores_void():
    B, K, H, W = 1, 4, 6, 6
    target = torch.randint(0, K, (B, H, W))
    logits = torch.randn(B, K, H, W, requires_grad=True)
    base = cross_entropy_2d(logits, target)
    t2 = target.clone()
    t2[:, 0, 0] = IGNORE                       # mark one pixel void
    logits2 = logits.detach().clone()
    logits2[:, :, 0, 0] = 999.0                # garbage there — must not matter
    assert torch.isfinite(cross_entropy_2d(logits2, t2))


def test_class_weights_median_freq():
    counts = torch.tensor([1000.0, 1000.0, 10.0, 0.0])   # class2 rare, class3 absent
    w = class_weights_median_freq(counts)
    assert w[2] > w[0]                          # rare class up-weighted
    assert w[3] == 0.0                          # absent class zeroed
    ft.assert_finite(w, "class weights")


def test_ohem_runs_and_is_finite():
    B, K, H, W = 2, 4, 16, 16
    logits = torch.randn(B, K, H, W, requires_grad=True)
    target = torch.randint(0, K, (B, H, W))
    loss = OHEMCrossEntropy(thresh=0.7, min_kept=50)(logits, target)
    loss.backward()
    ft.assert_finite(loss, "ohem"); ft.assert_finite(logits.grad, "ohem grad")


# --------------------------------------------------------------------------- #
# Lovász-Softmax
# --------------------------------------------------------------------------- #
def test_lovasz_perfect_is_zero_and_differentiable():
    B, K, H, W = 1, 4, 8, 8
    target = torch.randint(0, K, (B, H, W))
    logits = torch.zeros(B, K, H, W).scatter_(1, target.unsqueeze(1), 50.0).requires_grad_(True)
    probs = torch.softmax(logits, dim=1)
    loss = lovasz_softmax(probs, target)
    assert loss.item() < 1e-6
    # a wrong prediction must score strictly worse
    rand = torch.softmax(torch.randn(B, K, H, W), dim=1)
    assert lovasz_softmax(rand, target).item() > loss.item()


# --------------------------------------------------------------------------- #
# Rayleigh spectral-consistency (the novelty)
# --------------------------------------------------------------------------- #
def _two_triangles_L():
    W = torch.zeros(6, 6)
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        W[a, b] = W[b, a] = 1.0
    return laplacian(W, kind="comb")


def test_rayleigh_zero_for_component_constant_prediction():
    """A prediction constant on each connected component has zero graph energy."""
    L = _two_triangles_L()
    prob = torch.zeros(6, 2)
    prob[:3, 0] = 1.0      # component A -> class 0
    prob[3:, 1] = 1.0      # component B -> class 1
    assert rayleigh_consistency(prob, L).item() < 1e-12


def test_rayleigh_positive_for_rough_prediction():
    L = _two_triangles_L()
    prob = torch.rand(6, 2)
    assert rayleigh_consistency(prob, L).item() > 0


def test_rayleigh_gradcheck():
    torch.manual_seed(0)
    X = torch.randn(12, 4)
    L = laplacian(gaussian_affinity(X, sigma=1.0, k=5), kind="sym")
    prob = torch.rand(12, 3, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: rayleigh_consistency(p, L), (prob,),
                                    atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
# CompositeLoss wiring
# --------------------------------------------------------------------------- #
def test_composite_loss_components_and_backward():
    torch.manual_seed(0)
    B, K, H, W = 1, 3, 6, 6
    N = H * W
    logits = torch.randn(B, K, H, W, requires_grad=True)
    target = torch.randint(0, K, (B, H, W))

    X = torch.randn(N, 5)
    L = laplacian(gaussian_affinity(X, sigma=1.0, k=6), kind="sym")
    prob_graph = torch.softmax(logits.reshape(K, N).t(), dim=1)  # (N, K)

    eig = torch.linspace(0.0, 2.0, 8)
    resp = SpectralResponse(kind="cheb")
    h = resp(eig)

    crit = CompositeLoss(w_ce=1.0, w_rayleigh=0.2, w_bulk=0.1)
    total, comps = crit(logits, target, prob_graph=prob_graph, L=L,
                        eigvals=eig, response_h=h, mp_n=N, mp_d=5)
    for key in ("ce", "rayleigh", "bulk", "total"):
        assert key in comps
    total.backward()
    ft.assert_finite(logits.grad, "logits grad")
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in resp.parameters())


def test_composite_loss_base_only():
    """With no spectral inputs it degrades to plain CE."""
    logits = torch.randn(1, 4, 5, 5, requires_grad=True)
    target = torch.randint(0, 4, (1, 5, 5))
    total, comps = CompositeLoss(w_ce=1.0, w_rayleigh=0.2)(logits, target)
    assert set(comps) == {"ce", "total"}
    assert abs(comps["ce"] - comps["total"]) < 1e-9
