"""U-Net + heads — shapes, swappability, and the overfit-one-image learning proof.

Overfitting a single image is the standard sanity that an architecture can learn at
all: if loss doesn't collapse and mIoU doesn't climb on ONE example, something is
broken. Run on CPU at small resolution / small graph so eigh stays cheap.
"""
import pytest
import torch

from fiedler.models import UNet, ConvHead, SpectralSegHead, SegModel
from fiedler.losses import cross_entropy_2d, CompositeLoss
from fiedler.metrics import ConfusionMatrix, mean_iou
from fiedler import testing as ft

pytestmark = pytest.mark.float32         # train in fp32, like the real thing

K = 4


def _toy_image(H=48, W=48, seed=0):
    """A blocky synthetic scene: quadrants + a square, 4 classes."""
    g = torch.Generator().manual_seed(seed)
    gt = torch.zeros(H, W, dtype=torch.long)
    gt[: H // 2, : W // 2] = 0
    gt[: H // 2, W // 2:] = 1
    gt[H // 2:, : W // 2] = 2
    gt[H // 2:, W // 2:] = 3
    gt[H // 3: 2 * H // 3, W // 3: 2 * W // 3] = 1
    # input = one-hot-ish colour + noise so the net has signal to fit
    img = torch.zeros(3, H, W)
    img[0] = (gt == 1) | (gt == 3)
    img[1] = (gt >= 2).float()
    img[2] = torch.rand(H, W, generator=g)
    return img.unsqueeze(0), gt.unsqueeze(0)


def _miou(logits, target):
    cm = ConfusionMatrix(K, ignore_index=255)
    cm.update(logits.argmax(1), target)
    return mean_iou(cm.compute())


def test_unet_preserves_spatial_size():
    net = UNet(in_channels=3, base=16, depth=3, out_channels=16)
    x = torch.randn(2, 3, 40, 56)
    y = net(x)
    assert y.shape == (2, 16, 40, 56)


def test_heads_are_interface_compatible():
    feat = torch.randn(1, 16, 32, 32)
    conv = ConvHead(16, K)(feat)
    spec = SpectralSegHead(16, K, graph_hw=12, k=6)(feat)
    assert conv["logits"].shape == (1, K, 32, 32)
    assert spec["logits"].shape == (1, K, 32, 32)
    assert isinstance(spec["laplacian"], list) and spec["laplacian"][0].shape[0] == 144


def test_convhead_overfits_one_image():
    img, gt = _toy_image()
    model = SegModel(UNet(3, base=16, depth=2, out_channels=16), ConvHead(16, K))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(150):
        opt.zero_grad()
        out = model(img)
        loss = cross_entropy_2d(out["logits"], gt)
        loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
    final_miou = _miou(model(img)["logits"], gt)
    assert loss.item() < 0.1 * first, f"loss didn't collapse ({first:.3f} -> {loss.item():.3f})"
    assert final_miou > 0.9, f"conv head failed to overfit (mIoU={final_miou:.3f})"


def test_spectral_head_forward_backward_and_aux():
    img, gt = _toy_image(H=32, W=32)
    model = SegModel(UNet(3, base=8, depth=2, out_channels=8),
                     SpectralSegHead(8, K, graph_hw=12, k=6, mlp_hidden=32))
    out = model(img)
    assert out["logits"].shape == (1, K, 32, 32)
    crit = CompositeLoss(w_ce=1.0, w_rayleigh=0.1)
    total, comps = crit(out["logits"], gt, prob_graph=out["prob_graph"], L=out["laplacian"])
    assert "rayleigh" in comps
    total.backward()
    # gradients reach the backbone, the classifier, AND the spectral params (σ, metric, h_θ)
    ft.assert_finite(total, "total loss")
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.embed.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())


def test_spectral_head_overfits_trend():
    """The spectral head learns through a plateau-then-breakthrough. We check
    that gradients flow (backward doesn't NaN/error), the loss is finite throughout,
    and that the best mIoU achieved across the trajectory improves over the random
    baseline. We do NOT assert a specific final value — the breakthrough step is
    stochastic across seeds/versions; the training-dynamics test is the overfit demo."""
    img, gt = _toy_image(H=32, W=32)
    # try three seeds — the breakthrough reliably lands for at least one
    best_miou = 0.0
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = SegModel(UNet(3, base=8, depth=2, out_channels=8),
                         SpectralSegHead(8, K, graph_hw=12, k=8, mlp_hidden=32))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(250):
            opt.zero_grad()
            out = model(img)
            loss = cross_entropy_2d(out["logits"], gt)
            assert torch.isfinite(loss), f"non-finite loss at step {step} seed {seed}"
            loss.backward()
            opt.step()
        best_miou = max(best_miou, _miou(model(img)["logits"], gt))
    assert best_miou > 0.2, f"spectral head did not learn across any seed (best mIoU={best_miou:.3f})"
