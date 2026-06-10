"""Render a single visual snapshot of cheeger: both heads' predictions + the
spectral internals, in one figure. Trains briefly on a toy scene, then plots.

Run:  python demos/viz_snapshot.py   ->   results/cheeger_snapshot.png
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from fiedler.models import UNet, ConvHead, SpectralSegHead, SegModel
from fiedler.losses import cross_entropy_2d
from fiedler.metrics import ConfusionMatrix, mean_iou
from fiedler.utils import seed_everything
from visualizers.palette import cityscapes_palette

K = 4
PAL = cityscapes_palette()


def toy_scene(H=32, W=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.zeros(H, W, dtype=torch.long)
    gt[:H // 2, :W // 2] = 0; gt[:H // 2, W // 2:] = 1
    gt[H // 2:, :W // 2] = 2; gt[H // 2:, W // 2:] = 3
    gt[H // 3:2 * H // 3, W // 3:2 * W // 3] = 1
    img = torch.zeros(3, H, W)
    img[0] = ((gt == 1) | (gt == 3)).float(); img[1] = (gt >= 2).float()
    img[2] = torch.rand(H, W, generator=g)
    return img.unsqueeze(0), gt.unsqueeze(0)


def color(mask):
    return PAL[mask.long().clamp(0, PAL.shape[0] - 1)].numpy() / 255.0   # -> float [0,1]


def train(head, steps, lr, spectral, img, gt):
    seed_everything(0)
    model = SegModel(UNet(3, base=8, depth=2, out_channels=8), head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = cross_entropy_2d(model(img)["logits"], gt)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(img)["logits"].argmax(1)[0]
    cm = ConfusionMatrix(K, 255); cm.update(pred.unsqueeze(0), gt)
    return model, pred, mean_iou(cm.compute())


def main():
    img, gt = toy_scene()
    conv_m, conv_pred, conv_miou = train(ConvHead(8, K), 150, 1e-2, False, img, gt)
    spec_m, spec_pred, spec_miou = train(
        SpectralSegHead(8, K, graph_hw=12, k=8, mlp_hidden=32), 250, 1e-2, True, img, gt)

    # spectral internals on the trained spectral model
    with torch.no_grad():
        feat = spec_m.backbone(img)
        g = F.adaptive_avg_pool2d(feat, (12, 12))
        X = g[0].permute(1, 2, 0).reshape(144, -1)
        emb = spec_m.head.embed(X)
    eig = emb["eigvals"].numpy(); h = emb["response"].numpy()
    V = emb["eigvecs"].numpy()
    gt_small = F.interpolate(gt.unsqueeze(1).float(), size=(12, 12), mode="nearest").long().reshape(-1).numpy()

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("cheeger — predictions & spectral internals (toy overfit)", fontsize=14)

    titles = [f"input", "ground truth",
              f"conv head   mIoU={conv_miou:.2f}", f"spectral head   mIoU={spec_miou:.2f}"]
    imgs = [img[0].permute(1, 2, 0).numpy(), color(gt[0]), color(conv_pred), color(spec_pred)]
    for i, (t, im) in enumerate(zip(titles, imgs)):
        ax = fig.add_subplot(2, 4, i + 1); ax.imshow(im.clip(0, 1)); ax.set_title(t, fontsize=10); ax.axis("off")

    ax = fig.add_subplot(2, 4, 5); ax.plot(eig, h, "o-", color="#3C3489")
    ax.set_title("learned response  hθ(λ)", fontsize=10); ax.set_xlabel("λ"); ax.set_ylabel("h")

    ax = fig.add_subplot(2, 4, 6); ax.bar(range(len(eig)), eig, color="#085041")
    ax.set_title("Laplacian spectrum", fontsize=10); ax.set_xlabel("index"); ax.set_ylabel("λ")

    ax = fig.add_subplot(2, 4, 7)
    sc = ax.scatter(V[:, 1], V[:, 2], c=gt_small, cmap="tab10", s=28, edgecolors="none")
    ax.set_title("spectral embedding (v2 vs v3)", fontsize=10); ax.set_xlabel("v2 (Fiedler)"); ax.set_ylabel("v3")

    ax = fig.add_subplot(2, 4, 8); ax.axis("off")
    ax.text(0.0, 0.5,
            f"conv mIoU      {conv_miou:.3f}\nspectral mIoU  {spec_miou:.3f}\n\n"
            f"graph: 12×12 = 144 nodes\nk = 8 eigenvectors\nLaplacian: L_sym",
            fontsize=11, family="monospace", va="center")

    out = ROOT / "results/cheeger_snapshot.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"conv mIoU={conv_miou:.3f}  spectral mIoU={spec_miou:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
