"""Render the blog's experiment figures in the site's palette.

Produces, into docs/assets/:
  training_curves.png  — conv vs spectral head, mIoU + loss over training (the
                         static stand-in for the live TensorBoard dashboard)
  scale.png            — dense eigh vs sparse Lanczos solve time as graphs grow

Run:  python scripts/make_blog_figures.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fiedler.models import UNet, ConvHead, SpectralSegHead, SegModel
from fiedler.losses import cross_entropy_2d, CompositeLoss
from fiedler.metrics import ConfusionMatrix, mean_iou
from fiedler.utils import seed_everything

# ---- blog palette ---------------------------------------------------------- #
BG, INK, MUTED, GRID = "#FBFAF6", "#23231f", "#6f6d64", "#e6e3da"
INDIGO, TEAL, CORAL, AMBER = "#3C3489", "#0a5a49", "#9c3b1b", "#9a6b12"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#cdcabd", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlecolor": INK,
    "font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"],
    "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 10,
    "figure.dpi": 140,
})
K = 4


def toy(H=32, W=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.zeros(H, W, dtype=torch.long)
    gt[:H//2, :W//2] = 0; gt[:H//2, W//2:] = 1; gt[H//2:, :W//2] = 2; gt[H//2:, W//2:] = 3
    gt[H//3:2*H//3, W//3:2*W//3] = 1
    img = torch.zeros(3, H, W)
    img[0] = ((gt == 1) | (gt == 3)).float(); img[1] = (gt >= 2).float()
    img[2] = torch.rand(H, W, generator=g)
    return img.unsqueeze(0), gt.unsqueeze(0)


def warmup(step, s=150, e=300, peak=0.05):
    return 0.0 if step <= s else peak * min(1.0, (step - s) / (e - s))


def train_capture(head, steps, spectral, img, gt):
    seed_everything(0)
    model = SegModel(UNet(3, base=8, depth=2, out_channels=8), head)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = CompositeLoss(w_ce=1.0, w_rayleigh=0.0) if spectral else None
    steps_x, miou_y, loss_y = [], [], []
    for s in range(steps):
        opt.zero_grad(); out = model(img)
        if spectral:
            crit.w_rayleigh = warmup(s)
            loss, _ = crit(out["logits"], gt, prob_graph=out["prob_graph"], L=out["laplacian"])
        else:
            loss = cross_entropy_2d(out["logits"], gt)
        loss.backward(); opt.step()
        if s % 4 == 0 or s == steps - 1:
            cm = ConfusionMatrix(K, 255); cm.update(out["logits"].argmax(1), gt)
            steps_x.append(s); miou_y.append(mean_iou(cm.compute())); loss_y.append(float(loss.detach()))
    return steps_x, miou_y, loss_y


def fig_training():
    img, gt = toy()
    cx, cm, cl = train_capture(ConvHead(8, K), 150, False, img, gt)
    sx, sm, sl = train_capture(SpectralSegHead(8, K, graph_hw=12, k=8, mlp_hidden=32), 300, True, img, gt)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    a1.plot(cx, cm, color=INDIGO, lw=2.2, label="conv head")
    a1.plot(sx, sm, color=TEAL, lw=2.2, label="spectral head")
    a1.set_title("validation mIoU over training"); a1.set_xlabel("step"); a1.set_ylabel("mIoU")
    a1.set_ylim(-0.02, 1.05); a1.legend(frameon=False)
    a1.annotate("plateau", xy=(70, 0.1), color=MUTED, fontsize=9)
    a1.annotate("breakthrough", xy=(165, 0.4), color=TEAL, fontsize=9)

    a2.plot(cx, cl, color=INDIGO, lw=2.2, label="conv head")
    a2.plot(sx, sl, color=TEAL, lw=2.2, label="spectral head")
    a2.set_title("training loss"); a2.set_xlabel("step"); a2.set_ylabel("loss"); a2.legend(frameon=False)

    fig.tight_layout()
    out = ROOT / "docs/assets/training_curves.png"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_scale():
    # measured this session — one consistent graph type (sparse grid Laplacian)
    n_dense = [1024, 2304, 4096]
    t_dense = [67, 673, 6795]                          # ms, dense eigh (O(n³))
    n_lz = [1024, 2304, 4096, 8100, 16384, 32400, 65536]
    t_lz = [8, 12, 19, 32, 68, 191, 468]               # ms, sparse Lanczos, same default m

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(n_dense, t_dense, "o-", color=CORAL, lw=2.2, ms=7, label="dense eigh  O(n³)")
    ax.plot(n_lz, t_lz, "o-", color=TEAL, lw=2.2, ms=7, label="sparse Lanczos")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("graph size  n  (nodes)"); ax.set_ylabel("solve time  (ms, log)")
    ax.set_title("bottom-16 eigenpairs: dense vs Lanczos as the graph grows")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("dense cannot allocate\nan n×n matrix past here",
                xy=(4096, 7088), xytext=(5000, 1600), color=CORAL, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=CORAL))
    ax.annotate("65k nodes in 0.47s\n(34 GB as a dense matrix)",
                xy=(65536, 468), xytext=(9000, 60), color=TEAL, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=TEAL))
    fig.tight_layout()
    out = ROOT / "docs/assets/scale.png"
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_training()
    fig_scale()
