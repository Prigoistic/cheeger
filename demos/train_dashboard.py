"""Live training dashboard — conv head vs spectral head, streamed to TensorBoard.

Trains both heads on a small synthetic driving-ish set and logs, live, the panels a
DINO/JEPA-style run would show — plus spectral-specific views:

  scalars : train loss + components (ce, rayleigh), train/val mIoU, val boundary-IoU,
            and the learned affinity bandwidth σ
  images  : [input | GT | prediction] overlay strip on a held-out image
  figures : the learned spectral response h_θ(λ) and the Laplacian eigen-spectrum

Run:
    python demos/train_dashboard.py
    tensorboard --logdir results/runs        # then open http://localhost:6006
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fiedler.models import UNet, ConvHead, SpectralSegHead, SegModel
from fiedler.losses import cross_entropy_2d, CompositeLoss
from fiedler.metrics import ConfusionMatrix, mean_iou, boundary_iou
from fiedler.engine.logging import ExperimentLogger
from fiedler.utils import seed_everything
from visualizers.palette import cityscapes_palette

K = 4
PALETTE = cityscapes_palette()


def toy_scene(H=32, W=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.zeros(H, W, dtype=torch.long)
    gt[: H // 2, : W // 2] = 0
    gt[: H // 2, W // 2:] = 1
    gt[H // 2:, : W // 2] = 2
    gt[H // 2:, W // 2:] = 3
    gt[H // 3: 2 * H // 3, W // 3: 2 * W // 3] = 1
    img = torch.zeros(3, H, W)
    img[0] = ((gt == 1) | (gt == 3)).float()
    img[1] = (gt >= 2).float()
    img[2] = torch.rand(H, W, generator=g)
    return img.unsqueeze(0), gt.unsqueeze(0)


def evaluate(model, img, gt):
    model.eval()
    with torch.no_grad():
        logits = model(img)["logits"]
    pred = logits.argmax(1)
    cm = ConfusionMatrix(K, 255); cm.update(pred, gt)
    model.train()
    return mean_iou(cm.compute()), boundary_iou(pred[0], gt[0], K), pred[0]


def spectral_panels(out):
    """matplotlib figure: learned response h_θ(λ) and the eigen-spectrum."""
    eig = out["eigvals"][0].detach().cpu()
    h = out["response"][0].detach().cpu()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3))
    a1.plot(eig.numpy(), h.numpy(), "o-"); a1.set_title("learned response hθ(λ)")
    a1.set_xlabel("λ"); a1.set_ylabel("h")
    a2.bar(range(len(eig)), eig.numpy()); a2.set_title("Laplacian spectrum")
    a2.set_xlabel("index"); a2.set_ylabel("λ")
    fig.tight_layout()
    return fig


def rayleigh_warmup(step, start=150, end=350, peak=0.05):
    """Ramp the Rayleigh weight 0 -> peak over [start, end]. The consistency term is
    harmful at init (a near-uniform graph makes 'smooth' = uniform, fighting CE) and
    helpful once features organise — so warm it up after the CE breakthrough."""
    if step <= start:
        return 0.0
    return peak * min(1.0, (step - start) / max(1, end - start))


def train_head(name, head, steps, lr, spectral=False, log_every=20):
    seed_everything(0)
    # overfit-one-image dashboard: train and evaluate on the same scene so the live
    # curve cleanly shows each head's learning dynamics (incl. the spectral plateau).
    train = [toy_scene(seed=0)]
    val_img, val_gt = train[0]

    model = SegModel(UNet(3, base=8, depth=2, out_channels=8), head)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = CompositeLoss(w_ce=1.0, w_rayleigh=0.0) if spectral else None  # warmed up below
    logger = ExperimentLogger(str(ROOT / "results/runs" / name))
    print(f"[{name}] training {steps} steps ...")

    for step in range(steps):
        img, gt = train[step % len(train)]
        opt.zero_grad()
        out = model(img)
        if spectral:
            crit.w_rayleigh = rayleigh_warmup(step)
            loss, comps = crit(out["logits"], gt, prob_graph=out["prob_graph"], L=out["laplacian"])
        else:
            loss = cross_entropy_2d(out["logits"], gt); comps = {"ce": float(loss.detach())}
        loss.backward(); opt.step()

        scalars = {"train/loss": float(loss), "train/ce": comps.get("ce", 0.0)}
        if spectral:
            scalars["train/rayleigh"] = comps.get("rayleigh", 0.0)
            scalars["params/w_rayleigh"] = crit.w_rayleigh
            scalars["params/sigma"] = float(model.head.embed.sigma)
        logger.log_scalars(step, **scalars)

        if step % log_every == 0 or step == steps - 1:
            vmiou, vbiou, vpred = evaluate(model, val_img, val_gt)
            logger.log_scalars(step, **{"val/mIoU": vmiou, "val/boundary_iou": vbiou})
            logger.log_seg("val/prediction", val_img[0], val_gt[0], vpred, PALETTE, step)
            if spectral:
                logger.log_figure("spectral/response", spectral_panels(model(val_img)), step)
            logger.flush()
            print(f"  [{name}] step {step:3d}  loss={float(loss):.3f}  val mIoU={vmiou:.3f}  bIoU={vbiou:.3f}")

    logger.close()
    print(f"[{name}] done.")


def main():
    train_head("conv", ConvHead(8, K), steps=150, lr=1e-2, spectral=False)
    train_head("spectral", SpectralSegHead(8, K, graph_hw=12, k=8, mlp_hidden=32),
               steps=400, lr=1e-2, spectral=True)
    print("\nDashboards written to results/runs/{conv,spectral}")
    print("View:  tensorboard --logdir results/runs   ->   http://localhost:6006")


if __name__ == "__main__":
    main()
