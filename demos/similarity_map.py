"""Feature-similarity heatmap visualizer.

Click any pixel (via --query r c) and see which other pixels share a similar
spectral embedding — the same visualization used in DINO / deep-spectral papers.

Left panel : original image + query marker
Right panel: cosine similarity of the query pixel's embedding to every other pixel

Works with random backbone weights (shows graph structure, not semantics) or a
trained checkpoint (shows semantic grouping). The spectral embedding is computed
at ``--graph-hw`` resolution (default 32 → 1024 nodes; affordable on CPU).

Usage
-----
    # synthetic colored image, query at centre
    python demos/similarity_map.py

    # real image, custom query point
    python demos/similarity_map.py --image path/to/img.jpg --query 120 80

    # save to a specific path
    python demos/similarity_map.py --out results/my_sim.png
"""
import sys
import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fiedler.models.unet import UNet
from fiedler.spectral.embedding import SpectralEmbedding
from fiedler.utils import get_device


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _synthetic_image(size=(128, 256)) -> torch.Tensor:
    """Three-region colored image (H, W, 3) float32 in [0,1]."""
    H, W = size
    img = torch.zeros(H, W, 3)
    img[:, : W // 3] = torch.tensor([0.85, 0.20, 0.15])   # red zone
    img[:, W // 3 : 2 * W // 3] = torch.tensor([0.15, 0.65, 0.25])  # green zone
    img[:, 2 * W // 3 :] = torch.tensor([0.15, 0.35, 0.80])  # blue zone
    # add a bright foreground blob (top-centre) so there's a focal object
    r0, c0, r1, c1 = H // 5, W // 3, 3 * H // 5, 2 * W // 3
    img[r0:r1, c0:c1] = torch.tensor([0.95, 0.90, 0.20])
    return img


def _load_image(path: str, size=(128, 256)) -> torch.Tensor:
    """Load an image file → (H, W, 3) float32 tensor in [0,1]."""
    try:
        from PIL import Image
        import numpy as np
        pil = Image.open(path).convert("RGB").resize((size[1], size[0]))
        return torch.from_numpy(np.array(pil)).float() / 255.0
    except ImportError:
        raise SystemExit("Pillow is required to load image files: pip install pillow")


def _embed(img_hwc: torch.Tensor, graph_hw: int, k: int,
           device: torch.device) -> torch.Tensor:
    """Run UNet backbone + SpectralEmbedding → phi (graph_hw², k)."""
    img = img_hwc.permute(2, 0, 1).unsqueeze(0).to(device)   # (1, 3, H, W)

    backbone = UNet(3, base=32, depth=3, out_channels=64).to(device)
    backbone.eval()

    with torch.no_grad():
        feat = backbone(img)                                   # (1, 64, H, W)
        g = F.adaptive_avg_pool2d(feat, (graph_hw, graph_hw)) # (1, 64, gh, gw)
        X = g[0].permute(1, 2, 0).reshape(graph_hw * graph_hw, -1)  # (N, C)

        emb = SpectralEmbedding(k=k, knn=16, solver="dense").to(device)
        out = emb(X)
        phi = out["phi"]                                       # (N, k)

    return phi.cpu()


def _cosine_sim(phi: torch.Tensor, node_idx: int) -> torch.Tensor:
    """Cosine similarity of node_idx to every node → (N,)."""
    q = phi[node_idx]
    q = q / (q.norm() + 1e-8)
    p = phi / (phi.norm(dim=1, keepdim=True) + 1e-8)
    return (p @ q).clamp(-1, 1)


def _pixel_to_node(r: int, c: int, img_h: int, img_w: int,
                   graph_hw: int) -> int:
    gr = int(r / img_h * graph_hw)
    gc = int(c / img_w * graph_hw)
    gr = min(gr, graph_hw - 1)
    gc = min(gc, graph_hw - 1)
    return gr * graph_hw + gc


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="path to an image file")
    ap.add_argument("--query", nargs=2, type=int, default=None,
                    metavar=("ROW", "COL"),
                    help="query pixel in the original image (default: centre)")
    ap.add_argument("--graph-hw", type=int, default=32,
                    help="graph resolution (graph_hw² nodes; default 32)")
    ap.add_argument("--k", type=int, default=16,
                    help="number of eigenvectors (default 16)")
    ap.add_argument("--size", nargs=2, type=int, default=[128, 256],
                    metavar=("H", "W"),
                    help="resize image to H W before processing (default 128 256)")
    ap.add_argument("--out", default="results/similarity_map.png")
    args = ap.parse_args()

    size = tuple(args.size)  # (H, W)
    H, W = size

    # --- load image ---
    if args.image:
        img_hwc = _load_image(args.image, size)
        print(f"loaded {args.image}  →  {H}×{W}")
    else:
        img_hwc = _synthetic_image(size)
        print(f"using synthetic image  {H}×{W}  (pass --image to use your own)")

    # --- query pixel ---
    if args.query:
        qr, qc = args.query
    else:
        qr, qc = H // 2, W // 2
    qr = max(0, min(qr, H - 1))
    qc = max(0, min(qc, W - 1))
    node_idx = _pixel_to_node(qr, qc, H, W, args.graph_hw)
    print(f"query pixel ({qr}, {qc})  →  graph node {node_idx}"
          f"  (graph {args.graph_hw}×{args.graph_hw})")

    # --- embed ---
    device = get_device()
    # eigh not supported on MPS; fall back silently
    if str(device) == "mps":
        device = torch.device("cpu")
    print(f"running on {device}  (graph_hw={args.graph_hw}, k={args.k}) …")
    phi = _embed(img_hwc, args.graph_hw, args.k, device)

    # --- similarity map ---
    sim = _cosine_sim(phi, node_idx)                          # (N,)
    sim_map = sim.reshape(args.graph_hw, args.graph_hw)       # (gh, gw)
    sim_up = F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode="bilinear", align_corners=False,
    )[0, 0].numpy()

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#111111")

    # left: original + query marker
    ax = axes[0]
    ax.imshow(img_hwc.numpy())
    ax.scatter([qc], [qr], s=120, c="yellow", marker="s",
               linewidths=1.5, edgecolors="white", zorder=5)
    ax.set_title("input  +  query pixel", color="white", fontsize=11)
    ax.axis("off")

    # right: heatmap
    ax = axes[1]
    ax.set_facecolor("#111111")
    hm = ax.imshow(sim_up, cmap="magma", vmin=-1, vmax=1)
    ax.set_title("spectral embedding similarity", color="white", fontsize=11)
    ax.axis("off")
    cb = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    note = ("random weights — structure reflects graph geometry\n"
            "train the model to get semantic grouping")
    fig.text(0.5, 0.01, note, ha="center", color="#888888", fontsize=8)

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
