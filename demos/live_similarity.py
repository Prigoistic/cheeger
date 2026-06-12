"""Live webcam feature-similarity heatmap.

Click anywhere on the LEFT panel to pick a query pixel.
The RIGHT panel shows cosine similarity of that pixel's spectral embedding
to every other pixel — bright = similar, dark = dissimilar.

Controls
--------
  left-click   set query pixel
  n            toggle auto-range (per-frame contrast stretch vs fixed [-1,1])
  c            cycle colourmap (INFERNO → MAGMA → VIRIDIS → JET → INFERNO …)
  s            save current frame to results/live_sim_<n>.png
  q / ESC      quit

Feature modes
-------------
  default      : raw RGB + std + gradient (7-D) + self-attention diffusion
                 Best effort until the UNet is trained.
  --backbone   : our UNet feature extractor.  With random weights the result
                 won't look semantic — the value comes once we plug in a
                 trained checkpoint via --ckpt.
  --no-diffuse : disable diffusion (pure local features, baseline)

Usage
-----
    python demos/live_similarity.py                  # diffuse mode, 24×24 graph
    python demos/live_similarity.py --backbone --ckpt results/best.pt
    python demos/live_similarity.py --graph-hw 32    # larger graph (slower)
    python demos/live_similarity.py --fixed-range    # disable per-frame stretch
"""
import sys
import argparse
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fiedler.spectral.embedding import SpectralEmbedding


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

_COLORMAPS = [
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_JET,
]

_BLUR_SIGMA_FRAC = 0.018   # gaussian blur sigma as fraction of display height


# --------------------------------------------------------------------------- #
# feature extraction
# --------------------------------------------------------------------------- #

def _raw_features(frame_rgb: np.ndarray, graph_hw: int) -> torch.Tensor:
    """Per-cell: mean RGB (3) + local std RGB (3) + mean grad mag (1) → (N, 7)."""
    gh = graph_hw
    rgb_f = frame_rgb.astype(np.float32) / 255.0

    mean_rgb  = cv2.resize(rgb_f, (gh, gh), interpolation=cv2.INTER_AREA)
    mean_rgb2 = cv2.resize(rgb_f ** 2, (gh, gh), interpolation=cv2.INTER_AREA)
    std_rgb   = np.sqrt(np.maximum(mean_rgb2 - mean_rgb ** 2, 0.0))

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_small = cv2.resize(np.sqrt(gx**2 + gy**2), (gh, gh),
                            interpolation=cv2.INTER_AREA)[:, :, None]

    feat_map = np.concatenate([mean_rgb, std_rgb, grad_small], axis=2)
    return torch.from_numpy(feat_map.reshape(gh * gh, 7).astype(np.float32))


def _diffuse(X: torch.Tensor, n_iters: int = 3, temperature: float = 10.0) -> torch.Tensor:
    """Iterative self-attention feature diffusion — no learned weights.

    Each step: A = softmax(X Xᵀ · temp),  X ← A X,  X ← normalize(X).

    Propagates similarity information globally so every cell "sees" which other
    cells share its character.  After k iters, a face-skin cell has aggregated
    from ALL similar-looking cells; the spectral graph then finds a much cleaner
    cut between face and background even without trained features.

    Once the UNet is trained this step becomes redundant — the backbone features
    will carry that semantic context directly.
    Cost: O(k · N² · D) — < 0.5 ms for N = 576, D = 7.
    """
    X = F.normalize(X, dim=1)
    for _ in range(n_iters):
        A = X @ X.T * temperature
        A.fill_diagonal_(float("-inf"))      # no self-loop
        A = torch.softmax(A, dim=1)
        X = F.normalize(A @ X, dim=1)
    return X


def _backbone_features(frame_rgb: np.ndarray, graph_hw: int,
                       backbone, device: torch.device) -> torch.Tensor:
    """Run our UNet backbone → pool to graph_hw×graph_hw → (N, C).

    With random weights this gives noise.  Pass --ckpt to load trained weights
    and this becomes the semantic feature extractor.
    """
    t = (torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
         .permute(2, 0, 1).unsqueeze(0).to(device))
    with torch.no_grad():
        feat = backbone(t)                                    # (1, C, H, W)
        g = F.adaptive_avg_pool2d(feat, (graph_hw, graph_hw))
        return g[0].permute(1, 2, 0).reshape(graph_hw * graph_hw, -1).cpu()


# --------------------------------------------------------------------------- #
# spectral similarity
# --------------------------------------------------------------------------- #

def _embed(X: torch.Tensor, emb: SpectralEmbedding) -> torch.Tensor:
    with torch.no_grad():
        return emb(X)["phi"]                                  # (N, k)


def _cosine_sim(phi: torch.Tensor, node_idx: int) -> np.ndarray:
    q = phi[node_idx]
    q = q / (q.norm() + 1e-8)
    p = phi / (phi.norm(dim=1, keepdim=True) + 1e-8)
    return (p @ q).clamp(-1, 1).numpy()                      # (N,)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def _sim_to_heatmap(sim: np.ndarray, graph_hw: int,
                    disp_h: int, disp_w: int,
                    auto_range: bool = True,
                    cmap_idx: int = 0) -> np.ndarray:
    grid = sim.reshape(graph_hw, graph_hw)
    if auto_range:
        lo, hi = grid.min(), grid.max()
        u8 = ((grid - lo) / (hi - lo + 1e-6) * 255).clip(0, 255).astype(np.uint8)
    else:
        u8 = ((grid + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

    big = cv2.resize(u8, (disp_w, disp_h), interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=max(1.0, _BLUR_SIGMA_FRAC * disp_h))
    return cv2.applyColorMap(big, _COLORMAPS[cmap_idx % len(_COLORMAPS)])


# --------------------------------------------------------------------------- #
# mouse callback
# --------------------------------------------------------------------------- #

class State:
    def __init__(self, disp_w: int):
        self.qr = self.qc = None
        self.disp_w = disp_w
        self.clicked = False

    def callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and x < self.disp_w:
            self.qc, self.qr = x, y
            self.clicked = True


def _pixel_to_node(r: int, c: int, H: int, W: int, gh: int) -> int:
    return min(int(r / H * gh), gh - 1) * gh + min(int(c / W * gh), gh - 1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",      type=int,   default=0)
    ap.add_argument("--graph-hw",    type=int,   default=24,
                    help="graph resolution side (default 24 → 576 nodes)")
    ap.add_argument("--k",           type=int,   default=12,
                    help="number of eigenvectors (default 12)")
    ap.add_argument("--disp-h",      type=int,   default=360)
    ap.add_argument("--backbone",    action="store_true",
                    help="use our UNet feature extractor (needs --ckpt for trained weights)")
    ap.add_argument("--ckpt",        default=None,
                    help="path to trained UNet checkpoint (used with --backbone)")
    ap.add_argument("--no-diffuse",  action="store_true",
                    help="disable self-attention diffusion in default mode")
    ap.add_argument("--n-iters",     type=int,   default=3,
                    help="diffusion iterations (default 3)")
    ap.add_argument("--fixed-range", action="store_true",
                    help="use fixed [-1,1] range instead of per-frame stretch")
    args = ap.parse_args()

    # --- camera ---
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.device}")
    ret, frame = cap.read()
    if not ret:
        sys.exit("Cannot read from camera")

    disp_h = args.disp_h
    disp_w = int(frame.shape[1] / frame.shape[0] * disp_h)
    gh = args.graph_hw
    knn = min(16, gh * gh - 1)

    # --- feature backbone ---
    backbone = None
    if args.backbone:
        from fiedler.models.unet import UNet
        torch_device = torch.device("cpu")
        backbone = UNet(3, base=16, depth=2, out_channels=32).to(torch_device)
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt)
            backbone.load_state_dict(state)
            mode = f"UNet [trained: {pathlib.Path(args.ckpt).name}]"
        else:
            mode = "UNet [random weights — train first for semantic results]"
        backbone.eval()
    else:
        torch_device = torch.device("cpu")
        mode = "rgb+diffuse" if not args.no_diffuse else "rgb+tex+grad"

    # --- spectral embedding (always CPU — eigh not on MPS) ---
    emb = SpectralEmbedding(k=args.k, knn=knn, solver="dense").to(torch.device("cpu"))
    emb.eval()

    print(f"features: {mode}  |  graph {gh}×{gh}={gh*gh} nodes  |  k={args.k}  |  knn={knn}")
    print("left-click=query  n=auto-range  c=colormap  s=save  q/ESC=quit")

    # --- UI state ---
    state = State(disp_w)
    qr, qc = disp_h // 2, disp_w // 2
    node_idx = _pixel_to_node(qr, qc, disp_h, disp_w, gh)
    sim = np.zeros(gh * gh, dtype=np.float32)
    save_count = 0
    fps_t = time.time()
    fps_display = 0.0
    auto_range = not args.fixed_range
    cmap_idx = 0

    win = "cheeger — spectral similarity  [click left panel to set query]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w * 2, disp_h)
    cv2.setMouseCallback(win, state.callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )

        if state.clicked:
            qr, qc = state.qr, state.qc
            node_idx = _pixel_to_node(qr, qc, disp_h, disp_w, gh)
            state.clicked = False

        # --- features ---
        if backbone is not None:
            X = _backbone_features(frame_rgb, gh, backbone, torch_device)
        else:
            X = _raw_features(frame_rgb, gh)
            if not args.no_diffuse:
                X = _diffuse(X, n_iters=args.n_iters)

        # --- spectral embed + cosine similarity ---
        phi = _embed(X, emb)
        sim = _cosine_sim(phi, node_idx)
        heatmap = _sim_to_heatmap(sim, gh, disp_h, disp_w,
                                  auto_range=auto_range, cmap_idx=cmap_idx)

        # --- compose canvas ---
        left_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
        cv2.rectangle(left_bgr, (qc - 6, qr - 6), (qc + 6, qr + 6), (0, 255, 255), 2)

        now = time.time()
        fps_display = 0.8 * fps_display + 0.2 / max(now - fps_t, 1e-4)
        fps_t = now
        cv2.putText(left_bgr, f"{fps_display:.1f} fps", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(left_bgr, f"{mode}  |  {'auto' if auto_range else 'fixed'}",
                    (8, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(win, np.concatenate([left_bgr, heatmap], axis=1))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            auto_range = not auto_range
            print(f"auto-range: {'on' if auto_range else 'off'}")
        if key == ord("c"):
            cmap_idx = (cmap_idx + 1) % len(_COLORMAPS)
        if key == ord("s"):
            out = pathlib.Path("results") / f"live_sim_{save_count}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), np.concatenate([left_bgr, heatmap], axis=1))
            print(f"saved {out}")
            save_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
