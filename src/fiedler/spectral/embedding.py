"""SpectralEmbedding — the end-to-end differentiable module.

features (N, C)  ->  learned affinity graph  ->  L_sym  ->  bottom-k eigenpairs
                 ->  Φ = U · diag(h_θ(λ))  ->  spectral coordinates (N, k)

Every stage is differentiable, so gradients from a downstream segmentation loss
flow back into:
  * the eigenvectors (via the broadened backward),
  * the affinity bandwidth σ and the learned metric (Novelty #1),
  * the spectral response h_θ (the theory).

This is the object that becomes a U-Net head in Phase 3. It is deliberately
backbone-agnostic: hand it any per-node feature tensor.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..graph.laplacian import gaussian_affinity, laplacian
from .diffeig import smallest_k
from .response import SpectralResponse


class SpectralEmbedding(nn.Module):
    def __init__(
        self,
        k: int = 16,
        laplacian_kind: str = "sym",
        knn: int | None = None,
        sigma_init: float = 1.0,
        learn_sigma: bool = True,
        metric_dim: int | None = None,
        response: SpectralResponse | None = None,
        eig_eps: float = 1e-6,
    ):
        super().__init__()
        self.k = k
        self.kind = laplacian_kind
        self.knn = knn
        self.eig_eps = eig_eps

        log_sigma = torch.log(torch.tensor(float(sigma_init)))
        if learn_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

        # Novelty #1: a learned diagonal metric on feature channels
        if metric_dim is not None:
            self.metric = nn.Parameter(torch.ones(metric_dim))
        else:
            self.metric = None

        self.response = response if response is not None else SpectralResponse()

    @property
    def sigma(self) -> Tensor:
        return torch.exp(self.log_sigma)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """Returns a dict: phi (N,k) embedding, eigvals (k,), eigvecs (N,k),
        response (k,) — downstream losses consume eigvals/eigvecs directly."""
        metric = self.metric.abs() if self.metric is not None else None  # keep PSD
        W = gaussian_affinity(features, sigma=self.sigma, k=self.knn, metric=metric)
        L = laplacian(W, kind=self.kind)
        eigvals, eigvecs = smallest_k(L, self.k, eps=self.eig_eps)
        h = self.response(eigvals)
        phi = eigvecs * h.unsqueeze(0)
        return {"phi": phi, "eigvals": eigvals, "eigvecs": eigvecs, "response": h}
