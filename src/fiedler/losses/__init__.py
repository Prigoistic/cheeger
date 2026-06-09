"""Loss functions.

Planned contents:
  * ``spectral_loss.py`` — Rayleigh-quotient spectral-consistency regulariser
        Rq = vᵀ L v / vᵀ v  (Novelty #5): a CRF-free global smoothness prior used
        alongside cross-entropy. Encourages structurally connected pixels to share
        labels without explicit pairwise CRF inference.
"""
