"""Network architectures.

Planned contents:
  * ``unet.py``  — encoder/decoder U-Net backbone (ResNet encoder optional).
  * ``heads.py`` — segmentation heads:
        - ConvHead          (1x1 conv baseline we benchmark against)
        - SpectralSegHead   (our differentiable spectral embedding head)

The head is the swappable component: same backbone, different head, controlled
comparison. Kept empty until the spectral embedding module lands.
"""
