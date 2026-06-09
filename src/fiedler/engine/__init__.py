"""Training / evaluation orchestration.

Planned contents:
  * ``config.py``  — typed experiment config (dataclass; serialisable to configs/*.yaml).
  * ``trainer.py`` — device-agnostic train/eval loop, checkpointing, logging.

The trainer is backend-blind: the same loop runs on MPS locally, a T4 on Colab,
or an A100 on AWS — device comes from fiedler.utils.get_device().
"""
