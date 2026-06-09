# notebooks/

Thin Jupyter/Colab **drivers** — no library code lives here. A notebook should
clone the repo, install the package, and call into `fiedler`:

```python
!git clone <repo-url> spec && cd spec && pip install -e ".[data]"
from fiedler.engine import Trainer        # all logic stays in the package
```

Planned:
- `colab_train_T4.ipynb` — end-to-end training-loop smoke test on a Cityscapes
  subset using a free Colab T4 (16 GB). Proves install + import + one train epoch
  before moving heavy runs to a persistent AWS/Azure GPU.
