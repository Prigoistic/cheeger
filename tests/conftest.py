"""Make `import fiedler` work in tests even before `pip install -e .`."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
import torch


@pytest.fixture(autouse=True)
def _default_dtype(request):
    """Per-test default dtype isolation: float64 everywhere, float32 only for tests
    marked ``@pytest.mark.float32`` (the model/training tests). Restores the global
    afterwards so one module's dtype choice never leaks into another's assertions."""
    saved = torch.get_default_dtype()
    want = torch.float32 if request.node.get_closest_marker("float32") else torch.float64
    torch.set_default_dtype(want)
    yield
    torch.set_default_dtype(saved)
