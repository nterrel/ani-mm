import warnings

import pytest


# Global warning filters to reduce noise in test output
@pytest.fixture(autouse=True)
def _filter_warnings():
    warnings.filterwarnings("ignore", message=r"ANI-2xr is experimental")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module=r"openmm")
    # Suppress noisy Torch tracing / TracerWarnings produced during TorchScript trace
    try:
        import torch.jit
        from torch.jit import TracerWarning  # type: ignore

        warnings.filterwarnings("ignore", category=TracerWarning)
    except Exception:
        warnings.filterwarnings(
            "ignore",
            message=r"Converting a tensor to a Python (boolean|integer) might cause the trace",
        )
    yield
