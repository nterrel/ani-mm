import warnings

import pytest


# Global warning filters to reduce noise in test output
@pytest.fixture(autouse=True)
def _filter_warnings():
    warnings.filterwarnings("ignore", message=r"ANI-2xr is experimental")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"openmm")
    # Suppress noisy Torch tracing / TracerWarnings produced during TorchScript trace
    try:  # pragma: no cover
        import torch.jit  # noqa: WPS433
        from torch.jit import TracerWarning  # type: ignore

        warnings.filterwarnings("ignore", category=TracerWarning)
    except Exception:  # pragma: no cover - if torch layout changes
        warnings.filterwarnings(
            "ignore",
            message=r"Converting a tensor to a Python (boolean|integer) might cause the trace",
        )
    # Generic shape assertion tracer noise
    warnings.filterwarnings(
        "ignore",
        message=r"Converting a tensor to a Python boolean might cause the trace",
    )
    yield
