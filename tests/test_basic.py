import pytest

try:
    from animm.ani import load_ani_model
except Exception:  # pragma: no cover - torchani optional
    pytest.skip("torchani not available", allow_module_level=True)


def test_dummy():
    assert callable(load_ani_model)
