import pytest

from animm.ani import get_raw_ani_model, list_available_ani_models


def test_list_available_models():
    models = list_available_ani_models()
    assert all(name.isupper() for name in models)
    # At least one expected model present
    assert any(m in ("ANI2DR", "ANI2X") for m in models)


def test_get_raw_model_case_insensitive():
    m1 = get_raw_ani_model("ani2dr")
    m2 = get_raw_ani_model("ANI2DR")
    assert type(m1) is type(m2)
