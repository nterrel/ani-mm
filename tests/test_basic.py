import pytest
from animm.ani import load_ani_model


def test_dummy():
    assert callable(load_ani_model)
