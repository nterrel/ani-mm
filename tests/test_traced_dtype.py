import pytest

from animm.ani_openmm import (  # type: ignore
    _TRACED_CACHE,
    build_ani_torch_force,
    clear_traced_cache,
)

try:
    import openmm  # noqa: F401
    import openmm.app as app  # type: ignore
except Exception:  # pragma: no cover
    openmm = None  # type: ignore
    app = None  # type: ignore


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_traced_dtype_attribute():
    pdb_str = """\nATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\nATOM      2  H1  MOL A   1       0.629   0.629   0.629  1.00  0.00           H\nATOM      3  H2  MOL A   1      -0.629  -0.629   0.629  1.00  0.00           H\nATOM      4  H3  MOL A   1       0.629  -0.629  -0.629  1.00  0.00           H\nATOM      5  H4  MOL A   1      -0.629   0.629  -0.629  1.00  0.00           H\nTER\nEND\n"""
    from io import StringIO

    assert app is not None
    pdb = app.PDBFile(StringIO(pdb_str))
    clear_traced_cache()
    try:
        tf = build_ani_torch_force(pdb.topology, model_name="ANI2DR", dtype="float64", cache=True)
    except ImportError:
        pytest.skip("TorchForce unavailable")
    traced_dtype = getattr(tf, "_animm_traced_dtype", None)
    assert traced_dtype in ("float64", "float32")
    # Cache should contain an entry reflecting that dtype
    assert any(key[-1] == traced_dtype for key in _TRACED_CACHE.keys())
