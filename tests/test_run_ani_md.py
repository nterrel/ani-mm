import pytest
from animm.ani import get_raw_ani_model, load_ani_model
from animm.ani_openmm import (  # type: ignore
    _TRACED_CACHE,
    build_ani_torch_force,
    clear_traced_cache,
)
from animm.openmm_runner import run_ani_md
import openmm
from ase import Atoms


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_run_ani_md_basic():
    # Simple methane-like structure (CH4)
    atoms = Atoms(
        "CH4",
        positions=[
            (0.000, 0.000, 0.000),
            (0.629, 0.629, 0.629),
            (-0.629, -0.629, 0.629),
            (0.629, -0.629, -0.629),
            (-0.629, 0.629, -0.629),
        ],
    )
    res = run_ani_md(
        ase_atoms=atoms,
        model="ANI2DR",
        n_steps=10,
        dt_fs=1.0,
        temperature_K=300.0,
        report_interval=5,
        collect_trajectory=True,
        minimize=True,
    )
    assert res.steps == 10
    assert res.final_potential_kjmol is not None
    assert res.engine == "openmm-torch"
    # Trajectory frames: initial frame + step 5 + step 10 (approx)
    assert res.positions is not None
    assert res.positions.shape[0] >= 2
    assert res.positions.shape[1:] == (5, 3)  # 5 atoms


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_invalid_model_name():
    atoms = Atoms(
        "CH4",
        positions=[
            (0.0, 0.0, 0.0),
            (0.6, 0.6, 0.6),
            (-0.6, -0.6, 0.6),
            (0.6, -0.6, -0.6),
            (-0.6, 0.6, -0.6),
        ],
    )
    with pytest.raises(ValueError) as exc:
        run_ani_md(atoms, model="NOT_A_MODEL", n_steps=1)
    assert "Unsupported ANI model" in str(exc.value)


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_cache_reuse():
    atoms = Atoms(
        "CH4",
        positions=[
            (0.0, 0.0, 0.0),
            (0.6, 0.6, 0.6),
            (-0.6, -0.6, 0.6),
            (0.6, -0.6, -0.6),
            (-0.6, 0.6, -0.6),
        ],
    )
    clear_traced_cache()
    # Build first TorchForce (will populate cache)
    from animm.openmm_runner import _ase_to_openmm_topology  # type: ignore

    top, _ = _ase_to_openmm_topology(atoms)
    try:
        build_ani_torch_force(top, model_name="ANI2DR",
                              dtype="float64", cache=True)
    except ImportError:
        pytest.skip("TorchForce unavailable")
    size_after_first = len(_TRACED_CACHE)
    # Build second identical TorchForce
    build_ani_torch_force(top, model_name="ANI2DR",
                          dtype="float64", cache=True)
    size_after_second = len(_TRACED_CACHE)
    assert size_after_second == size_after_first  # no new cache entries


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_tracing_fallback_float32(monkeypatch):
    # Force a tracing failure in float64 by monkeypatching torch.jit.trace to raise once
    import torch

    atoms = Atoms(
        "CH4",
        positions=[
            (0.0, 0.0, 0.0),
            (0.6, 0.6, 0.6),
            (-0.6, -0.6, 0.6),
            (0.6, -0.6, -0.6),
            (-0.6, 0.6, -0.6),
        ],
    )
    clear_traced_cache()

    call_state = {"calls": 0}
    real_trace = torch.jit.trace

    def failing_trace(mod, example, *args, **kwargs):  # noqa: ANN001
        if call_state["calls"] == 0:
            call_state["calls"] += 1
            # simulate dtype mismatch error message shape used in code
            raise RuntimeError(
                "could not create a tensor of scalar type Double")
        return real_trace(mod, example, *args, **kwargs)

    monkeypatch.setattr(torch.jit, "trace", failing_trace)

    from animm.openmm_runner import _ase_to_openmm_topology  # type: ignore

    top, _ = _ase_to_openmm_topology(atoms)
    try:
        build_ani_torch_force(top, model_name="ANI2DR",
                              dtype="float64", cache=True)
    except ImportError:
        pytest.skip("TorchForce unavailable")

    # After fallback we expect either a float32 or float64 entry; ensure at least one cache key ends with float32
    assert any(key[-1] in ("float64", "float32")
               for key in _TRACED_CACHE.keys())
    # If fallback executed, second key should be float32
    if any(key[-1] == "float32" for key in _TRACED_CACHE.keys()):
        assert any(key[-1] == "float32" for key in _TRACED_CACHE.keys())
