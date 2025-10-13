import pytest
import openmm
import openmm.app as app
from openmm.app import element as elem
from animm.ani_openmm import build_ani_torch_force, clear_traced_cache
from typing import Any, Optional, Tuple


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_species_tensor_uses_atomic_numbers():
    # Start from a clean cache
    clear_traced_cache()

    # Build a minimal methane-like topology programmatically (CH4)
    top = app.Topology()
    chain = top.addChain("A")
    res = top.addResidue("MOL", chain)
    top.addAtom("C", elem.carbon, res)
    top.addAtom("H1", elem.hydrogen, res)
    top.addAtom("H2", elem.hydrogen, res)
    top.addAtom("H3", elem.hydrogen, res)
    top.addAtom("H4", elem.hydrogen, res)

    # Build the ANI force; enable cache so the traced module is stored
    try:
        tf = build_ani_torch_force(
            top, model_name="ANI2DR", dtype="float64", cache=True
        )
    except ImportError:
        pytest.skip(
            "TorchForce / openmm-torch not available in this test environment")

    # Prefer reading the attached species metadata from the returned TorchForce
    species_list = getattr(tf, "_animm_species", None)
    if not species_list:
        # Fallback: Inspect the traced module from the internal cache
        from animm.ani_openmm import _TRACED_CACHE  # type: ignore

        for key, traced in _TRACED_CACHE.items():
            if key[0] == "ANI2DR" and key[1] == 5:
                try:
                    # type: ignore[attr-defined]
                    species = traced.original_module.species
                except AttributeError:
                    continue
                species_list = species.squeeze(0).tolist()
                break
        if not species_list:
            pytest.fail(
                "Could not determine species from TorchForce or cache for (ANI2DR, 5)")

    # Expect atomic numbers: Carbon=6, Hydrogen=1
    assert set(species_list) == {6, 1}
    assert species_list.count(6) == 1
    assert species_list.count(1) == 4

    clear_traced_cache()
