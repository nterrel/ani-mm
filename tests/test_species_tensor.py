import pytest

from animm.ani_openmm import build_ani_torch_force, clear_traced_cache

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except Exception:  # pragma: no cover
    openmm = None  # type: ignore
    app = None  # type: ignore
    unit = None  # type: ignore


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_species_tensor_uses_atomic_numbers():
    # Minimal methane-like topology (CH4) using an inline PDB
    pdb_str = """\nATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\nATOM      2  H1  MOL A   1       0.629   0.629   0.629  1.00  0.00           H\nATOM      3  H2  MOL A   1      -0.629  -0.629   0.629  1.00  0.00           H\nATOM      4  H3  MOL A   1       0.629  -0.629  -0.629  1.00  0.00           H\nATOM      5  H4  MOL A   1      -0.629   0.629  -0.629  1.00  0.00           H\nTER\nEND\n"""
    from io import StringIO

    assert app is not None
    pdb = app.PDBFile(StringIO(pdb_str))

    # Build force; this will trigger species tensor generation. We don't need full simulation.
    try:
        _ = build_ani_torch_force(
            pdb.topology, model_name="ANI2DR", dtype="float64", cache=False
        )  # noqa: F841
    except ImportError:
        pytest.skip(
            "TorchForce / openmm-torch not available in this test environment")

    # Indirect validation: traced module stored in cache contains species buffer of atomic numbers.
    # Pull a traced module from internal cache if present.
    from animm.ani_openmm import _TRACED_CACHE  # type: ignore

    for key, traced in _TRACED_CACHE.items():
        if key[0] == "ANI2DR" and key[1] == 5:  # (model, n_atoms)
            # Extract species from the scripted module's original module (original attribute names kept)
            # TorchScript stores buffers as attributes.
            try:
                # type: ignore[attr-defined]
                species = traced.original_module.species
            except AttributeError:
                continue
            species_list = species.squeeze(0).tolist()
            # Expect atomic numbers: Carbon=6, Hydrogen=1
            assert set(species_list) == {6, 1}
            assert species_list.count(6) == 1
            assert species_list.count(1) == 4
            break
    else:
        pytest.skip(
            "Traced module not cached; unable to inspect species tensor")

    clear_traced_cache()
