import pytest

from animm.ani import load_ani_model, ani_energy_forces
from animm.convert import smiles_to_ase


def test_ani_energy_forces_ethanol():
    atoms = smiles_to_ase("CCO")
    model = load_ani_model("ANI2DR")
    eval_res = ani_energy_forces(model, atoms)
    # Basic shape / type checks
    assert eval_res.energy.shape == (1,)
    assert eval_res.forces.shape[1] == 3
    assert eval_res.forces.shape[0] == len(atoms)
    # Energy should be finite
    assert float(eval_res.energy.item()) == pytest.approx(
        eval_res.energy.item())
