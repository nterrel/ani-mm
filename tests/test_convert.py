import pytest

from animm.convert import smiles_to_ase


def test_smiles_to_ase_basic():
    atoms = smiles_to_ase("CCO")  # ethanol with hydrogens
    # C2H6O minimally 9 atoms; with added hydrogens more
    assert len(atoms) >= 9
    positions = atoms.get_positions()
    assert positions.shape[0] == len(atoms)
    assert positions.shape[1] == 3


def test_smiles_invalid():
    with pytest.raises(ValueError):
        smiles_to_ase("not_a_smiles")
