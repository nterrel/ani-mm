"""ANI model loading and energy/force calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchani


@dataclass
class ANIEvaluation:
    energy: torch.Tensor  # shape: (1,) energy in Hartree
    forces: torch.Tensor  # shape: (N, 3) forces in Hartree/Bohr


def load_ani_model(model_name: str = "ANI2x"):
    """Load a TorchANI pretrained model.

    Parameters
    ----------
    model_name: str
        One of the pretrained model identifiers (e.g., 'ANI2x').
    """
    if model_name.upper() == "ANI2dr":
        return torchani.models.ANI2dr().ase()
    raise ValueError(f"Unsupported ANI model: {model_name}")


def ani_energy_forces(ani_model, ase_atoms) -> ANIEvaluation:
    """Compute energy and forces using ANI for an ASE Atoms object.

    Returns energy (Hartree) and forces (Hartree/Bohr).
    """
    # TorchANI ASE interface returns energy in Hartree and forces in Hartree/Bohr
    energy = ani_model.get_potential_energy(ase_atoms, include_forces=True)
    # The ase() wrapper stores last results in model.atoms.calc.results
    forces = ani_model.atoms.calc.results["forces"]  # type: ignore[index]
    return ANIEvaluation(
        energy=torch.tensor([energy], dtype=torch.float64),
        forces=torch.tensor(forces, dtype=torch.float64),
    )
