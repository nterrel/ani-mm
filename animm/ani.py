"""ANI model loading and energy/force calculation utilities.

Supported model names (case-insensitive):
    ANI2DR (default), ANI2X, ANI2XPERIODIC (if available in torchani build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torchani


@dataclass
class ANIEvaluation:
    energy: torch.Tensor  # shape: (1,) energy in Hartree
    forces: torch.Tensor  # shape: (N, 3) forces in Hartree/Bohr


def _load_ani2dr():  # pragma: no cover
    return torchani.models.ANI2dr()


def _load_ani2x():  # pragma: no cover
    return torchani.models.ANI2x()


def _load_ani2xperiodic():  # pragma: no cover
    if not hasattr(torchani.models, "ANI2xPeriodic"):
        raise ValueError("ANI2xPeriodic not available in this torchani build")
    return torchani.models.ANI2xPeriodic()


MODEL_LOADERS: Dict[str, Callable[[], torch.nn.Module]] = {
    "ANI2DR": _load_ani2dr,
    "ANI2X": _load_ani2x,
    "ANI2XPERIODIC": _load_ani2xperiodic,
}

DEFAULT_MODEL = "ANI2DR"


def get_raw_ani_model(model_name: str = DEFAULT_MODEL) -> torch.nn.Module:
    key = model_name.upper()
    if key not in MODEL_LOADERS:
        raise ValueError(
            f"Unsupported ANI model '{model_name}'. Supported: {', '.join(MODEL_LOADERS)}"
        )
    return MODEL_LOADERS[key]()


def load_ani_model(model_name: str = DEFAULT_MODEL):
    return get_raw_ani_model(model_name).ase()


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
