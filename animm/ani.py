"""ANI model loading and energy/force calculation utilities.

Supported model names (case-insensitive):
    ANI2DR (default), ANI2X, ANI2XPERIODIC (if available in torchani build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

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


MODEL_LOADERS: Dict[str, Callable[[], torch.nn.Module]] = {
    "ANI2DR": _load_ani2dr,
    "ANI2X": _load_ani2x,
}

DEFAULT_MODEL = "ANI2DR"


def get_raw_ani_model(model_name: str = DEFAULT_MODEL) -> torch.nn.Module:
    key = model_name.strip().upper()
    if key not in MODEL_LOADERS:
        raise ValueError(
            f"Unsupported ANI model '{model_name}'. Supported: {', '.join(MODEL_LOADERS.keys())}"
        )
    return MODEL_LOADERS[key]()


def load_ani_model(model_name: str = DEFAULT_MODEL):
    # TorchANI model exposes .ase() to build an ASE-compatible wrapper
    return get_raw_ani_model(model_name).ase()  # type: ignore[attr-defined]


def list_available_ani_models() -> List[str]:  # pragma: no cover - simple probe
    available: List[str] = []
    for name in MODEL_LOADERS:
        try:
            MODEL_LOADERS[name]()
        except Exception:
            continue
        else:
            available.append(name)
    return available


def ani_energy_forces(ani_model, ase_atoms) -> ANIEvaluation:
    """Compute energy and forces using ANI for an ASE Atoms object.

    Returns energy (Hartree) and forces (Hartree/Bohr).
    """
    # TorchANI ASE wrapper exposes both get_potential_energy(atoms) and get_forces(atoms)
    # returning energy in Hartree and forces in Hartree/Bohr.
    energy = ani_model.get_potential_energy(ase_atoms)
    try:
        forces = ani_model.get_forces(ase_atoms)
    except Exception as exc:  # pragma: no cover - unexpected API change
        raise RuntimeError(
            "Unable to retrieve forces via TorchANI ASE interface"
        ) from exc
    return ANIEvaluation(
        energy=torch.tensor([energy], dtype=torch.float64),
        forces=torch.tensor(forces, dtype=torch.float64),
    )


__all__ = [
    "ANIEvaluation",
    "load_ani_model",
    "get_raw_ani_model",
    "list_available_ani_models",
    "ani_energy_forces",
]
