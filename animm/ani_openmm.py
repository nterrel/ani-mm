"""OpenMM integration helpers for adding an ANI potential via TorchForce.

Requires the openmm-torch plugin (``conda install -c conda-forge openmm-torch``).
Python import name historically was ``openmmtorch`` (still used upstream); some
build variants could expose ``openmm_torch``. We attempt both for robustness.

The helper builds a TorchScript module wrapping a TorchANI model so that
OpenMM can obtain energies (and forces via autograd) each integration step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch

_torchforce_import_error = None
TorchForce = None  # type: ignore
for _mod_name in ("openmmtorch", "openmm_torch"):
    if TorchForce is not None:
        break
    try:  # pragma: no cover - import guard
        # type: ignore[attr-defined]
        TorchForce = __import__(_mod_name, fromlist=["TorchForce"]).TorchForce
    except Exception as exc:  # store last error
        _torchforce_import_error = exc
        continue


_TRACED_CACHE: Dict[Tuple[str, int, str], Any] = {}


class ANIPotentialModule(torch.nn.Module):  # pragma: no cover - executed inside OpenMM
    HARTREE_TO_KJMOL = 2625.499638

    def __init__(self, ani_model: torch.nn.Module, species: torch.Tensor):
        super().__init__()
        self.ani_model = ani_model
        # species: shape (1, N)
        self.register_buffer("species", species.long())

    def forward(self, positions_nm: torch.Tensor) -> torch.Tensor:
        # positions_nm shape (N, 3) in nm (OpenMM convention). Cast to model dtype.
        # type: ignore[stop-iteration]
        model_dtype = next(self.ani_model.parameters()).dtype
        pos_ang = positions_nm.to(model_dtype).unsqueeze(
            0) * 10.0  # (1, N, 3) Å
        out = self.ani_model((self.species, pos_ang))
        # TorchANI returns (energies) or object with energies
        if hasattr(out, "energies"):
            energy_ha = out.energies
        elif isinstance(out, (tuple, list)):
            energy_ha = out[0]
        else:
            energy_ha = out
        energy_kjmol = energy_ha * self.HARTREE_TO_KJMOL
        return energy_kjmol.sum()


def _species_atomic_numbers(topology) -> torch.Tensor:
    """Return (1, N) tensor of atomic numbers for the given topology.

    TorchANI models (e.g., ANI2x/ANI2dr) expect atomic numbers, not a local
    reindexed species mapping. Using atomic numbers avoids element order
    assumptions and supports any subset the model was trained on.
    """
    nums = []
    for atom in topology.atoms():
        element = atom.element
        if element is None or getattr(element, "atomic_number", None) is None:  # pragma: no cover
            raise ValueError(
                f"Atom '{atom.name}' missing atomic number; cannot build species tensor"
            )
        nums.append(int(element.atomic_number))
    return torch.tensor([nums], dtype=torch.long)


def build_ani_torch_force(
    topology,
    model_name: str = "ANI2DR",
    dtype: str = "float64",  # default to double precision for maximum accuracy
    threads: int | None = None,
    cache: bool = True,
):
    """Create a TorchForce for an ANI model for the given OpenMM Topology.

    Parameters
    ----------
    topology : openmm.app.Topology
        System topology whose atomic ordering must match the Simulation.
    model_name : str
        Name of ANI pretrained model (currently only 'ANI2x').
    dtype : str
        'float64' (default for numerical fidelity) or 'float32' (faster).
    threads : int | None
        If provided, sets torch.set_num_threads for this module build.
    """
    if TorchForce is None:  # pragma: no cover
        raise ImportError(
            "openmm-torch plugin not importable (tried modules 'openmmtorch' and 'openmm_torch'). "
            "Install or reinstall via: conda install -c conda-forge openmm-torch"
        ) from _torchforce_import_error

    from .ani import get_raw_ani_model

    if threads is not None:  # pragma: no cover - environment dependent
        torch.set_num_threads(int(threads))

    ani_model = get_raw_ani_model(model_name)
    requested_dtype = dtype
    if requested_dtype == "float64":  # pragma: no cover
        ani_model = ani_model.double()
    else:
        ani_model = ani_model.float()

    species = _species_atomic_numbers(topology)

    module = ANIPotentialModule(ani_model, species)
    n_atoms = species.shape[1]
    key = (model_name.upper(), n_atoms, dtype)
    if cache and key in _TRACED_CACHE:
        traced = _TRACED_CACHE[key]
    else:
        example = torch.zeros(
            (n_atoms, 3), dtype=getattr(torch, requested_dtype))
        try:
            with torch.no_grad():  # tracing only
                traced = torch.jit.trace(module, example)
        except RuntimeError as exc:  # fallback on dtype mismatch
            msg = str(exc)
            if "scalar type" in msg and requested_dtype == "float64":
                logging.warning(
                    "TorchScript trace failed in double precision; falling back to float32 for TorchForce (detail: %s)",
                    msg.splitlines()[0],
                )
                # Rebuild model & module in float32
                ani_model_fp32 = get_raw_ani_model(model_name).float()
                module_fp32 = ANIPotentialModule(ani_model_fp32, species)
                example_fp32 = torch.zeros((n_atoms, 3), dtype=torch.float32)
                with torch.no_grad():
                    traced = torch.jit.trace(module_fp32, example_fp32)
                key = (model_name.upper(), n_atoms, "float32")
            else:
                raise
        if cache:
            _TRACED_CACHE[key] = traced

    return TorchForce(traced)


def clear_traced_cache():  # pragma: no cover
    _TRACED_CACHE.clear()


__all__ = ["build_ani_torch_force", "clear_traced_cache"]
