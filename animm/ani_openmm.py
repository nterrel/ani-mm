"""OpenMM integration helpers for adding an ANI potential via TorchForce.

Requires the openmm-torch plugin (``conda install -c conda-forge openmm-torch``).

The helper builds a TorchScript module wrapping a TorchANI model so that
OpenMM can obtain energies (and forces via autograd) each integration step.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Any

import torch

try:  # optional plugin
    from openmm_torch import TorchForce  # type: ignore
except Exception as exc:  # pragma: no cover - module load guard
    TorchForce = None  # type: ignore
    _torchforce_import_error = exc
else:
    _torchforce_import_error = None


# Order of species indices expected by ANI2x. (Subset typical for biomolecules.)
ANI2X_SPECIES_ORDER = ["H", "C", "N", "O", "F", "S", "Cl"]
_SYMBOL_TO_INDEX = {s: i for i, s in enumerate(ANI2X_SPECIES_ORDER)}
_TRACED_CACHE: Dict[Tuple[str, int, str], Any] = {}


class ANIPotentialModule(torch.nn.Module):  # pragma: no cover - executed inside OpenMM
    HARTREE_TO_KJMOL = 2625.499638

    def __init__(self, ani_model: torch.nn.Module, species: torch.Tensor):
        super().__init__()
        self.ani_model = ani_model
        # species: shape (1, N)
        self.register_buffer("species", species.long())

    def forward(self, positions_nm: torch.Tensor) -> torch.Tensor:
        # positions_nm shape (N, 3) in nm (OpenMM convention)
        pos_ang = positions_nm.unsqueeze(0) * 10.0  # (1, N, 3) Å
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


def _species_tensor(symbols: Sequence[str]) -> torch.Tensor:
    try:
        idxs = [_SYMBOL_TO_INDEX[s] for s in symbols]
    except KeyError as missing:
        raise ValueError(
            f"Element '{missing.args[0]}' not supported by ANI2x helper. Supported: {ANI2X_SPECIES_ORDER}"  # noqa: E501
        ) from None
    return torch.tensor([idxs], dtype=torch.long)


def build_ani_torch_force(
    topology,
    model_name: str = "ANI2DR",
    dtype: str = "float32",
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
        'float32' (default) or 'float64'. Float32 is faster.
    threads : int | None
        If provided, sets torch.set_num_threads for this module build.
    """
    if TorchForce is None:  # pragma: no cover
        raise ImportError(
            "openmm-torch not available. Install with: conda install -c conda-forge openmm-torch"
        ) from _torchforce_import_error

    from .ani import get_raw_ani_model

    if threads is not None:  # pragma: no cover - environment dependent
        torch.set_num_threads(int(threads))

    ani_model = get_raw_ani_model(model_name)
    if dtype == "float64":  # pragma: no cover
        ani_model = ani_model.double()
    else:
        ani_model = ani_model.float()

    symbols = [atom.element.symbol for atom in topology.atoms()]
    species = _species_tensor(symbols)

    module = ANIPotentialModule(ani_model, species)
    key = (model_name.upper(), len(symbols), dtype)
    if cache and key in _TRACED_CACHE:
        traced = _TRACED_CACHE[key]
    else:
        example = torch.zeros((len(symbols), 3), dtype=getattr(torch, dtype))
        with torch.no_grad():  # tracing only
            traced = torch.jit.trace(module, example)
        if cache:
            _TRACED_CACHE[key] = traced

    return TorchForce(traced)


def clear_traced_cache():  # pragma: no cover
    _TRACED_CACHE.clear()


__all__ = ["build_ani_torch_force", "clear_traced_cache"]
