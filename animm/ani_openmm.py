"""Glue for building an OpenMM ``TorchForce`` from a TorchANI model.

We trace the selected ANI network with TorchScript once (per model / atom count /
dtype) and reuse that graph inside OpenMM so forces are evaluated each step.

Quick facts:
* Needs the ``openmm-torch`` plugin (we try both import spellings).
* Prefers a float64 trace; retries in float32 only if necessary.
* Caches traces keyed by (MODEL, NATOMS, DTYPE) in‑process.
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
    """Return shape (1, N) tensor of atomic numbers.

    Using true atomic numbers (rather than an internal reindex) mirrors how
    modern TorchANI models expect species and reduces assumptions about element
    ordering.
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
    """Return a ``TorchForce`` for an ANI model & topology.

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology whose atom ordering must match the target ``Simulation``.
    model_name : str
        ANI model name (e.g. ``ANI2DR`` or ``ANI2X``).
    dtype : str
        Requested tracing precision (``float64`` default; falls back to ``float32``).
    threads : int | None
        Optional override for ``torch.set_num_threads`` during trace.
    cache : bool
        If True (default) reuse/store a traced module in the in‑process cache.
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
    log = logging.getLogger("animm.ani_openmm")
    cache_hit = False
    if cache and key in _TRACED_CACHE:
        traced = _TRACED_CACHE[key]
        cache_hit = True
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
        log.debug(
            "Traced ANI model=%s natoms=%d requested_dtype=%s final_dtype=%s cache_store=%s",
            model_name.upper(), n_atoms, requested_dtype, key[2], cache
        )
    if cache_hit:
        log.debug(
            "Cache hit ANI model=%s natoms=%d dtype=%s", model_name.upper(
            ), n_atoms, key[2]
        )
    tf = TorchForce(traced)
    # attach metadata so callers can inspect true traced dtype
    try:  # pragma: no cover - attribute assignment safety
        setattr(tf, "_animm_traced_dtype", key[2])
        setattr(tf, "_animm_cache_hit", cache_hit)
    except Exception:  # pragma: no cover
        pass
    log.debug(
        "Built TorchForce model=%s natoms=%d traced_dtype=%s cache_hit=%s",
        model_name.upper(), n_atoms, key[2], cache_hit
    )
    return tf


def clear_traced_cache():  # pragma: no cover
    _TRACED_CACHE.clear()


__all__ = ["build_ani_torch_force", "clear_traced_cache"]
