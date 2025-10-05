"""ani-mm: TorchANI + OpenMM utilities.

Focused scope: fast ANI evaluation and small MD examples with minimal ceremony.
Warning filters reduce known third‑party noise (deprecated pkg_resources,
legacy ``simtk.openmm`` bridge message). Adjust or remove these filters locally
if you need full visibility for debugging.
"""

from __future__ import annotations

import warnings

# Suppress the specific pkg_resources deprecation warning triggered inside torchani
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Warning: importing 'simtk.openmm' is deprecated",
    category=UserWarning,
)

from .ani import (  # noqa: F401
    ani_energy_forces,
    get_raw_ani_model,
    list_available_ani_models,
    load_ani_model,
)
from .convert import smiles_to_ase  # noqa: F401
from .openmm_runner import minimize_and_md  # noqa: F401
from .version import __version__  # noqa: F401

__all__ = [
    "ani_energy_forces",
    "load_ani_model",
    "list_available_ani_models",
    "get_raw_ani_model",
    "smiles_to_ase",
    "minimize_and_md",
    "__version__",
]
