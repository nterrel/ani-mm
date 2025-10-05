"""ani-mm: Hybrid ANI + OpenMM molecular modelling utilities.

Public API is experimental and may change until 0.2.0.

This package applies a focused warnings filter to silence the verbose
`pkg_resources` deprecation notice that PyTorchANI currently emits at import
time. Once torchani migrates off `pkg_resources`, this filter can be removed.
"""

from __future__ import annotations

import warnings

# Suppress the specific pkg_resources deprecation warning triggered inside torchani
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
)

from .ani import (  # noqa: F401
    ani_energy_forces,
    load_ani_model,
    list_available_ani_models,
    get_raw_ani_model,
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
