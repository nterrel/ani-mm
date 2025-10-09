"""ani-mm: TorchANI + OpenMM utilities.

Focused scope: fast ANI evaluation and small MD examples with minimal ceremony.
Select warning/namespace hygiene:
* Silence deprecated ``simtk.openmm`` banner by pre-aliasing ``simtk.openmm`` to
    the modern ``openmm`` module before any plugin (e.g. ``openmmtorch``) tries to
    import the legacy path.
* Ignore a noisy pkg_resources deprecation warning triggered indirectly.

This keeps user‑facing CLI output clean without broad stderr redirection or
print monkeypatching.
"""

from __future__ import annotations

import sys
import types
import warnings


def _alias_simtk_openmm() -> None:
    """Alias ``simtk.openmm`` to ``openmm`` before any legacy import.

    Some plugins (e.g. older ``openmmtorch`` builds) still import
    ``simtk.openmm`` which triggers a one-line deprecation banner. By
    inserting a synthetic module mapping first, the legacy package's
    ``__init__`` (which prints) is never executed.
    """
    if "simtk.openmm" in sys.modules:  # already real or aliased
        return
    try:
        import openmm  # noqa: WPS433
    except ImportError:  # OpenMM genuinely missing; do nothing
        return
    simtk_pkg = types.ModuleType("simtk")
    openmm_alias = types.ModuleType("simtk.openmm")
    for name in dir(openmm):  # mirror public attrs
        try:
            setattr(openmm_alias, name, getattr(openmm, name))
        except Exception:  # pragma: no cover - defensive
            continue
    simtk_pkg.openmm = openmm_alias  # type: ignore[attr-defined]
    sys.modules["simtk"] = simtk_pkg
    sys.modules["simtk.openmm"] = openmm_alias


_alias_simtk_openmm()

# Suppress the specific pkg_resources deprecation warning triggered inside torchani
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API",
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
