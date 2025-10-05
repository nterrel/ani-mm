"""Simple OpenMM minimization + MD runner integrating ANI forces (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:  # pragma: no cover - optional dependency placeholder
    openmm = None  # type: ignore
    app = None  # type: ignore
    unit = None  # type: ignore

from ase import Atoms


@dataclass
class MDState:
    positions: np.ndarray  # (T, N, 3)
    velocities: np.ndarray  # (T, N, 3)
    time: np.ndarray  # (T,)


def minimize_and_md(
    ase_atoms: Atoms, ani_model, n_steps: int = 1000, temperature: float = 300.0, dt_fs: float = 0.5
) -> MDState:
    """Placeholder function: at this stage just returns initial state replicated.

    A future implementation will build an OpenMM System that queries ANI for forces.
    """
    positions = np.repeat(np.expand_dims(ase_atoms.get_positions(), 0), repeats=2, axis=0)
    velocities = np.zeros_like(positions)
    time = np.array([0.0, n_steps * dt_fs * 1e-3])
    return MDState(positions=positions, velocities=velocities, time=time)
