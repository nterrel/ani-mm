"""Lightweight OpenMM Langevin MD loop with ANI forces.

Single responsibility: build a system (one ``TorchForce``) and integrate.
Optional: trajectory capture, DCD output, live viewer integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import List, Optional

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

from .ani_openmm import build_ani_torch_force

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MDState:
    """Legacy two‑frame (initial/final) container."""

    positions: np.ndarray  # (T, N, 3)
    velocities: np.ndarray  # (T, N, 3)
    time: np.ndarray  # (T,)


@dataclass
class MDResult:
    """Result bundle (energies in kJ/mol, times ps, positions Å)."""

    final_potential_kjmol: float
    final_temperature_K: Optional[float]
    steps: int
    model: str
    dtype: str
    engine: str
    positions: Optional[np.ndarray] = None
    times_ps: Optional[np.ndarray] = None
    dcd_path: Optional[str] = None
    reporter_outputs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ase_to_openmm_topology(ase_atoms: Atoms):
    """Create a minimal (non‑bonded) OpenMM topology + positions (nm).

    Bonds are omitted (ANI does not use them). Periodic vectors are assigned
    only if any cell axis is non‑zero.
    """
    if app is None or unit is None:
        raise ImportError(
            "OpenMM is required to build a topology from ASE atoms")
    top = app.Topology()
    chain = top.addChain()
    residue = top.addResidue("MOL", chain)
    element_mod = app.element
    for sym in ase_atoms.get_chemical_symbols():
        el = element_mod.Element.getBySymbol(sym)
        top.addAtom(sym, el, residue)
    # Positions: ASE in Å -> convert to nm
    pos_ang = ase_atoms.get_positions()  # Å
    pos_nm = pos_ang * 0.1
    positions = [openmm.Vec3(*xyz)
                 for xyz in pos_nm]  # type: ignore[attr-defined]
    box = ase_atoms.get_cell()  # (3,3) in Å (may be all zeros for non-periodic)
    lengths = None
    if hasattr(box, "lengths"):
        try:  # pragma: no cover - defensive
            lengths = box.lengths()
        except Exception:
            lengths = None
    if lengths is not None:
        import numpy as _np

        arr = _np.asarray(lengths, dtype=float)
        # Treat as periodic only if any dimension is meaningfully non-zero (> 1e-6 Å)
        if _np.any(arr > 1e-6):
            a, b, c = arr.tolist()
            top.setPeriodicBoxVectors(
                openmm.Vec3(a * 0.1, 0, 0),  # type: ignore[attr-defined]
                openmm.Vec3(0, b * 0.1, 0),  # type: ignore[attr-defined]
                openmm.Vec3(0, 0, c * 0.1),  # type: ignore[attr-defined]
            )
    return top, positions


class _InMemoryReporter:
    """Collect positions in memory every ``interval`` steps."""

    def __init__(self, interval: int):
        self._interval = int(interval)
        self.frames: List[np.ndarray] = []
        self.times_ps: List[float] = []

    def describeNextReport(self, simulation):  # noqa: N802 (OpenMM API naming)
        # Return standard 5-tuple: (nextStep, needPositions, needVelocities, needForces, needEnergy)
        return (self._interval, True, False, False, False)

    def report(self, simulation, state):  # noqa: N802
        # Positions in nm -> convert to Å for user friendliness
        pos = (
            state.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer) * 10.0
        )  # type: ignore[attr-defined]
        self.frames.append(np.array(pos, dtype=float))
        self.times_ps.append(
            state.getTime().value_in_unit(unit.picosecond)
        )  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ani_md(
    ase_atoms: Atoms,
    model: str = "ANI2DR",
    n_steps: int = 1000,
    temperature_K: float = 300.0,
    friction_per_ps: float = 1.0,
    dt_fs: float = 1.0,
    platform: str | None = None,
    minimize: bool = True,
    report_interval: int = 100,
    dcd_path: str | None = None,
    collect_trajectory: bool = False,
    dtype: str = "float64",
    ani_threads: int | None = None,
    seed: int | None = None,
    live_view: bool = False,
    live_interval: int | None = None,
    hold_open: bool = False,
) -> MDResult:
    """Run Langevin MD with ANI forces.

    Parameters
    ----------
    ase_atoms : Atoms
        Input coordinates (Å) and species.
    model : str
        ANI model name (case-insensitive, e.g. ANI2DR, ANI2X).
    n_steps : int
        Number of integration steps.
    temperature_K : float
        Target thermostat temperature (Kelvin).
    friction_per_ps : float
        Langevin friction (1/ps).
    dt_fs : float
        Time step (fs).
    platform : str | None
        OpenMM platform name (e.g. CUDA, CPU). Auto if None.
    minimize : bool
        Whether to perform energy minimization before dynamics.
    report_interval : int
        Interval (steps) for reporters and optional trajectory capture.
    dcd_path : str | None
        If provided, write a DCD trajectory at the report interval.
    collect_trajectory : bool
        If True, store positions (Å) in-memory at each report interval (including initial frame).
    dtype : str
        Torch dtype for tracing ('float64' default, falls back internally if unsupported).
    ani_threads : int | None
        Override torch thread count for force evaluation.
    seed : int | None
        Random seed for integrator RNG.
    live_view : bool
        If True, open a lightweight desktop scatter plot window that updates
        every ``live_interval`` steps (matplotlib required). Falls back
        gracefully if matplotlib or a GUI backend is unavailable.
    live_interval : int | None
        Interval for live view updates. Defaults to ``report_interval`` if
        not provided.
    hold_open : bool
        Keep Matplotlib viewer window open after completion.
    """
    log = logging.getLogger("animm.md")
    if openmm is None or app is None or unit is None:
        raise ImportError("OpenMM (and openmm-torch) required for run_ani_md")

    topology, positions_nm = _ase_to_openmm_topology(ase_atoms)

    # Build System: add particles with proper masses
    system = openmm.System()
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)

    # Attach ANI TorchForce
    ani_force = build_ani_torch_force(
        topology=topology, model_name=model, dtype=dtype, threads=ani_threads
    )
    log.debug(
        "Attached ANI TorchForce model=%s natoms=%d requested_dtype=%s traced_dtype=%s",
        model.upper(), topology.getNumAtoms(), dtype, getattr(
            ani_force, "_animm_traced_dtype", dtype)
    )
    system.addForce(ani_force)

    # Integrator & Simulation
    dt_ps = dt_fs * 1e-3
    integrator = openmm.LangevinIntegrator(  # type: ignore[attr-defined]
        temperature_K * unit.kelvin,  # type: ignore[attr-defined]
        friction_per_ps / unit.picosecond,  # type: ignore[attr-defined]
        dt_ps * unit.picosecond,  # type: ignore[attr-defined]
    )
    if seed is not None:
        try:  # pragma: no cover - integrator attribute differences
            integrator.setRandomNumberSeed(int(seed))
        except Exception:
            pass
    use_platform = None
    if platform:
        use_platform = openmm.Platform.getPlatformByName(platform)
    sim = app.Simulation(topology, system, integrator, use_platform)
    sim.context.setPositions(positions_nm)

    # Minimization
    if minimize:
        log.debug("Starting energy minimization")
        sim.minimizeEnergy()
        log.debug("Minimization complete")

    # Reporters
    inmem_reporter = None
    if collect_trajectory:
        inmem_reporter = _InMemoryReporter(report_interval)
        sim.reporters.append(inmem_reporter)
    if dcd_path:
        sim.reporters.append(app.DCDReporter(dcd_path, report_interval))
    live_viewer = None
    if live_view:
        try:  # Lazy import to avoid hard dependency
            from .gui import build_live_viewer_reporter  # type: ignore

            lv_interval = int(live_interval) if live_interval else int(
                report_interval)
            symbols = ase_atoms.get_chemical_symbols()
            live_viewer, live_reporter = build_live_viewer_reporter(
                symbols, interval=lv_interval, hold_open=hold_open
            )
            sim.reporters.append(live_reporter)
        except Exception:  # pragma: no cover - GUI optional
            live_viewer = None
    # Always add a lightweight state reporter to STDOUT disabled by default? (Skipped for library)

    # Ensure we capture initial frame if collecting
    if collect_trajectory:
        state0 = sim.context.getState(
            getPositions=True, enforcePeriodicBox=False)
        inmem_reporter.report(sim, state0)  # type: ignore[arg-type]

    log.debug(
        "MD start steps=%d dt_fs=%s temp_K=%.2f platform=%s report_interval=%d live_view=%s",
        n_steps, dt_fs, temperature_K, platform or "auto", report_interval, live_view
    )
    sim.step(int(n_steps))
    log.debug("MD complete steps=%d", n_steps)

    # Final state
    final_state = sim.context.getState(
        getEnergy=True, getPositions=True, getVelocities=True)
    potential = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    kinetic = None
    temperature_final = None
    try:
        kinetic = final_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception:  # pragma: no cover - some builds may need velocities flag
        pass
    if kinetic is not None:
        # T = 2 KE / (dof * kB). Use approximate dof = 3N (no constraints assumed)
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kB_kj_per_mol_K = kB.value_in_unit(
            unit.kilojoule_per_mole / unit.kelvin)
        dof = 3 * topology.getNumAtoms()
        temperature_final = (2.0 * kinetic) / (dof * kB_kj_per_mol_K)

    traj_positions = None
    times_ps = None
    if collect_trajectory and inmem_reporter is not None:
        traj_positions = np.stack(inmem_reporter.frames, axis=0)
        times_ps = np.array(inmem_reporter.times_ps)

    # Determine traced dtype actually used (inspect internal cache key if needed)
    # heuristic; build_ani_torch_force updates cache key but we did not expose it
    actual_dtype = getattr(ani_force, "_animm_traced_dtype", dtype)

    # Finalize live viewer (leave window open)
    try:  # pragma: no cover - GUI specific
        if live_viewer is not None and getattr(live_viewer, "enabled", False):
            live_viewer.finalize()
    except Exception:
        pass

    res = MDResult(
        final_potential_kjmol=float(potential),
        final_temperature_K=temperature_final,
        steps=int(n_steps),
        model=model.upper(),
        dtype=actual_dtype,
        engine="openmm-torch",
        positions=traj_positions,
        times_ps=times_ps,
        dcd_path=dcd_path,
    )
    log.debug(
        "Result model=%s traced_dtype=%s final_potential=%.3fKJ/mol temp=%.2fK",
        res.model, res.dtype, res.final_potential_kjmol, (
            res.final_temperature_K or float('nan'))
    )
    return res


def minimize_and_md(
    ase_atoms: Atoms,
    ani_model,
    n_steps: int = 1000,
    temperature: float = 300.0,
    dt_fs: float = 0.5,
) -> MDState:  # pragma: no cover - thin wrapper
    """Compatibility wrapper delegating to :func:`run_ani_md`.

    Returns only an initial/final two–frame trajectory for legacy callers.
    """
    res = run_ani_md(
        ase_atoms=ase_atoms,
        model=ani_model,
        n_steps=n_steps,
        temperature_K=temperature,
        dt_fs=dt_fs,
        report_interval=n_steps,  # only initial + final if trajectory collected
        collect_trajectory=True,
        minimize=True,
    )
    if res.positions is not None and res.positions.shape[0] >= 2:
        positions = res.positions[:2]
    else:  # fallback synthetic
        positions = np.repeat(np.expand_dims(
            ase_atoms.get_positions(), 0), repeats=2, axis=0)
    velocities = np.zeros_like(positions)
    time = np.array([0.0, n_steps * dt_fs * 1e-3])
    return MDState(positions=positions, velocities=velocities, time=time)


__all__ = [
    "run_ani_md",
    "MDResult",
    "MDState",
    "minimize_and_md",
]
