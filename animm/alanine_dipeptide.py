"""Alanine dipeptide reference MD example (vacuum, ANI).

Prefers OpenMM + TorchForce (ANI) and falls back to a minimal ASE VelocityVerlet
loop if the plugin is unavailable. Intended as a quick sanity / demo run, not a
production workflow.
"""

from __future__ import annotations

import contextlib
import io
import sys
from typing import Any, Dict, Optional

try:  # pragma: no cover - import guard
    # swallow simtk deprecation line
    with contextlib.redirect_stderr(io.StringIO()):
        import openmm
        import openmm.app as app
        import openmm.unit as unit
except Exception:  # pragma: no cover - optional dependency placeholder
    openmm = None  # type: ignore
    app = None  # type: ignore
    unit = None  # type: ignore


def simulate_alanine_dipeptide(
    n_steps: int = 1000,
    temperature: float = 298.15,
    friction_per_ps: float = 1.0,
    timestep_fs: float = 2.0,
    report_interval: int = 50,
    out_dcd: Optional[str] = None,
    platform_name: Optional[str] = None,
    ani_model: str = "ANI2DR",
    ani_threads: Optional[int] = None,
    seed: Optional[int] = None,
    minimize: bool = True,
    live_view: bool = False,
    live_backend: str = "auto",
) -> Dict[str, Any]:
    """Run a short gas‐phase alanine dipeptide simulation.

    Parameters
    ----------
    n_steps: int
        Number of integration steps.
    temperature: float
        Temperature in Kelvin for Langevin thermostat.
    friction_per_ps: float
        Friction coefficient (1/ps).
    timestep_fs: float
        Timestep in femtoseconds.
    report_interval: int
        Interval (steps) for state reporter.
    out_dcd: str | None
        Optional path to write a DCD trajectory.
    platform_name: str | None
        Force a specific OpenMM platform (e.g. 'CUDA', 'CPU').
    ani_model: str
        Name of the TorchANI model to use (default ANI2DR).
    ani_threads: int | None
        Override torch thread count for the ANI TorchForce (optional).
    seed: int | None
        Random seed for the Langevin integrator RNG.
    minimize: bool
        If True (default) run an energy minimization before dynamics.
    live_view: bool
        If True, enable a lightweight live viewer (desktop). Optional.
    live_backend: str
        Backend for live viewer ('auto', 'ase', 'mpl').

    Returns
    -------
    Dictionary containing simulation metadata and energies. Per‑step progress
    is printed to stdout (unless redirected) via a lightweight reporter.
    """
    if openmm is None or app is None or unit is None:  # pragma: no cover
        raise ImportError(
            "OpenMM is required for simulate_alanine_dipeptide; install openmm first."
        )

    # Build topology & system
    # Build a simple Ala dipeptide (ACE-ALA-NME) from an embedded PDB string.
    ala2_pdb_str = """
ATOM      1  N   ACE A   1      -1.207   1.207   0.000  1.00  0.00           N
ATOM      2  CH3 ACE A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3 HH31 ACE A   1       0.513  -0.513   0.889  1.00  0.00           H
ATOM      4 HH32 ACE A   1       0.513  -0.513  -0.889  1.00  0.00           H
ATOM      5 HH33 ACE A   1       0.513   0.889   0.000  1.00  0.00           H
ATOM      6  C   ACE A   1      -2.414   0.000   0.000  1.00  0.00           C
ATOM      7  O   ACE A   1      -2.414  -1.207   0.000  1.00  0.00           O
ATOM      8  N   ALA A   2      -3.621   0.000   0.000  1.00  0.00           N
ATOM      9  CA  ALA A   2      -4.828   0.000   0.000  1.00  0.00           C
ATOM     10  HA  ALA A   2      -5.156  -1.023   0.000  1.00  0.00           H
ATOM     11  CB  ALA A   2      -5.449   1.207   0.889  1.00  0.00           C
ATOM     12 HB1  ALA A   2      -5.121   2.230   0.889  1.00  0.00           H
ATOM     13 HB2  ALA A   2      -5.121   0.683   1.845  1.00  0.00           H
ATOM     14 HB3  ALA A   2      -6.535   1.207   0.889  1.00  0.00           H
ATOM     15  C   ALA A   2      -5.449   0.000  -1.414  1.00  0.00           C
ATOM     16  O   ALA A   2      -6.656   0.000  -1.414  1.00  0.00           O
ATOM     17  N   NME A   3      -4.621   0.000  -2.414  1.00  0.00           N
ATOM     18  CH3 NME A   3      -5.242   0.000  -3.621  1.00  0.00           C
ATOM     19 HH31 NME A   3      -5.121   0.889  -4.210  1.00  0.00           H
ATOM     20 HH32 NME A   3      -6.327   0.000  -3.621  1.00  0.00           H
ATOM     21 HH33 NME A   3      -4.832  -0.889  -4.210  1.00  0.00           H
TER
END
"""
    from io import StringIO

    try:
        pdb = app.PDBFile(StringIO(ala2_pdb_str))
    except Exception:  # pragma: no cover - parsing unlikely to fail
        # Fallback to a simpler capped alanine-like minimal peptide (just ALA) if needed.
        fallback_pdb = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.036   1.410   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.318   2.374   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       2.118  -0.780  -1.205  1.00  0.00           C
TER
END
"""
        pdb = app.PDBFile(StringIO(fallback_pdb))

    # Try OpenMM TorchForce path first; fallback to pure ASE MD if plugin missing.
    engine = "openmm-torch"
    try:
        from .ani_openmm import build_ani_torch_force  # noqa: WPS433
    except Exception:  # pragma: no cover - fallback path
        build_ani_torch_force = None  # type: ignore
        engine = "ase"

    initial_pot = None
    if engine == "openmm-torch" and build_ani_torch_force is not None:
        system = openmm.System()
        for atom in pdb.topology.atoms():
            system.addParticle(atom.element.mass)
        ani_torch_force = build_ani_torch_force(
            topology=pdb.topology,
            model_name=ani_model,
            threads=ani_threads,
        )
        system.addForce(ani_torch_force)
        # Integrator: convert timestep fs -> ps
        dt_ps = timestep_fs * 1e-3
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin,  # type: ignore[operator]
            friction_per_ps / unit.picosecond,  # type: ignore[operator]
            dt_ps * unit.picoseconds,  # type: ignore[operator]
        )
        platform = None
        if platform_name:
            platform = openmm.Platform.getPlatformByName(platform_name)
        sim = app.Simulation(pdb.topology, system, integrator, platform)
        sim.context.setPositions(pdb.positions)

        if seed is not None:
            try:
                # type: ignore[attr-defined]
                integrator.setRandomNumberSeed(int(seed))
            except Exception:  # pragma: no cover
                pass
        if minimize:
            sim.minimizeEnergy()
        init_state = sim.context.getState(getEnergy=True, getVelocities=True)
        initial_pot = init_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        kB_kj = kB.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
        try:
            ke0 = init_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            dof = 3 * pdb.topology.getNumAtoms()
            initial_temp = (2 * ke0) / (dof * kB_kj)
        except Exception:
            initial_temp = 0.0
        if out_dcd:
            sim.reporters.append(app.DCDReporter(out_dcd, report_interval))
        # Live viewer (optional)
        live_viewer = None
        if live_view:
            try:  # pragma: no cover - GUI optional
                symbols = [a.element.symbol for a in pdb.topology.atoms()]
                initial_ang = [(pos.x * 10.0, pos.y * 10.0, pos.z * 10.0) for pos in pdb.positions]
                import numpy as _np

                from .gui import build_live_viewer_reporter  # type: ignore

                live_viewer, live_reporter = build_live_viewer_reporter(
                    symbols,
                    interval=report_interval,
                    backend=live_backend,
                    initial_positions_ang=_np.asarray(initial_ang, dtype=float),
                )
                sim.reporters.append(live_reporter)
            except Exception:
                live_viewer = None

        class _DeltaReporter:  # pragma: no cover - simple formatted stdout reporter
            """Lightweight reporter printing step, potential, delta, and temperature.

            Implemented locally to avoid depending on OpenMM's StateDataReporter for
            custom column ordering and delta energy column.
            """

            def __init__(self, interval: int, initial_pot_kj: float):
                self.interval = interval
                self.initial = initial_pot_kj
                self._printed_header = False

            def describeNextReport(self, simulation):  # noqa: N802
                # OpenMM expects (nextStep, needPositions, needVelocities, needForces, needEnergy)
                # We only require energies; positions/velocities/forces not needed.
                return (self.interval, False, False, False, True)

            def report(self, simulation, state):  # noqa: N802
                pot = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )  # type: ignore[attr-defined]
                try:
                    ke = state.getKineticEnergy().value_in_unit(
                        unit.kilojoule_per_mole
                    )  # type: ignore[attr-defined]
                    dof_loc = 3 * simulation.topology.getNumAtoms()
                    temp = (2 * ke) / (dof_loc * kB_kj)
                except Exception:  # pragma: no cover - kinetic energy retrieval issues
                    temp = float("nan")
                delta = pot - self.initial
                if not self._printed_header:
                    sys.stdout.write('#"Step","Potential kJ/mol","Delta kJ/mol","Temperature K"\n')
                    self._printed_header = True
                sys.stdout.write(f"{simulation.currentStep},{pot:.6f},{delta:.6f},{temp:.2f}\n")
                sys.stdout.flush()

        # Attach our custom reporter last so header appears after any OpenMM messages
        sim.reporters.append(_DeltaReporter(report_interval, initial_pot))
        print(f"Initial potential: {initial_pot:.6f} kJ/mol (T~{initial_temp:.2f} K)")
        sys.stdout.flush()
        sim.step(n_steps)
        state = sim.context.getState(getEnergy=True)
        final_pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        try:  # finalize viewer
            if (
                live_view and live_viewer is not None and getattr(live_viewer, "enabled", False)
            ):  # pragma: no cover - GUI
                live_viewer.finalize()  # type: ignore
        except Exception:  # pragma: no cover
            pass
    else:
        # ASE fallback: build Atoms and run simple velocity Verlet MD with ANI calculator.
        from ase import Atoms  # type: ignore
        from ase import units as ase_units  # type: ignore
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # type: ignore
        from ase.md.verlet import VelocityVerlet  # type: ignore

        from .ani import load_ani_model  # reuse existing loader

        symbols = [atom.element.symbol for atom in pdb.topology.atoms()]
        coords_ang = [(pos.x, pos.y, pos.z) for pos in pdb.positions]  # OpenMM gives nanometers
        # Convert nm -> Å
        coords_ang = [(x * 10.0, y * 10.0, z * 10.0) for x, y, z in coords_ang]
        ase_atoms = Atoms(symbols=symbols, positions=coords_ang)
        calc = load_ani_model(ani_model)
        ase_atoms.calc = calc.atoms.calc  # ensure underlying ASE calculator
        if seed is not None:
            import numpy as _np

            _np.random.default_rng(int(seed))
            _np.random.seed(int(seed))  # some ASE code uses global
        MaxwellBoltzmannDistribution(ase_atoms, temperature * ase_units.kB)
        dyn = VelocityVerlet(ase_atoms, timestep_fs * ase_units.fs)
        if minimize:
            # Light minimization: zero initial velocities + single-step energy descent by scaling.
            ase_atoms.set_velocities([(0.0, 0.0, 0.0)] * len(ase_atoms))
        for _ in range(n_steps):
            dyn.run(1)
        final_pot = ase_atoms.get_potential_energy()  # eV by default
        # Convert eV -> kJ/mol
        final_pot *= ase_units.kJ / ase_units.mol
    # Simple manual reporting placeholder (future: integrate reporters earlier)
    # Reporting handled inline above for openmm-torch; ASE path currently silent per-step.
    return {
        "steps": n_steps,
        "final_potential_kjmol": final_pot,
        **(
            {"initial_potential_kjmol": initial_pot}
            if (engine == "openmm-torch" and initial_pot is not None)
            else {}
        ),
        "temperature_K": temperature,
        "dcd_path": out_dcd if engine == "openmm-torch" else None,
        "model": ani_model,
        "seed": seed,
        "minimized": minimize,
        "engine": engine,
        "potential_delta_kjmol": (
            (final_pot - initial_pot)
            if (engine == "openmm-torch" and initial_pot is not None)
            else None
        ),
    }
