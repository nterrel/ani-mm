"""Alanine dipeptide OpenMM simulation helpers.

Provides a convenience function to build an alanine dipeptide system in vacuum
(using Amber14 force field) and run a short Langevin dynamics trajectory.

If OpenMM is not installed, calling the public function will raise an
ImportError with a helpful message.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:  # pragma: no cover - optional dependency placeholder
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
    force_mode: str = "amber",  # 'amber' | 'ani' | 'hybrid'
    ani_model: str = "ANI2x",
    ani_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a short alanine dipeptide MD simulation in vacuum.

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

    Returns
    -------
    dict with keys: steps, final_potential_kjmol, temperature_K.
    """
    if openmm is None or app is None or unit is None:  # pragma: no cover
        raise ImportError(
            "OpenMM is required for simulate_alanine_dipeptide; install openmm first.")

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

    pdb = app.PDBFile(StringIO(ala2_pdb_str))

    force_mode_lc = force_mode.lower()
    if force_mode_lc not in {"amber", "ani", "hybrid"}:
        raise ValueError("force_mode must be one of: amber, ani, hybrid")

    if force_mode_lc in {"amber", "hybrid"}:
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
        )
    else:
        system = openmm.System()
        # Add particles with masses from topology
        for atom in pdb.topology.atoms():
            system.addParticle(atom.element.mass)

    if force_mode_lc in {"ani", "hybrid"}:
        try:
            from .ani_openmm import build_ani_torch_force  # local helper
        except ImportError as exc:  # pragma: no cover - plugin missing
            raise ImportError(
                "ANI force requested but openmmtorch / torchani dependencies not satisfied."
            ) from exc
        torch_force = build_ani_torch_force(
            topology=pdb.topology,
            model_name=ani_model,
            threads=ani_threads,
        )
        system.addForce(torch_force)

    # Integrator: convert timestep fs -> ps
    dt_ps = timestep_fs * 1e-3
    integrator = openmm.LangevinIntegrator(
        temperature * unit.kelvin,
        friction_per_ps / unit.picosecond,
        dt_ps * unit.picoseconds,
    )

    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    reporters = []
    if out_dcd:
        reporters.append(app.DCDReporter(out_dcd, report_interval))
    reporters.append(app.StateDataReporter(
        file="-", reportInterval=report_interval, step=True, potentialEnergy=True, temperature=True
    ))
    for r in reporters:
        sim.reporters.append(r)

    sim.minimizeEnergy()
    sim.step(n_steps)

    state = sim.context.getState(getEnergy=True)
    final_pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return {
        "steps": n_steps,
        "final_potential_kjmol": final_pot,
        "temperature_K": temperature,
        "dcd_path": out_dcd,
        "force_mode": force_mode_lc,
    }
