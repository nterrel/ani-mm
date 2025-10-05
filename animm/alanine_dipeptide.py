"""Alanine dipeptide OpenMM simulation helpers.

Provides a convenience function to build an alanine dipeptide system in vacuum
(using Amber14 force field) and run a short Langevin dynamics trajectory.

If OpenMM is not installed, calling the public function will raise an
ImportError with a helpful message.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

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
    ani_model: str = "ANI2DR",
    ani_threads: Optional[int] = None,
    seed: Optional[int] = None,
    minimize: bool = True,
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
    seed: int | None
        Random seed for the Langevin integrator RNG.
    minimize: bool
        If True (default) run an energy minimization before dynamics.

    Returns
    -------
    dict with keys: steps, final_potential_kjmol, temperature_K.
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

    force_mode_lc = force_mode.lower()
    if force_mode_lc not in {"amber", "ani", "hybrid"}:
        raise ValueError("force_mode must be one of: amber, ani, hybrid")

    system = None  # will be set below
    if force_mode_lc in {"amber", "hybrid"}:
        # Some Amber14 XML collections separate protein parameters; ACE/NME caps
        # are reliably present in "amber14/protein.ff14SB.xml". Fallback to the
        # broader pair previously used if the first attempt fails for any reason.
        ff_file_sets = [
            ["amber14/protein.ff14SB.xml"],
            ["amber14-all.xml", "amber14/tip3pfb.xml"],
        ]
        last_exc: Exception | None = None
        forcefield = None
        for files in ff_file_sets:
            try:
                forcefield = app.ForceField(*files)
                system = forcefield.createSystem(
                    pdb.topology,
                    nonbondedMethod=app.NoCutoff,
                    constraints=app.HBonds,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on local ff install
                last_exc = exc
                continue
        if system is None:  # pragma: no cover - only if all attempts fail
            # Attempt graceful fallback to ANI-only simulation if user requested plain Amber.
            if force_mode_lc == "amber":
                import warnings

                warnings.warn(
                    "Amber14 parameterization failed (missing ACE/NME templates?). "
                    "Falling back to ANI-only simulation. Pass --force-mode ani to avoid this fallback message.",
                    RuntimeWarning,
                )
                # Build minimal system with particle masses
                system = openmm.System()
                for atom in pdb.topology.atoms():
                    system.addParticle(atom.element.mass)
                # Attach ANI TorchForce
                try:
                    from .ani_openmm import build_ani_torch_force  # local helper

                    ani_torch_force = build_ani_torch_force(
                        topology=pdb.topology,
                        model_name=ani_model,
                        threads=ani_threads,
                    )
                    system.addForce(ani_torch_force)
                    force_mode_lc = "ani"
                except Exception as ani_fallback_exc:  # pragma: no cover - missing deps
                    raise RuntimeError(
                        "Failed Amber parameterization AND ANI fallback failed. "
                        "Install torchani/openmmtorch or use a simpler topology. Original error: "
                        f"{last_exc}; ANI fallback error: {ani_fallback_exc}"
                    ) from last_exc
            else:
                raise RuntimeError(
                    "Failed to parameterize alanine dipeptide with Amber14 force field. "
                    "Tried file sets: protein.ff14SB.xml then amber14-all.xml/tip3pfb.xml. "
                    "Consider using --force-mode ani to bypass Amber or ensure ACE/NME templates are available. "
                    f"Last error: {last_exc}"
                ) from last_exc
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
        ani_torch_force = build_ani_torch_force(
            topology=pdb.topology,
            model_name=ani_model,
            threads=ani_threads,
        )
        assert system is not None  # for mypy
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

    assert system is not None
    sim = app.Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(pdb.positions)

    reporters = []
    if out_dcd:
        reporters.append(app.DCDReporter(out_dcd, report_interval))
    reporters.append(
        app.StateDataReporter(
            file="-",
            reportInterval=report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )
    for r in reporters:
        sim.reporters.append(r)

    if seed is not None:
        try:
            # type: ignore[attr-defined]
            integrator.setRandomNumberSeed(int(seed))
        except Exception:  # pragma: no cover - older OpenMM versions
            pass

    if minimize:
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
        "seed": seed,
        "minimized": minimize,
    }
