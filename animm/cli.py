"""Command line interface for ani-mm."""
from __future__ import annotations

import argparse
import sys

from .convert import smiles_to_ase
from .ani import load_ani_model, ani_energy_forces
from .alanine_dipeptide import simulate_alanine_dipeptide  # noqa: F401


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="ani-mm", description="ANI + OpenMM utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Evaluate ANI energy for a SMILES")
    p_eval.add_argument("smiles", help="SMILES string")
    p_eval.add_argument("--model", default="ANI2x",
                        help="ANI model name (default: ANI2x)")

    p_ala2 = sub.add_parser(
        "ala2-md", help="Run a short alanine dipeptide vacuum MD simulation")
    p_ala2.add_argument("--steps", type=int, default=2000,
                        help="Number of MD steps (default 2000)")
    p_ala2.add_argument("--t", type=float, default=300.0,
                        help="Temperature in K (default 300)")
    p_ala2.add_argument("--dt", type=float, default=2.0,
                        help="Timestep in fs (default 2.0)")
    p_ala2.add_argument("--report", type=int, default=200,
                        help="Report interval (steps)")
    p_ala2.add_argument("--dcd", default=None,
                        help="Optional DCD trajectory output path")
    p_ala2.add_argument("--platform", default=None,
                        help="OpenMM platform name (e.g. CUDA, CPU)")

    args = parser.parse_args(argv)

    if args.cmd == "eval":
        atoms = smiles_to_ase(args.smiles)
        model = load_ani_model(args.model)
        eval_res = ani_energy_forces(model, atoms)
        hartree_to_kcalmol = 627.509474
        energy_kcal = eval_res.energy.item() * hartree_to_kcalmol
        print(
            f"Energy: {energy_kcal:.4f} kcal/mol ({eval_res.energy.item():.6f} Ha)")
        return 0

    if args.cmd == "ala2-md":
        sim_info = simulate_alanine_dipeptide(
            n_steps=args.steps,
            temperature=args.t,
            timestep_fs=args.dt,
            report_interval=args.report,
            out_dcd=args.dcd,
            platform_name=args.platform,
        )
        print(
            f"Finished: {sim_info['steps']} steps, final potential {sim_info['final_potential_kjmol']:.2f} kJ/mol")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
