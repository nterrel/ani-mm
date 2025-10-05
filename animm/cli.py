"""Command line interface for ani-mm.

Mitigations:
    * Set ANIMM_NO_OMP=1 to clamp thread counts to 1 for common math libs.
    * Use --allow-dup-omp (unsafe) to set KMP_DUPLICATE_LIB_OK=TRUE if you cannot
        immediately rebuild a clean environment and are hitting the duplicate
        libomp abort on macOS.
"""
from __future__ import annotations

import argparse
import os
import sys

# Apply OpenMP thread limiting early if user requested
if os.environ.get("ANIMM_NO_OMP") == "1":  # pragma: no cover - environment specific
    for var, val in {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(var, val)

from .convert import smiles_to_ase


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="ani-mm", description="ANI + OpenMM utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    parser.add_argument(
        "--allow-dup-omp",
        action="store_true",
        help="Unsafe: set KMP_DUPLICATE_LIB_OK=TRUE to bypass duplicate OpenMP runtime abort (macOS).",
    )

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
    p_ala2.add_argument("--force-mode", default="amber", choices=["amber", "ani", "hybrid"],
                        help="Which forces to use: classical Amber, ANI only, or hybrid (Amber + ANI additive)")
    p_ala2.add_argument("--ani-model", default="ANI2x", help="ANI model name (currently ANI2x)")
    p_ala2.add_argument("--ani-threads", type=int, default=None, help="Override Torch thread count for ANI force")

    args = parser.parse_args(argv)

    # Apply unsafe duplicate OpenMP override if explicitly requested
    if args.allow_dup_omp:  # pragma: no cover - environment specific
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Lazy import heavy torch/openmm dependent modules after env tweaks
    from .ani import load_ani_model, ani_energy_forces  # noqa: WPS433

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
        from .alanine_dipeptide import simulate_alanine_dipeptide  # noqa: WPS433

        sim_info = simulate_alanine_dipeptide(
            n_steps=args.steps,
            temperature=args.t,
            timestep_fs=args.dt,
            report_interval=args.report,
            out_dcd=args.dcd,
            platform_name=args.platform,
            force_mode=args.force_mode,
            ani_model=args.ani_model,
            ani_threads=args.ani_threads,
        )
        print(
            f"Finished: {sim_info['steps']} steps, mode={sim_info['force_mode']}, final potential {sim_info['final_potential_kjmol']:.2f} kJ/mol")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
