"""Command‑line interface.

Environment knobs:
* ``ANIMM_NO_OMP=1`` – force single threading (BLAS / OpenMP libs).
* ``--allow-dup-omp`` – add ``KMP_DUPLICATE_LIB_OK=TRUE`` (macOS workaround).

Use ``--debug`` for provenance (trace dtype, cache hits). Noisy third‑party
debug logs are down‑leveled so ``animm.*`` messages stand out.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import warnings

from .convert import smiles_to_ase

# Suppress noisy third-party warnings/log messages before heavy imports
warnings.filterwarnings(
    "ignore",
    message=r"ANI-2xr is experimental",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Warning: importing 'simtk.openmm' is deprecated",
    category=UserWarning,
)


class _SimTKDeprecationFilter(logging.Filter):  # pragma: no cover - simple filter
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return "importing 'simtk.openmm' is deprecated" not in record.getMessage()


logging.getLogger().addFilter(_SimTKDeprecationFilter())


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


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="ani-mm", description="ANI + OpenMM utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (DEBUG, INFO, WARNING, ERROR) (overridden by --debug)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging (cache hits, trace dtype, backend provenance)",
    )

    parser.add_argument(
        "--allow-dup-omp",
        action="store_true",
        help="Unsafe: set KMP_DUPLICATE_LIB_OK=TRUE to bypass duplicate OpenMP runtime abort (macOS).",  # noqa: E501
    )

    p_eval = sub.add_parser("eval", help="Evaluate ANI energy for a SMILES")
    p_eval.add_argument("smiles", help="SMILES string")
    p_eval.add_argument(
        "--model",
        default="ANI2DR",
        help="ANI model name (default: ANI2DR; options: ANI2DR, ANI2X, ANI2XPeriodic)",
    )
    p_eval.add_argument("--json", action="store_true",
                        help="Emit JSON instead of text")

    p_ala2 = sub.add_parser(
        "ala2-md", help="Run a short alanine dipeptide vacuum MD simulation")
    p_ala2.add_argument("--steps", type=int, default=2000,
                        help="Number of MD steps (default 2000)")
    p_ala2.add_argument("--t", type=float, default=300.0,
                        help="Temperature in K (default 300)")
    p_ala2.add_argument("--dt", type=float, default=2.0,
                        help="Timestep in fs (default 2.0)")
    p_ala2.add_argument(
        "--report", type=int, default=50, help="Report interval (steps, default 50)"
    )
    p_ala2.add_argument("--dcd", default=None,
                        help="Optional DCD trajectory output path")
    p_ala2.add_argument("--platform", default=None,
                        help="OpenMM platform name (e.g. CUDA, CPU)")
    p_ala2.add_argument(
        "--ani-model",
        default="ANI2DR",
        help="ANI model name (default ANI2DR; options: ANI2DR, ANI2X, ANI2XPeriodic)",
    )
    p_ala2.add_argument(
        "--ani-threads", type=int, default=None, help="Override Torch thread count for ANI force"
    )
    p_ala2.add_argument("--seed", type=int, default=None,
                        help="Random seed for integrator RNG")
    p_ala2.add_argument("--no-min", action="store_true",
                        help="Skip energy minimization")
    p_ala2.add_argument("--json", action="store_true",
                        help="Emit JSON instead of text")
    p_ala2.add_argument(
        "--live-view",
        action="store_true",
        help="Enable live desktop viewer (ASE GUI preferred, fallback matplotlib)",
    )
    p_ala2.add_argument(
        "--live-backend",
        default="auto",
        choices=["auto", "ase", "mpl"],
        help="Force live viewer backend (default auto)",
    )
    p_ala2.add_argument(
        "--live-hold",
        action="store_true",
        help="If set and using live viewer, keep window open after dynamics (blocks until closed)",
    )

    p_models = sub.add_parser("models", help="List available ANI models")
    p_models.add_argument("--json", action="store_true", help="Emit JSON list")

    args = parser.parse_args(argv)

    # Apply unsafe duplicate OpenMP override if explicitly requested
    if args.allow_dup_omp:  # pragma: no cover - environment specific
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    effective_level = "DEBUG" if args.debug else args.log_level.upper()
    logging.basicConfig(
        level=getattr(logging, effective_level,
                      logging.DEBUG if args.debug else logging.WARNING)
    )
    if args.debug:
        # Reduce noise from third-party debug spew so our provenance stands out
        for noisy in [
            "filelock",
            "urllib3",
            "huggingface_hub",
            "PIL",
            "matplotlib.font_manager",
        ]:
            logging.getLogger(noisy).setLevel(logging.INFO)
    log = logging.getLogger("animm.cli")
    log.debug("CLI args: %s", vars(args))

    # Lazy import heavy torch/openmm dependent modules after env tweaks
    from .ani import ani_energy_forces, list_available_ani_models, load_ani_model  # noqa: WPS433

    if args.cmd == "models":
        models = list_available_ani_models()
        if getattr(args, "json", False):
            print(json.dumps(models))
        else:
            print("Available ANI models:")
            for m in models:
                print(" -", m)
        log.debug("Listed %d models", len(models))
        return 0

    if args.cmd == "eval":
        atoms = smiles_to_ase(args.smiles)
        model = load_ani_model(args.model)
        eval_res = ani_energy_forces(model, atoms)
        hartree_to_kcalmol = 627.509474
        energy_kcal = eval_res.energy.item() * hartree_to_kcalmol
        log.debug(
            "Eval SMILES=%s model=%s natoms=%d energy(Ha)=%.6f", args.smiles, args.model, len(
                atoms), eval_res.energy.item()
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "smiles": args.smiles,
                        "model": args.model,
                        "energy_hartree": eval_res.energy.item(),
                        "energy_kcal_mol": energy_kcal,
                        "natoms": len(atoms),
                    }
                )
            )
        else:
            print(
                f"SMILES={args.smiles} model={args.model} energy={energy_kcal:.4f} kcal/mol ({eval_res.energy.item():.6f} Ha)"
            )
        return 0

    if args.cmd == "ala2-md":
        from .alanine_dipeptide import simulate_alanine_dipeptide  # noqa: WPS433

        if args.json:
            # Suppress internal stdout (reporter + initial potential) so we emit pure JSON
            with contextlib.redirect_stdout(io.StringIO()):
                sim_info = simulate_alanine_dipeptide(
                    n_steps=args.steps,
                    temperature=args.t,
                    timestep_fs=args.dt,
                    report_interval=args.report,
                    out_dcd=args.dcd,
                    platform_name=args.platform,
                    ani_model=args.ani_model,
                    ani_threads=args.ani_threads,
                    seed=args.seed,
                    minimize=not args.no_min,
                    live_view=args.live_view,
                    live_backend=args.live_backend,
                    hold_open=args.live_hold,
                )
            print(json.dumps(sim_info))
        else:
            sim_info = simulate_alanine_dipeptide(
                n_steps=args.steps,
                temperature=args.t,
                timestep_fs=args.dt,
                report_interval=args.report,
                out_dcd=args.dcd,
                platform_name=args.platform,
                ani_model=args.ani_model,
                ani_threads=args.ani_threads,
                seed=args.seed,
                minimize=not args.no_min,
                live_view=args.live_view,
                live_backend=args.live_backend,
                hold_open=args.live_hold,
            )
            extra = ""
            if "initial_potential_kjmol" in sim_info:
                delta = sim_info["final_potential_kjmol"] - \
                    sim_info["initial_potential_kjmol"]
                extra = f" initial_potential={sim_info['initial_potential_kjmol']:.2f} delta={delta:.2f}"
            print(
                f"Finished steps={sim_info['steps']} model={sim_info['model']} final_potential={sim_info['final_potential_kjmol']:.2f} kJ/mol{extra}"
            )
            # Summary line (distinct from 'Finished' for easy grepping)
            traced = sim_info.get("traced_dtype")
            cache = sim_info.get("cache_hit")
            if traced is not None:
                summary = (
                    f"SUMMARY steps={sim_info['steps']} natoms={sim_info.get('natoms','?')} model={sim_info['model']} "
                    f"traced_dtype={traced} cache={'hit' if cache else 'miss' if cache is not None else 'n/a'} "
                    f"initial={sim_info.get('initial_potential_kjmol','?')} final={sim_info['final_potential_kjmol']:.6f}"
                )
                delta = sim_info.get("potential_delta_kjmol")
                if delta is not None:
                    summary += f" delta={delta:.6f} kJ/mol"
                print(summary)
        log.debug(
            "Alanine MD done steps=%s model=%s final=%.3f engine=%s",
            sim_info.get('steps'), sim_info.get('model'), sim_info.get(
                'final_potential_kjmol'), sim_info.get('engine')
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
