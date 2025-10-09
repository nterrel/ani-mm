# ani-mm

Helpers to run TorchANI ANI models inside OpenMM via `TorchForce`, plus tiny utilities for SMILES → atoms, single‑point energy/forces, and a minimal MD example (alanine dipeptide) with an optional live viewer.

> Keep the README light. Detailed guides, provenance, and advanced notes now live under `docs/`.

Tested on macOS (Apple Silicon) and Linux. Windows unverified.

## Features

* SMILES → ASE `Atoms` (RDKit preferred; small fallback builder).
* Pretrained ANI models: `ANI2DR`, `ANI2X` (float64‑first trace, float32 fallback).
* TorchScript trace cache keyed by `(model, natoms, dtype)`.
* OpenMM `TorchForce` builder stamps provenance: `_animm_traced_dtype`, `_animm_cache_hit`.
* Minimal MD runner + alanine dipeptide vacuum shortcut (`ala2-md`).
* Optional live viewer (ASE GUI → Matplotlib fallback) with `--live-hold`.
* `--debug` flag surfaces cache hits, traced dtype, and a SUMMARY line.

## Quick Install (source)

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda env create -f environment.yml
conda activate ani-mm
pip install -e .[docs]
```

Planned PyPI: `pip install "ani-mm[ml]"` (will pull TorchANI instead of using submodule).

## Quick Usage

Python energy & forces:

```python
from animm.convert import smiles_to_ase
from animm.ani import load_ani_model, ani_energy_forces

atoms = smiles_to_ase("CCO")
calc = load_ani_model("ANI2DR")
ev = ani_energy_forces(calc, atoms)
print(ev.energy.item(), ev.forces.shape)
```

CLI (installed entry point `ani-mm`):

```bash
ani-mm models
ani-mm eval CCO --model ANI2X --json
ani-mm md "NCC(=O)O" --steps 1000 --model ANI2DR --dt 1.0 --t 300 --report 100 --dcd gly.dcd
ani-mm ala2-md --steps 400 --t 300 --dt 2.0 --report 50 --debug
ani-mm ala2-md --steps 400 --live-view --live-backend auto --live-hold
```

Sample provenance (debug) excerpt:

```text
[DEBUG] animm.ani_openmm: Cache miss trace attempt model=ANI2DR natoms=21 requested=float64
[DEBUG] animm.ani_openmm: Traced ANI model=ANI2DR natoms=21 traced=float64 cache_key=ANI2DR|21|float64
[DEBUG] animm.md: Attached ANI TorchForce model=ANI2DR natoms=21 traced_dtype=float64 cache=miss
SUMMARY steps=400 natoms=21 model=ANI2DR traced_dtype=float64 cache=hit initial=-1.4415e+06 final=-1.4415e+06 …
```

## Limitations

* Vacuum only (no periodic / cutoffs yet)
* Single ANI potential per system
* Minimal reporters (stdout + SUMMARY)
* No restarts / checkpoints

## Where Did The Details Go?

See the documentation (MkDocs Material): tracing & dtype rules, provenance attributes, environment knobs (`ANIMM_NO_OMP`, `--allow-dup-omp`), advanced install, roadmap, and live viewer notes.

## Roadmap (abridged)

| Done | Next |
|------|------|
| TorchForce + cache + provenance | Periodic / cutoffs |
| Energy/forces helpers | Rich reporters (CSV / JSON) |
| Alanine example & live viewer | Restart / checkpoint |
| Float64‑first tracing fallback | Mixed precision knobs |

Full list: `docs/development.md`.

## Tests

```bash
pytest -q tests
```

## License

MIT

---
If this saved you hand‑wiring a TorchForce, mission accomplished 🚀
