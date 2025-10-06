# User Guide

Practical recipes for `ani-mm`.

## 1. List Available Models

```bash
ani-mm models
```

Programmatic:

```python
from animm.ani import list_available_ani_models
print(list_available_ani_models())
```

## 2. Single‑Point Energy & Forces

CLI (SMILES):

```bash
ani-mm eval CCO --model ANI2X --json
```

Python:

```python
from animm.convert import smiles_to_ase
from animm.ani import load_ani_model, ani_energy_forces

atoms = smiles_to_ase("CCO")
calc = load_ani_model("ANI2DR")
ev = ani_energy_forces(calc, atoms)
print(ev.energy.item(), ev.forces.shape)
```

## 3. Alanine Dipeptide MD Example

Run a short vacuum MD, reporting every 50 steps:

```bash
ani-mm ala2-md --steps 500 --report 50 --ani-model ANI2DR
```

JSON output (quiet mode) for integration with scripts:

```bash
ani-mm ala2-md --steps 500 --json
```

Key flags:

* `--steps` number of MD steps (default 2000)
* `--t` temperature (K)
* `--dt` timestep (fs)
* `--report` reporting interval (steps)
* `--ani-model` model name (`ANI2DR`, `ANI2X`)
* `--platform` force a specific OpenMM platform (e.g. CUDA)

## 4. Programmatic MD Runner

```python
from animm.alanine_dipeptide import simulate_alanine_dipeptide

info = simulate_alanine_dipeptide(n_steps=300, ani_model="ANI2X", report_interval=50)
print(info["steps"], info["traced_dtype"], info["cache_hit"])
```

## 5. TorchScript Tracing & Cache

Tracing attempts float64 first. If the underlying model / operations reject double precision, a float32 retry occurs. The effective dtype appears as:

* Force object attribute `_animm_traced_dtype`
* MD result dict key `traced_dtype`

Cache key: `(model_name, natoms, dtype)`. Cache hit state is surfaced as `_animm_cache_hit` on the force and `cache_hit` in results.

## 6. Provenance / Debugging

Add `--debug` (or `--log-level DEBUG`) to surface lines showing:

* Trace attempt & success (model, natoms, dtype)
* Attachment of `TorchForce` with traced dtype & cache status
* A final `SUMMARY` line summarizing steps, natoms, model, traced dtype, cache, and energy delta

Example fragment:

```text
[DEBUG] animm.ani_openmm: Traced ANI model=ANI2DR natoms=21 traced=float64 cache_key=ANI2DR|21|float64
[DEBUG] animm.md: Attached ANI TorchForce model=ANI2DR natoms=21 traced_dtype=float64 cache=miss
SUMMARY steps=200 natoms=21 model=ANI2DR traced_dtype=float64 cache=hit initial=-1.4415e+06 final=-1.4415e+06
```

## 7. Converting SMILES → Atoms

```python
from animm.convert import smiles_to_ase
atoms = smiles_to_ase("CCO")
```

RDKit is preferred; if absent a minimal fallback builder is used.

## 8. Live Viewer

Enable during MD:

```bash
ani-mm ala2-md --steps 400 --live-view --live-backend auto --live-hold
```

Backends:

* `auto` (ASE GUI → Matplotlib)
* `ase` force ASE GUI
* `mpl` force Matplotlib

Headless environments auto‑disable viewing. See `live-viewer.md` for performance and color notes.

Programmatic generic MD (simplified example):

```python
from animm.convert import smiles_to_ase
from animm.openmm_runner import run_ani_md

atoms = smiles_to_ase("CCO")
res = run_ani_md(atoms, n_steps=250, live_view=True, live_interval=50)
print(res.traced_dtype)
```

## 9. Environment Knobs

* `ANIMM_NO_OMP=1` limit OpenMP / BLAS threads (sets OMP_NUM_THREADS etc.)
* `--allow-dup-omp` sets `KMP_DUPLICATE_LIB_OK=TRUE` (macOS duplicate runtime workaround)

## 10. Troubleshooting

* Unexpected float32? Look at `_animm_traced_dtype` (float64 attempt failed). Performance differences are usually minor at this scale.
* Force mismatch shapes: ensure atoms ordering unchanged after conversion.
* No GUI? Use `--live-backend mpl` or omit `--live-view`.
* Warnings about deprecated `simtk.openmm` are filtered; can be restored by removing the filters in `animm/__init__.py` and `cli.py`.

## 11. Next Steps

For roadmap, development workflow, and contribution guide see `development.md`.

## 12. Minimal API Surface

Primary entry points:

* `animm.ani.load_ani_model`
* `animm.ani.ani_energy_forces`
* `animm.convert.smiles_to_ase`
* `animm.alanine_dipeptide.simulate_alanine_dipeptide`
* `animm.openmm_runner.run_ani_md` (generic; internal usage may evolve)

See API reference for full docstrings.
