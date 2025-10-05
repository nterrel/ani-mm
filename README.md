# ani-mm

Small, pragmatic helpers for using TorchANI ANI models with OpenMM (and ASE for a fallback or a GUI). Goal: stay lean: load a model, get energies/forces, drop it into an OpenMM `TorchForce`, run a quick MD loop, optionally watch atoms jiggle, then move on.

Tested mainly on macOS (Apple Silicon) and Linux so far. Windows is untested.

## Scope (what it really does)

* Convert a SMILES string to an ASE `Atoms` (RDKit first, tiny fallback otherwise).
* Load a named pretrained ANI model (`ANI2DR`, `ANI2X`).
* Single‑point energy + forces (try float64 trace → fall back to float32 only if required).
* Build a single OpenMM `TorchForce` hosting that trace; forces come from the NN each MD step.
* Provide a “press go” alanine dipeptide vacuum example (`ala2-md`).
* Generic `run_ani_md` (optional in‑memory trajectory or DCD file).
* Live viewer (ASE GUI if available, otherwise Matplotlib 3D scatter) with `--live-hold` to keep the window open after the run.

Not trying to be: a general workflow platform, a training suite, or a periodic variant of non‑periodic models (yet).

## Install

The repo vendors TorchANI as a submodule for reproducibility, but you can also grab TorchANI via the `ml` extra.

### Fast path

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda env create -f environment.yml
conda activate ani-mm
pip install -e .[dev,docs]
```

### Manual (Conda)

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda create -n ani-mm python=3.10 -y
conda activate ani-mm
conda install -c conda-forge openmm openmm-torch ase rdkit pytorch -y
pip install -e external/torchani
pip install -e .[dev,docs]
```

### Pure pip

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openmm openmm-torch ase torch rdkit-pypi
pip install -e external/torchani
pip install -e .[dev,docs]
```

### GitHub TorchANI (no submodule)

```bash
pip install .[ml]
```

### Update vendored TorchANI

```bash
git submodule update --remote external/torchani
pip install -e external/torchani
```

### Pin a TorchANI commit

```bash
cd external/torchani
git checkout <COMMIT_SHA>
cd ../..
pip install -e external/torchani
```

## Quick energy + forces

```python
from animm.ani import load_ani_model, ani_energy_forces
from animm.convert import smiles_to_ase

atoms = smiles_to_ase("CCO")
model = load_ani_model("ANI2DR")
res = ani_energy_forces(model, atoms)
print(res.energy.item(), res.forces.shape)
```

## Alanine example (vacuum)

Basic run:

```bash
ani-mm ala2-md --steps 1000 --t 300 --dt 2.0 --report 50
```

Write a DCD trajectory:

```bash
ani-mm ala2-md --steps 1000 --dcd traj.dcd
```

Live viewer (auto backend) and keep window open:

```bash
ani-mm ala2-md --steps 500 --live-view --live-backend auto --live-hold
```

Force Matplotlib viewer:

```bash
ani-mm ala2-md --steps 500 --live-view --live-backend mpl --live-hold
```

Force ASE GUI (falls back with a diagnostic if Tk/Qt missing):

```bash
ani-mm ala2-md --steps 500 --live-view --live-backend ase
```

## Live viewer quick facts

* `auto` prefers ASE GUI, else Matplotlib, else disables silently (CI/headless).
* Matplotlib colors by element (approx JMol palette + categorical fallback).
* `--live-hold` only matters for Matplotlib (ASE GUI already persists).
* ASE fallback messages include a hint if `tkinter` is missing.

More details: `docs/live-viewer.md`.

## Precision & tracing cache

1. Attempt TorchScript trace in float64.
2. On dtype mismatch, warn and retry float32.
3. Cache by `(MODEL, NATOMS, DTYPE)`.
4. Actual traced dtype recorded on the force as `_animm_traced_dtype`.

Manual cache clear:

```python
from animm.ani_openmm import clear_traced_cache
clear_traced_cache()
```

## Warning suppression

Filters a few recurring lines (legacy `simtk.openmm` deprecation, experimental model notices). Remove the filters or set `PYTHONWARNINGS=default` to restore noise.

## Limitations (current)

* Vacuum only (no cutoff / periodic handling).
* No restart or checkpoint logic.
* Minimal reporting (stdout lines) in the example runner.
* Single potential per system.

## Roadmap (abridged)

| Status | Item |
| ------ | ---- |
| ✅ | Energy/forces helpers |
| ✅ | OpenMM TorchForce integration |
| ✅ | Alanine MD example |
| ✅ | Float64 trace + float32 fallback |
| ✅ | Live viewer (ASE/Matplotlib) + `--live-hold` |
| ⏳ | Periodic / cutoff support |
| ⏳ | Rich reporters & logging |
| ⏳ | Mixed precision knobs |
| ⏳ | Restart / checkpoint |
| ⏳ | More examples |

## Tests

```bash
pytest -q
```

Coverage:

```bash
pytest --cov=animm --cov-report=term-missing
```

## Dev loop

```bash
ruff check .
black --check .
isort --check-only .
pytest -q
mkdocs serve
```

## Contributing

Small, focused PRs welcome. Include/update tests if you change behavior; minimize unrelated formatting churn.

## License

MIT

---
If this saved you hand‑wiring a TorchForce, mission accomplished. 🚀

Small, practical helpers for using TorchANI models with OpenMM (and ASE when
needed). The intent is to stay compact: load a model, get energies/forces, wire
it into an OpenMM `TorchForce`, and run a quick MD experiment (alanine
dipeptide, your own molecule, etc.) without digging through a pile of boilerplate.

The 0.2.x series has mainly been exercised on macOS (Apple Silicon) and Linux.
Windows hasn’t been tried yet.

## What this project actually does

- Turn a SMILES string into an ASE `Atoms` (RDKit when available, a tiny fallback otherwise).
- Load a pretrained TorchANI model by name (currently `ANI2DR`, `ANI2X`).
- Ask for a single‑point energy and forces (double precision first; drop to float32 only if tracing forces us to).
- Build a single OpenMM `TorchForce` around that model so every MD step gets neural network forces.
- Provide a “just run it” alanine dipeptide vacuum simulation (OpenMM first, ASE if the plugin isn’t there).
- Offer a generic `run_ani_md` function with optional in‑memory coordinates or a DCD on disk.
- (New) Optional lightweight live viewer window (ASE GUI or Matplotlib) if you want to watch atoms drift while it runs.

## Things not done yet (but on the radar)

- Periodic boundary / cutoff aware variants (waiting on appropriate model + a clean interface).
- Mixed precision exploration (make it a toggle instead of secret fallback).
- A richer pile of reporters (CSV / JSON logs, progress bars, checkpointing).
- Restart support.
- Additional small examples beyond alanine.

## Install (development path)

TorchANI is vendored as a git submodule. The simplest route is “clone with
submodules, create environment, editable install.” That’s it.

### Quick start

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda env create -f environment.yml
conda activate ani-mm
pip install -e .
```

`environment.yml` already does an editable install of the TorchANI submodule,
so the last line only installs this project. Apple Silicon users get Metal
acceleration “for free” through conda‑forge’s PyTorch builds.

### Manual / custom install (if you’d rather not use the provided env)

Conda variant:

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda create -n ani-mm python=3.10 -y
conda activate ani-mm
conda install -c conda-forge openmm=8.0.0 openmm-torch ase rdkit pytorch -y
pip install -e external/torchani
pip install -e .
```

Pure pip variant (needs working wheels / compiler):

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openmm openmm-torch ase torch rdkit-pypi
pip install -e external/torchani
pip install -e .
```

### Updating the TorchANI submodule

Pull the latest TorchANI main:

```bash
git submodule update --remote external/torchani
pip install -e external/torchani
```

Pin a specific TorchANI commit if you need to reproduce results:

```bash
cd external/torchani
git checkout <COMMIT_SHA>
cd ../..
pip install -e external/torchani
```

If you later switch PyTorch builds (e.g. CUDA vs CPU) just reinstall the
vendored TorchANI editable again afterwards.

## Quick energy / forces example

```python
from animm.ani import load_ani_model
from animm.convert import smiles_to_ase

atoms = smiles_to_ase("CCO")  # ethanol
model = load_ani_model("ANI2DR")  # or ANI2X
energy = model.get_potential_energy(atoms)
forces = model.atoms.calc.results["forces"]  # Hartree/Bohr
print("Energy (Hartree):", energy)
print("Forces shape:", forces.shape)
```

### Run a short alanine dipeptide MD (vacuum)

```bash
ani-mm ala2-md --steps 1000 --t 300 --dt 2.0 --platform CPU
```

Add `--dcd traj.dcd` to also save a DCD (OpenMM / TorchForce path only).

Behind the scenes it (1) builds an OpenMM `System` with a single neural network
`TorchForce` (preferring a float64 trace, retrying in float32 only if truly
required), or (2) quietly drops back to a tiny ASE velocity Verlet loop if the
plugin is missing.

Current limitations:

- Vacuum only (no periodic boundary conditions yet).
- No restart / checkpoint.
- Alanine example writes a CSV‑ish line per report straight to stdout for now.
- Mixed precision is a future knob; the trace dtype sits on the force object as `_animm_traced_dtype` today.

### About the warning noise you *don’t* see

The package suppresses a few known noisy warnings (via `warnings.filterwarnings` and selective stderr redirection during critical imports):

- Deprecated `simtk.openmm` import notices (emitted by third-party legacy compatibility imports).
- Experimental TorchANI model (ANI-2xr) user warning.
- `pkg_resources` deprecation warning triggered during TorchANI import.

If you want everything back, run with `PYTHONWARNINGS=default` or delete those
filters locally.

### Precision & caching

Tracing starts in float64 for stability; if TorchScript complains about dtype
mismatches we fall back to float32, logging the reason. The traced module is
cached per `(MODEL, NATOMS, DTYPE)` so repeated runs don’t re‑trace every time.
Clear it manually if you like:

```python
from animm.ani_openmm import clear_traced_cache
clear_traced_cache()
```

to manually flush the cache.

## Roadmap snapshot

- [x] Basic ANI energy/force evaluation wrapper
- [x] ASE <-> OpenMM conversion helpers
- [x] OpenMM TorchForce integration (ANI-only)
- [x] Alanine dipeptide vacuum MD example
- [x] Warning suppression & model name normalization
- [x] Double precision default + float32 fallback
- [x] Generic MD runner (`run_ani_md`) with optional in-memory trajectory
- [ ] Rich reporter & logging framework (CSV/JSON, progress bars)
- [ ] Periodic system support (when suitable models available)
- [ ] Mixed precision / performance tuning knobs
- [ ] CI matrix for (CPU / GPU) and float32/float64 tracing
- [ ] Checkpoint / restart support

## Tests

The test suite here supplements TorchANI’s upstream tests (vendored in
`external/torchani/tests`). Local coverage touches models, energy/forces,
conversion, the OpenMM runner (including cache + dtype fallback), alanine
helper, dtype attribute, species tensor mapping, CLI modes, and the optional
live viewer.

Run the suite:

```bash
pytest -q
```

Coverage report:

```bash
pytest --cov=animm --cov-report=term-missing
```

## License

MIT

## Contributing

Short version: open an issue or PR, keep the diff focused, include/update
tests if behavior changes, and run the linters before pushing.

Setup (if you haven’t already):

```bash
conda env create -f environment.yml  # or your own env
conda activate ani-mm
pip install -e .[dev]
```

Linters & tests:

```bash
ruff check .
black --check .
isort --check-only .
pytest -q
```

Docs (MkDocs) locally:

```bash
pip install -e .[docs]
mkdocs serve
```

Please don’t commit the built `site/` folder; let CI or a release action build
public docs.
