# ani-mm

TorchANI + OpenMM molecular modeling utilities.
Current 0.2.x line targets macOS (Apple Silicon) primarily; Linux should work with
the same dependency stack. Windows is untested.

## Goals

Provide small, composable helpers to:

1. Convert a SMILES (via RDKit) or simple name to an ASE `Atoms` object.
2. Load a TorchANI pretrained model (currently ANI2DR, ANI2X) with a stable, case-insensitive API.
3. Evaluate ANI energies & forces (ASE interface) in double precision by default.
4. Construct an OpenMM `TorchForce` wrapping an ANI model (via `openmm-torch`).
5. Run a minimal alanine dipeptide vacuum MD example (OpenMM when TorchForce available, else ASE fallback).
6. Generic reusable `run_ani_md` function for arbitrary ASE `Atoms` with optional in-memory trajectory capture and DCD output.

Planned (not implemented yet):

* Periodic boundary & cutoff-aware variants (when suitable TorchANI models / parameters are available).
* Mixed precision / performance tuning switches (explicit autocast, optional half precision on supported hardware).
* Richer reporter composition (JSON/CSV loggers, force/temperature statistics, checkpointing).

## Install (development)

TorchANI is vendored as a git submodule. The recommended (minimal) workflow installs everything—TorchANI included—via the Conda environment file and then installs this package once.

### Quick Start (recommended)

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda env create -f environment.yml
conda activate ani-mm
pip install -e .
```

That’s it. The `environment.yml` already performs an editable install of the submodule (`external/torchani`), so the final `pip install -e .` only installs `ani-mm` itself.

Apple Silicon: conda-forge PyTorch includes Metal acceleration automatically.

### Manual / Custom Install (if you don’t want to use environment.yml)

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

Pure pip variant (needs a working compiler & wheels availability):

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

### Updating TorchANI Submodule

Pull the latest TorchANI main:

```bash
git submodule update --remote external/torchani
pip install -e external/torchani
```

Pin a specific TorchANI commit:

```bash
cd external/torchani
git checkout <COMMIT_SHA>
cd ../..
pip install -e external/torchani
```

Torch / CUDA: Install the appropriate torch build for your hardware following <https://pytorch.org>. Re-run the editable TorchANI install afterward if you replace PyTorch.

## Quick Example (Energy & Forces)

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

Add `--dcd traj.dcd` to write a DCD trajectory (only when using OpenMM TorchForce engine).

The helper will:

* Try to build an OpenMM `System` with a single `TorchForce` (ANI model traced to TorchScript in double precision; falls back to float32 if tracing fails).
* Fall back to an ASE velocity Verlet integrator if `openmm-torch` is not importable.

Limitations (0.2.x):

* No periodic systems / cutoffs yet (vacuum / gas‑phase only).
* No restart / checkpoint support.
* Alanine helper prints directly to stdout (potential / Δ / T). Structured logging pending.
* Actual traced dtype is attached to the produced ``TorchForce`` (``_animm_traced_dtype``); mixed precision & half precision still future work.

### Warning Suppression

The package suppresses a few known noisy warnings (via `warnings.filterwarnings` and selective stderr redirection during critical imports):

* Deprecated `simtk.openmm` import notices (emitted by third-party legacy compatibility imports).
* Experimental TorchANI model (ANI-2xr) user warning.
* `pkg_resources` deprecation warning triggered during TorchANI import.

Re-enable them with: `PYTHONWARNINGS=default` or by editing the filters in `animm/__init__.py` and `animm/cli.py`.

### Precision

Tracing prefers float64 (double) for improved force fidelity. If TorchScript tracing fails due to a dtype mismatch in the current environment, it logs a warning and retries in float32.

### Caching

Traced TorchScript modules for a given (MODEL, NATOMS, DTYPE) triplet are cached in-process to avoid repeated tracing overhead. Use:

```python
from animm.ani_openmm import clear_traced_cache
clear_traced_cache()
```

to manually flush the cache.

## Roadmap (high-level)

* [x] Basic ANI energy/force evaluation wrapper
* [x] ASE <-> OpenMM conversion helpers
* [x] OpenMM TorchForce integration (ANI-only)
* [x] Alanine dipeptide vacuum MD example
* [x] Warning suppression & model name normalization
* [x] Double precision default + float32 fallback
* [x] Generic MD runner (`run_ani_md`) with optional in-memory trajectory
* [ ] Rich reporter & logging framework (CSV/JSON, progress bars)
* [ ] Periodic system support (when suitable models available)
* [ ] Mixed precision / performance tuning knobs
* [ ] CI matrix for (CPU / GPU) and float32/float64 tracing
* [ ] Checkpoint / restart support

## Tests

Project tests (`tests/`) complement the upstream TorchANI suite (vendored under
`external/torchani/tests`). Implemented coverage includes:

* Model listing & case handling (`test_models.py`).
* Core energy/force wrapper (`test_ani_energy_forces.py`).
* SMILES → ASE conversion and error paths (`test_convert.py`).
* OpenMM runner functionality, cache reuse, dtype fallback (`test_run_ani_md.py`).
* Alanine helper smoke test (`test_alanine_helper.py`).
* Traced module dtype attribute (`test_traced_dtype.py`).
* Species tensor atomic numbers (`test_species_tensor.py`).
* CLI JSON/text modes (`test_cli.py`).

Run the suite:

```bash
pytest -q
```

Generate a coverage report:

```bash
pytest --cov=animm --cov-report=term-missing
```

## License

MIT

## Contributing

Contributions are welcome. Please keep changes small and focused; open an issue first for larger feature ideas.

Guidelines:

1. Fork / branch: create a feature branch off `main`.
2. Tests: add or update tests for any user‑visible behavior change (see `tests/`).
3. Style: run linters/formatters (`ruff`, `black`, `isort`) before submitting. (The dev extra installs them.)
4. Commit messages: concise, present tense (e.g. "add dtype attribute to TorchForce").
5. Pull request: include a short rationale and any performance / regression notes.

Development setup:

```bash
conda env create -f environment.yml  # or your own env
conda activate ani-mm
pip install -e .[dev]
```

Run linters & tests:

```bash
ruff check .
black --check .
isort --check-only .
pytest -q
```

Docs (MkDocs) preview locally:

```bash
pip install -e .[docs]
mkdocs serve
```

Please do not commit rendered documentation (`site/`); CI can build it.
