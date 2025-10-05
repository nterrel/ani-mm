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

Planned (not fully implemented yet):

* True per-step force recomputation in a reusable dynamics runner (the current alanine helper does, but the generic `openmm_runner` is still a placeholder).
* Reporting & trajectory output wiring earlier in the simulation lifecycle.
* Periodic boundary & cutoff-aware variants (when appropriate TorchANI models available).

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

* The generic `openmm_runner.minimize_and_md` is a stub (does not yet perform time integration with force updates). Use the CLI alanine example for real dynamics until the runner is implemented.
* No periodic systems / cutoffs.
* No long trajectory conveniences (checkpointing, logging) yet.
* Reporter attachment order in alanine helper is simplified; early reporters will be added in a future revision.

Planned next: implement a reusable MD driver that reuses the traced ANI module each step and supports optional mixed precision.

### Warning Suppression

The package suppresses a few known noisy warnings:

* Deprecated `simtk.openmm` import notices.
* Experimental TorchANI model (ANI-2xr) user warning.
* `pkg_resources` deprecation warning triggered during TorchANI import.

You can re-enable them by running with Python's warnings reset (e.g. `PYTHONWARNINGS=default`).

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
* [x] ASE <-> OpenMM conversion helpers (initial)
* [x] OpenMM TorchForce integration (ANI-only)
* [x] Alanine dipeptide vacuum MD example
* [x] Warning suppression & model name normalization
* [x] Double precision default + float32 fallback
* [ ] Generic MD runner (force updates each step) replacing placeholder
* [ ] Proper reporters & trajectory outputs for arbitrary systems
* [ ] Periodic system support (when suitable models available)
* [ ] Mixed precision / performance tuning knobs
* [ ] CI matrix for (CPU / GPU) and float32/float64 tracing

## License

MIT
