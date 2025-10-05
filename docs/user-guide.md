# User Guide

This guide walks through common tasks using ani-mm.

## 1. Listing Available Models

```bash
python -m animm.cli models
```

## 2. Evaluating Energy / Forces

You can pass either a SMILES string or a path to a structure file supported by ASE (e.g. PDB, XYZ):

```bash
python -m animm.cli eval --smiles "CCO" --model ani2x
```

To include forces in the JSON output:

```bash
python -m animm.cli eval --smiles "CCO" --model ani2x --json
```

Programmatic form:

```python
from animm.ani import load_model
from animm.convert import smiles_to_ase

atoms = smiles_to_ase("CCO")
model = load_model("ani2x")
energy, forces = model.energy_forces(atoms)
```

## 3. Running a Short MD Simulation

The `ala2-md` subcommand runs an implicit environment alanine dipeptide simulation using OpenMM+TorchForce:

```bash
python -m animm.cli ala2-md --nsteps 200 --platform CUDA --json
```

Key parameters:

- `--nsteps`: number of integration steps.
- `--timestep`: integration timestep (fs).
- `--platform`: OpenMM platform (CUDA, OpenCL, CPU, etc.).
- `--temperature`: thermostat target (K).

## 4. Species Tensor Internals

Internally, atomic numbers are mapped directly to TorchANI species indices. This removes dependence on global ordering and keeps the interface stable.

## 5. TorchScript Tracing Cache

Models are traced the first time they are used with a specific (model name, number of atoms, dtype) triple. If float64 tracing fails, the code retries with float32 and records the effective dtype in `_animm_traced_dtype` for provenance.

## 6. Reusing the MD Runner Programmatically

```python
from animm.openmm_runner import run_ani_md
from animm.alanine_dipeptide import load_alanine_dipeptide

pdb, system = load_alanine_dipeptide(model_name="ani2x")
result = run_ani_md(system, pdb.topology, pdb.positions, n_steps=500)
print(result.n_steps, result.traced_dtype)
```

## 7. Converting SMILES to ASE Atoms

```python
from animm.convert import smiles_to_ase
atoms = smiles_to_ase("CCO")
```

RDKit is preferred when installed; a fallback builder is used otherwise.

## 8. Troubleshooting

- Import errors referencing `simtk` are legacy noise; warnings are filtered where practical.
- If CUDA is not available, OpenMM will silently fall back to CPU unless you force a platform.
- Force shape should be (N, 3). If you see dtype mismatches, inspect `result.traced_dtype` or the force object's `_animm_traced_dtype` attribute.

## 9. Next Steps

See the API Reference for deeper details and the Development page if you plan to contribute.

## 10. Live Desktop Viewer (Experimental)

You can open a lightweight live viewer window during MD:

Command line (alanine example):

```bash
python -m animm.cli ala2-md --steps 1000 --live-view --live-backend auto
```

Programmatic (generic MD runner):

```python
from animm.convert import smiles_to_ase
from animm.openmm_runner import run_ani_md

atoms = smiles_to_ase("CCO")
res = run_ani_md(atoms, n_steps=500, live_view=True, live_interval=50)
```

Backends:
* auto – prefer ASE GUI if available, fallback to Matplotlib.
* ase – force ASE GUI.
* mpl – force Matplotlib scatter.

If no GUI backend is available (e.g. headless CI), it silently disables itself.
