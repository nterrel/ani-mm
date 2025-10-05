# ani-mm

Utilities for running ANI neural network potentials with OpenMM (TorchForce) plus small helpers for model evaluation, SMILES conversion, and a minimal MD workflow.

## Highlights

* Lean scope: ANI models only (currently `ANI2DR`, `ANI2X`).
* TorchScript tracing cache (float64 preferred, automatic float32 fallback).
* Lightweight MD runner (+ optional in‑memory trajectory, DCD output).
* Alanine dipeptide vacuum example with delta energy reporter.
* Experimental live viewer (ASE GUI or Matplotlib) with `--live-hold`.

## Install

```bash
pip install .
# or with docs extras
pip install .[docs]
```

From source with an editable install:

```bash
pip install -e .[docs]
```

## Quick Start

Evaluate ANI energy and forces for a simple SMILES string:

```bash
python -m animm.cli eval --smiles "CCO"
```

Run a short alanine dipeptide MD example with JSON output:

```bash
python -m animm.cli ala2-md --nsteps 200 --json
```

Programmatic usage:

```python
from animm.ani import load_model
from animm.convert import smiles_to_ase

atoms = smiles_to_ase("CCO")
model = load_model("ani2x")
energy, forces = model.energy_forces(atoms)
print(energy.item(), forces.shape)
```

## Documentation Map

* User Guide – practical CLI & programmatic recipes.
* Live Viewer – backend selection, troubleshooting, hold flag.
* API Reference – auto-generated public API.
* Development – contributor workflow.

Roadmap and caveats live in the top-level README.
