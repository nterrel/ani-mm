# ani-mm

Utilities for running ANI neural network potentials with OpenMM via the TorchForce plugin, plus convenience helpers for model evaluation, simple molecular dynamics, and conversion utilities.

## Highlights

- Minimal surface area: focuses on ANI models only.
- TorchScript tracing cache with automatic dtype fallback (float64 -> float32) for stability.
- Simple molecular dynamics runner that reports energies, temperature, and supports optional trajectory capture.
- Clean CLI for quick energy/forces evaluation, model listing, and an alanine dipeptide example run.

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

## Documentation Structure

- User Guide: practical usage patterns and CLI examples.
- API Reference: auto-generated documentation of the public Python API.
- Development: contributor workflow, testing, and release notes.

See the roadmap in the project README for planned enhancements.
