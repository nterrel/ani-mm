# ani-mm

Hybrid ANI + OpenMM molecular modelling utilities.
Current release only optimized for MacOS running on Apple Silicon.

## Goals

Provide simple utilities to:

1. Load a molecule (SMILES, XYZ, ASE object) and generate a 3D structure.
2. Evaluate ANI (TorchANI) energies / forces.
3. Embed ANI forces into an OpenMM System for hybrid or pure ML MD.
4. Run short MD trajectories and export to common formats (XYZ, PDB, ASE Trajectory).

## Install (development)

### Option A: Conda (recommended for OpenMM + TorchANI)

```bash
conda create -n ani-mm python=3.10 -y
conda activate ani-mm
conda install -c conda-forge openmm=8.0.0 ase rdkit pytorch torchani -y
pip install -e .[dev,ml]
```

For GPU (CUDA 12 example):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Option B: Pure pip (ensure you have a compatible C++ toolchain for OpenMM wheels)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openmm ase torch torchani rdkit-pypi
pip install -e .[dev,ml]
### Using bleeding-edge TorchANI (GitHub main)

If you need features or models newer than the packaged release:

Use the `ml-git` extra for bleeding-edge (GitHub main) instead of the packaged release:

```bash
pip install -e .[dev,ml-git]
```

Pin a specific commit for reproducibility (replace `SHA`):

```bash
pip install "torchani @ git+https://github.com/aiqm/torchani.git@SHA"
```

For local patching you can still clone manually:

```bash
git submodule add https://github.com/aiqm/torchani external/torchani
pip install -e external/torchani
```

```

Torch / CUDA: Install the appropriate torch build for your hardware following <https://pytorch.org> before (or after) installing `torchani` if you want GPU acceleration.

## Quick Example

```python
from animm.ani import load_ani_model
from animm.convert import smiles_to_ase

atoms = smiles_to_ase("CCO")  # ethanol
model = load_ani_model()  # default ANI2x
ener = model.get_potential_energy(atoms)
print("Energy (Hartree):", ener)
```

### Run a short alanine dipeptide MD

```bash
ani-mm ala2-md --steps 1000 --t 300 --dt 2.0 --platform CPU
```

Add `--dcd traj.dcd` to write a trajectory.

## Roadmap

- [ ] Basic ANI energy/force evaluation wrapper
- [ ] ASE <-> OpenMM conversion helpers
- [ ] OpenMM custom force integrating ANI
- [ ] Minimization + Langevin dynamics convenience function
- [x] CLI entry point
- [x] Alanine dipeptide vacuum MD example

## License

MIT
