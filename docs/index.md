# ani-mm

Lean helpers for embedding TorchANI ANI models (`ANI2DR`, `ANI2X`) into OpenMM via `TorchForce`, with SMILES conversion, single‑point evaluation, a minimal MD runner, and an optional live viewer.

## Highlights

* Float64‑first TorchScript tracing (float32 fallback) with cache
* Provenance metadata (`_animm_traced_dtype`, `_animm_cache_hit` + SUMMARY line)
* Alanine dipeptide vacuum example (`ala2-md`)
* Live viewer (ASE GUI → Matplotlib)

## Install (dev/source)

```bash
git clone --recurse-submodules https://github.com/nterrel/ani-mm.git
cd ani-mm
conda env create -f environment.yml
conda activate ani-mm
pip install -e .[docs]
```

## Quick CLI

```bash
ani-mm models
ani-mm eval CCO --model ANI2X --json
ani-mm ala2-md --steps 300 --report 50 --debug
```

## Python Snippet

```python
from animm.convert import smiles_to_ase
from animm.ani import load_ani_model, ani_energy_forces

atoms = smiles_to_ase("CCO")
calc = load_ani_model("ANI2DR")
ev = ani_energy_forces(calc, atoms)
print(ev.energy.item(), ev.forces.shape)
```

## Where Next

* User Guide: detailed recipes & provenance
* Live Viewer: backend details & caveats
* Development: roadmap & contribution guide
* API Reference: function docs (mkdocstrings)

For a very brief overview see the top-level README. Advanced install notes, roadmap, and provenance internals live here in the docs.
