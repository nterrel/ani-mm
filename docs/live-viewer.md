# Live Viewer (Experimental)

Watch positions in (near) real time while MD runs. Two backends:

* **ASE GUI (`ase`)** – higher fidelity 3D molecular view if ASE + a GUI toolkit (Tk/Qt) are available.
* **Matplotlib (`mpl`)** – lightweight 3D scatter (element‑colored) that works with standard interactive backends.

`backend=auto` tries ASE first, then falls back to Matplotlib, else disables silently (headless CI).

## CLI usage

```bash
ani-mm ala2-md --steps 500 --live-view --live-backend auto
```

Keep the Matplotlib window open after completion:

```bash
ani-mm ala2-md --steps 500 --live-view --live-backend mpl --live-hold
```

Force ASE GUI (will fall back with a diagnostic if the toolkit is missing):

```bash
ani-mm ala2-md --steps 300 --live-view --live-backend ase
```

## Programmatic usage

```python
from animm.openmm_runner import run_ani_md
from animm.convert import smiles_to_ase

atoms = smiles_to_ase("CCO")
res = run_ani_md(atoms, n_steps=400, live_view=True, live_interval=50)
# res.final_potential_kjmol etc.
```

## Element coloring (Matplotlib)

Common elements use approximate JMol palette; unknowns cycle a categorical set.

## The `--live-hold` flag

Only applies to Matplotlib: blocks at the end with a standard `plt.show()` so you can inspect the final frame. ASE GUI windows persist naturally.

## ASE fallback reasons

If ASE cannot start, a message like:

```text
[ani-mm] ASE backend unavailable (tkinter import failed (...)); falling back to matplotlib.
```

Typical fixes (macOS):

* Ensure Python has Tk support (Conda envs usually do). For pyenv builds, install `tcl-tk` via Homebrew and rebuild Python with the proper flags.
* Install a Qt alternative: `pip install pyqt6` (ASE can use it).

## Headless environments

CI or servers without a display will disable the viewer silently. This is expected; you can still run with `--live-view` and no error will be raised.

## Performance impact

The viewer reports only at its interval; choose a sparser interval (e.g. every 100–500 steps) for large systems.

## Known limitations

* No trajectory scrubbing or rewind.
* No bond rendering in Matplotlib mode (points only).
* ASE viewer may not reflect periodic images (vacuum focus).

## Future enhancements (candidates)

* Optional orthographic projection & simple camera controls for Matplotlib.
* Frame rate smoothing / decimation for very dense systems.
* Support for additional visualization backends if demand appears.

---
Feedback welcome – file an issue with your platform details if the viewer fails to launch.
