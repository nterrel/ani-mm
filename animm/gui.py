"""Lightweight desktop live viewers for ANI/OpenMM simulations.

Backends supported (best‑effort, optional):

* ``ase`` – ASE GUI (if ``ase.gui`` is importable); updates an ``Atoms`` object.
* ``mpl`` – Matplotlib 3D scatter (if matplotlib available).

Selection logic:
``backend='auto'`` prefers ASE (often more interactive for molecular data) and
falls back to matplotlib. Both fall back to a disabled no‑op viewer if neither
dependency or a suitable interactive backend is available (e.g. headless CI).

The viewer is intentionally minimal and only intended for quick visual sanity
checks; it is not intended to be a full analysis or high‑performance rendering
tool.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np

try:  # Lazy / optional dependency (matplotlib)
    import matplotlib
    # Try to use an interactive backend; if already set and non-interactive, leave it.
    if matplotlib.get_backend().lower() in {"agg", "pdf", "svg"}:
        try:  # pragma: no cover - backend selection is environment dependent
            matplotlib.use("TkAgg", force=False)  # noqa: E402
        except Exception:  # pragma: no cover
            pass
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - headless or matplotlib missing
    _HAVE_MPL = False
    plt = None  # type: ignore

try:  # ASE GUI optional dependency
    from ase import Atoms as _ASEAtoms  # type: ignore
    from ase.gui.gui import GUI as _ASEGUI  # type: ignore
    _HAVE_ASE_GUI = True
except Exception:  # pragma: no cover - missing ASE or GUI toolkit
    _HAVE_ASE_GUI = False
    _ASEAtoms = None  # type: ignore
    _ASEGUI = None  # type: ignore


class LiveTrajectoryViewer:
    """Simple 3D point cloud viewer for atom positions.

    Parameters
    ----------
    symbols : list[str]
        Chemical symbols to optionally color / label (currently just count based).
    title : str
        Window title.
    max_atoms : int | None
        Optional sanity limit; if exceeded, disables live drawing to avoid freezes.
    """

    def __init__(self, symbols, title: str = "ani-mm Live MD", max_atoms: Optional[int] = 5000):
        self.enabled = _HAVE_MPL
        self._disabled_reason: Optional[str] = None
        self.symbols = symbols
        if not self.enabled:
            self._disabled_reason = "matplotlib not available"
            return
        if max_atoms is not None and len(symbols) > max_atoms:
            self.enabled = False
            self._disabled_reason = f"atom count {len(symbols)} exceeds max_atoms={max_atoms}"
            return
        plt.ion()  # type: ignore[attr-defined]
        self.fig = plt.figure(figsize=(5, 5))  # type: ignore[call-arg]
        try:
            self.fig.canvas.manager.set_window_title(title)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        # placeholder scatter (3D scatter; ignore static type checker complaints.)
        self._scatter = self.ax.scatter([], [], [], s=40, depthshade=True)  # type: ignore[arg-type]
        self.ax.set_xlabel("X (Å)")
        self.ax.set_ylabel("Y (Å)")
        self.ax.set_zlabel("Z (Å)")
        self._frame_txt = self.ax.text2D(0.02, 0.95, "step: -", transform=self.ax.transAxes)

    def update(self, positions_ang: np.ndarray, step: int):  # (N,3)
        if not self.enabled:
            return
        if positions_ang.ndim != 2 or positions_ang.shape[1] != 3:
            return
        x, y, z = positions_ang[:, 0], positions_ang[:, 1], positions_ang[:, 2]
        # Update scatter (re-create for robustness; set_offsets not in 3D)
        self._scatter.remove()
        self._scatter = self.ax.scatter(x, y, z, s=40, depthshade=True)  # type: ignore[arg-type]
        self._frame_txt.set_text(f"step: {step}")
        # Autoscale gently
        pad = 1.5
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        zmin, zmax = float(z.min()), float(z.max())
        self.ax.set_xlim(xmin - pad, xmax + pad)
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.set_zlim(zmin - pad, zmax + pad)
        plt.pause(0.001)  # type: ignore[attr-defined]

    def finalize(self):
        if not self.enabled:
            return
        try:
            plt.ioff()  # type: ignore[attr-defined]
            self.fig.canvas.draw_idle()
        except Exception:  # pragma: no cover
            pass
class _ASEAtomsViewer:
    """Wrap ASE GUI viewer for live updating.

    We hold an ASE ``Atoms`` instance and update its positions (Å). The ASE GUI
    automatically reflects coordinate changes.
    """

    def __init__(self, symbols, initial_positions_ang: np.ndarray):
        self.enabled = _HAVE_ASE_GUI
        self._disabled_reason: Optional[str] = None
        if not self.enabled:
            self._disabled_reason = "ASE GUI not available"
            return
        self.atoms = _ASEAtoms(symbols=symbols, positions=initial_positions_ang)  # type: ignore
        try:
            self.gui = _ASEGUI(self.atoms)  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - GUI backend issues
            self.enabled = False
            self._disabled_reason = "failed to initialize ASE GUI"
            return

    def update(self, positions_ang: np.ndarray, step: int):  # noqa: D401
        if not self.enabled:
            return
        try:
            self.atoms.set_positions(positions_ang)
        except Exception:  # pragma: no cover
            pass

    def finalize(self):  # noqa: D401
        # ASE GUI keeps window alive; nothing to finalize.
        return

    def status(self) -> str:  # type: ignore[override]
        if self.enabled:
            return "active"
        return f"disabled: {self._disabled_reason}" if self._disabled_reason else "disabled"


class _LiveViewerReporter:
    """OpenMM Reporter that forwards positions to a ``LiveTrajectoryViewer``.

    Positions are converted from nm to Å inside ``report``.
    """

    def __init__(self, viewer: Any, interval: int):
        self.viewer = viewer
        self._interval = max(1, int(interval))
        self._step = 0

    def describeNextReport(self, simulation):  # noqa: N802
        return (self._interval, True, False, False, False)

    def report(self, simulation, state):  # noqa: N802
        self._step = simulation.currentStep
        if not self.viewer.enabled:
            return
        import openmm.unit as unit  # local import to avoid hard dependency earlier

        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # type: ignore[attr-defined]
        self.viewer.update(np.asarray(pos_nm) * 10.0, self._step)  # nm -> Å


def build_live_viewer_reporter(symbols, interval: int = 100, backend: str = "auto", initial_positions_ang: Optional[np.ndarray] = None) -> Tuple[object, object]:
    """Factory returning (viewer, reporter) for the selected backend.

    Parameters
    ----------
    symbols : list[str]
        Chemical symbols.
    interval : int
        Reporting interval (steps).
    backend : str
        One of 'auto', 'ase', 'mpl'.
    initial_positions_ang : np.ndarray | None
        Optional initial positions (Å) required for ASE backend.
    """
    chosen = backend
    viewer: object
    if backend == "auto":
        if _HAVE_ASE_GUI and initial_positions_ang is not None:
            chosen = "ase"
        elif _HAVE_MPL:
            chosen = "mpl"
        else:
            chosen = "none"
    if chosen == "ase" and initial_positions_ang is not None and _HAVE_ASE_GUI:
        viewer = _ASEAtomsViewer(symbols, initial_positions_ang)
    elif chosen == "mpl" and _HAVE_MPL:
        viewer = LiveTrajectoryViewer(symbols)
    else:  # disabled no-op
        class _DisabledViewer:
            enabled = False

            def update(self, *_, **__):
                return

            def finalize(self):
                return

            def status(self):
                return "disabled"
        viewer = _DisabledViewer()
    reporter = _LiveViewerReporter(viewer, interval=interval)
    return viewer, reporter


__all__ = ["LiveTrajectoryViewer", "build_live_viewer_reporter"]
