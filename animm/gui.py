"""Lightweight desktop live viewer for ANI/OpenMM simulations.

This module provides a minimal Matplotlib-based 3D scatter viewer that can be
updated incrementally during an OpenMM simulation via a custom reporter.

Design goals:
* Zero impact unless explicitly enabled (lazy import of matplotlib).
* Graceful fallback if a GUI backend is unavailable (no-op warnings).
* Keep update cost low: only a scatter set_offsets + pause.

Public entry point: :class:`LiveTrajectoryViewer` and helper
``build_live_viewer_reporter`` returning an OpenMM Reporter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # Lazy / optional dependency
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
        plt.ion()
        self.fig = plt.figure(figsize=(5, 5))
        try:
            self.fig.canvas.manager.set_window_title(title)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title(title)
        # placeholder scatter
        self._scatter = self.ax.scatter([], [], [], s=40, depthshade=True)
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
        self._scatter = self.ax.scatter(x, y, z, s=40, depthshade=True)
        self._frame_txt.set_text(f"step: {step}")
        # Autoscale gently
        pad = 1.5
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        zmin, zmax = float(z.min()), float(z.max())
        self.ax.set_xlim(xmin - pad, xmax + pad)
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.set_zlim(zmin - pad, zmax + pad)
        plt.pause(0.001)

    def finalize(self):
        if not self.enabled:
            return
        try:
            plt.ioff()
            self.fig.canvas.draw_idle()
        except Exception:  # pragma: no cover
            pass

    def status(self) -> str:
        if self.enabled:
            return "active"
        return f"disabled: {self._disabled_reason}" if self._disabled_reason else "disabled"


class _LiveViewerReporter:
    """OpenMM Reporter that forwards positions to a ``LiveTrajectoryViewer``.

    Positions are converted from nm to Å inside ``report``.
    """

    def __init__(self, viewer: LiveTrajectoryViewer, interval: int):
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


def build_live_viewer_reporter(symbols, interval: int = 100):
    """Factory returning (viewer, reporter).

    The caller should append the reporter to ``simulation.reporters`` and may
    also call ``viewer.update`` manually for an initial frame if desired.
    """
    viewer = LiveTrajectoryViewer(symbols)
    reporter = _LiveViewerReporter(viewer, interval=interval)
    return viewer, reporter


__all__ = [
    "LiveTrajectoryViewer",
    "build_live_viewer_reporter",
]
