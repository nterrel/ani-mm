"""Optional live viewer backends (desktop) for running simulations.

Backends (best effort):
* ``ase`` – ASE GUI (preferred if present).
* ``mpl`` – quick Matplotlib 3D scatter.

``backend='auto'`` picks ASE first, then Matplotlib, else silently disables
itself (e.g. headless CI). This is intentionally simple: just enough to watch
atoms move while you debug.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, cast
import sys

import numpy as np


class LiveViewer(Protocol):  # pragma: no cover
    """Protocol for live viewer backends (duck typed)."""

    enabled: bool

    def update(self, positions_ang: np.ndarray, step: int) -> None:  # noqa: D401
        ...

    def finalize(self) -> None:  # noqa: D401
        ...

    def status(self) -> str:  # noqa: D401
        ...


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

_ASE_GUI_IMPORT_ERROR: Optional[str] = None
try:  # ASE GUI optional dependency
    from ase import Atoms as _ASEAtoms  # type: ignore
    from ase.gui.gui import GUI as _ASEGUI  # type: ignore
    # Quick probe: some environments have ase + tkinter missing; ensure tkinter available
    try:  # pragma: no cover - environment specific
        import tkinter  # noqa: F401
    except Exception as _tk_err:  # pragma: no cover
        # We still mark as available (ASE will raise when constructing GUI) but record hint
        _ASE_GUI_IMPORT_ERROR = f"tkinter import failed ({_tk_err.__class__.__name__}: {_tk_err})"
    _HAVE_ASE_GUI = True
except Exception as _ase_err:  # pragma: no cover - missing ASE or GUI toolkit
    _HAVE_ASE_GUI = False
    _ASE_GUI_IMPORT_ERROR = f"{_ase_err.__class__.__name__}: {_ase_err}".strip()
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

    def __init__(
        self,
        symbols,
        title: str = "ani-mm Live MD",
        max_atoms: Optional[int] = 5000,
        hold_open: bool = False,
    ):
        self.enabled = _HAVE_MPL
        self._disabled_reason: Optional[str] = None
        self.symbols = symbols
        self._hold_open = bool(hold_open)
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
        # Precompute per‑atom colors (simple palette keyed by element symbol)
        self._colors = self._build_colors(symbols)
        # Lazily create scatter on first update to avoid double work and keep colors stable
        self._scatter = None  # type: ignore
        self.ax.set_xlabel("X (Å)")
        self.ax.set_ylabel("Y (Å)")
        self.ax.set_zlabel("Z (Å)")
        self._frame_txt = self.ax.text2D(0.02, 0.95, "step: -", transform=self.ax.transAxes)

    # --- internal helpers -------------------------------------------------
    @staticmethod
    def _build_colors(symbols):
        """Return a stable list of matplotlib color specs for each atom symbol.

        Uses a small element->color mapping (roughly following JMol colors) and
        falls back to a categorical palette cycling if an element is unknown.
        """
        # Common element color mapping (hex or named)
        palette = {
            "H": "#FFFFFF",
            "C": "#909090",
            "N": "#3050F8",
            "O": "#FF0D0D",
            "F": "#90E050",
            "Cl": "#1FF01F",
            "P": "#FF8000",
            "S": "#FFFF30",
            "Br": "#A62929",
        }
        cycle = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        colors = []
        fallback_i = 0
        for sym in symbols:
            c = palette.get(sym)
            if c is None:
                c = cycle[fallback_i % len(cycle)]
                fallback_i += 1
            colors.append(c)
        return colors

    def update(self, positions_ang: np.ndarray, step: int):  # (N,3)
        if not self.enabled:
            return
        if positions_ang.ndim != 2 or positions_ang.shape[1] != 3:
            return
        x, y, z = positions_ang[:, 0], positions_ang[:, 1], positions_ang[:, 2]
        if self._scatter is None:
            # First frame: build scatter once with colors
            self._scatter = cast(
                Any,
                self.ax.scatter(  # type: ignore[call-arg]
                    x,
                    y,
                    z,  # type: ignore[arg-type]  # 3D scatter accepts array-like
                    s=40,
                    depthshade=True,
                    c=self._colors if len(self._colors) == len(x) else None,
                ),
            )
        else:
            # Fast path: mutate existing scatter artist for stable colors
            try:
                # 3D PathCollection stores data in _offsets3d tuple
                self._scatter._offsets3d = (x, y, z)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: recreate scatter (should be rare)
                try:
                    self._scatter.remove()
                except Exception:  # pragma: no cover
                    pass
                self._scatter = cast(
                    Any,
                    self.ax.scatter(  # type: ignore[call-arg]
                        x,
                        y,
                        z,  # type: ignore[arg-type]
                        s=40,
                        depthshade=True,
                        c=self._colors if len(self._colors) == len(x) else None,
                    ),
                )
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
            if self._hold_open:
                # Switch to blocking show so window remains until user closes it.
                plt.ioff()  # type: ignore[attr-defined]
                self.fig.canvas.draw()
                plt.show(block=True)  # type: ignore[attr-defined]
            else:
                plt.ioff()  # type: ignore[attr-defined]
                self.fig.canvas.draw_idle()
        except Exception:  # pragma: no cover
            pass

    def status(self) -> str:  # added to satisfy LiveViewer Protocol
        return "active" if self.enabled else "disabled"


class _ASEAtomsViewer:
    """Wrap ASE GUI viewer for live updating.

    We hold an ASE ``Atoms`` instance and update its positions (Å). The ASE GUI
    automatically reflects coordinate changes.
    """

    def __init__(self, symbols, initial_positions_ang: np.ndarray):
        self.enabled = _HAVE_ASE_GUI
        self._disabled_reason: Optional[str] = None
        if not self.enabled:
            self._disabled_reason = _ASE_GUI_IMPORT_ERROR or "ASE GUI not available"
            return
        self.atoms = _ASEAtoms(symbols=symbols, positions=initial_positions_ang)  # type: ignore
        try:  # pragma: no cover - GUI backend issues vary
            self.gui = _ASEGUI(self.atoms)  # type: ignore[call-arg]
            # Try to launch the GUI event loop in non-blocking mode if supported
            try:
                run_method = getattr(self.gui, "run", None)
                if callable(run_method):
                    try:
                        run_method(block=False)  # type: ignore[call-arg]
                    except TypeError:
                        # Older ASE may not support block kwarg; fall back to direct call (may block, so skip)
                        pass
            except Exception:
                pass
        except Exception:
            self.enabled = False
            self._disabled_reason = "failed to initialize ASE GUI"
            return

    def update(self, positions_ang: np.ndarray, step: int):  # noqa: D401
        if not self.enabled:
            return
        try:
            self.atoms.set_positions(positions_ang)
            # Force a redraw if available
            gui_draw = getattr(self.gui, "draw", None)
            if callable(gui_draw):  # pragma: no cover
                try:
                    gui_draw()
                except Exception:
                    pass
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

    def __init__(self, viewer: LiveViewer, interval: int):
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

        pos_nm = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer  # type: ignore[attr-defined]  # openmm unit symbols are dynamic
        )
        self.viewer.update(np.asarray(pos_nm) * 10.0, self._step)  # nm -> Å


def build_live_viewer_reporter(
    symbols,
    interval: int = 100,
    backend: str = "auto",
    initial_positions_ang: Optional[np.ndarray] = None,
    hold_open: bool = False,
) -> Tuple[LiveViewer, _LiveViewerReporter]:
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
    hold_open : bool
        If True and using the matplotlib backend, block at finalize so the
        window stays open after the simulation ends.
    """
    chosen = backend
    viewer: LiveViewer  # type: ignore[assignment]
    if backend == "auto":
        if _HAVE_ASE_GUI and initial_positions_ang is not None:
            chosen = "ase"
        elif _HAVE_MPL:
            chosen = "mpl"
        else:
            chosen = "none"
    # Reusable disabled placeholder adhering to Protocol
    
    class _DisabledViewerProto:
        enabled = False

        def update(self, *_, **__):
            return None

        def finalize(self):  # noqa: D401
            return None

        def status(self):  # noqa: D401
            return "disabled"

    if chosen == "ase" and initial_positions_ang is not None and _HAVE_ASE_GUI:
        viewer = _ASEAtomsViewer(symbols, initial_positions_ang)
        # If ASE failed to enable (e.g., GUI toolkit missing), fall back to mpl
        if not getattr(viewer, "enabled", False) and _HAVE_MPL:
            reason = getattr(viewer, "_disabled_reason", None) or _ASE_GUI_IMPORT_ERROR or "missing GUI toolkit"
            hint = "Install a Tk or Qt binding (e.g. 'conda install tk' or ensure python was built with Tk)"
            print(
                f"[ani-mm] ASE backend unavailable ({reason}); falling back to matplotlib.\n"
                f"         Hint: {hint}",
                file=sys.stderr,
            )
            viewer = LiveTrajectoryViewer(symbols, hold_open=hold_open)
    elif chosen == "ase":
        # Forced ASE but cannot proceed; attempt graceful fallback
        if _HAVE_MPL:
            reason = (
                "initial positions required" if initial_positions_ang is None else "ASE GUI missing"
            )
            print(
                f"[ani-mm] Requested ASE backend but {reason}; using matplotlib instead.",
                file=sys.stderr,
            )
            viewer = LiveTrajectoryViewer(symbols, hold_open=hold_open)
        else:
            viewer = _DisabledViewerProto()
    elif chosen == "mpl" and _HAVE_MPL:
        viewer = LiveTrajectoryViewer(symbols, hold_open=hold_open)
    else:  # disabled no-op
        viewer = _DisabledViewerProto()
    reporter = _LiveViewerReporter(viewer, interval=interval)
    return viewer, reporter


__all__ = ["LiveTrajectoryViewer", "build_live_viewer_reporter"]
