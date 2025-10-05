import numpy as np

from animm.gui import build_live_viewer_reporter


def test_build_live_viewer_reporter_disabled_headless():
    # Provide empty positions so ASE backend requirement can be skipped
    symbols = ["H", "H"]
    viewer, reporter = build_live_viewer_reporter(symbols, interval=5, backend="mpl")
    assert hasattr(viewer, "enabled")
    # We can't guarantee matplotlib in test env; just assert reporter interface
    assert hasattr(reporter, "describeNextReport")
    assert hasattr(reporter, "report")

    # Update with dummy positions if viewer enabled (smoke path)
    if getattr(viewer, "enabled", False):
        viewer.update(np.zeros((2, 3)), step=0)  # type: ignore
