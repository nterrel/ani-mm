import pytest
import openmm


@pytest.mark.skipif(openmm is None, reason="OpenMM not available")
def test_alanine_helper_smoke(capfd):
    from animm.alanine_dipeptide import simulate_alanine_dipeptide

    info = simulate_alanine_dipeptide(n_steps=5, report_interval=5, minimize=False)
    assert info["steps"] == 5
    assert "final_potential_kjmol" in info
    if info.get("initial_potential_kjmol"):
        assert info["potential_delta_kjmol"] == pytest.approx(
            info["final_potential_kjmol"] - info["initial_potential_kjmol"], rel=1e-6
        )
    out = capfd.readouterr().out
    assert "Initial potential:" in out
