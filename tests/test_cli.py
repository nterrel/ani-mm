import json
import sys

import pytest

from animm.cli import main


def run_cli(args):
    return main(args)


def test_models_list(capsys):
    rc = run_cli(["models", "--json"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    models = json.loads(out)
    assert "ANI2DR" in models


def test_eval_smiles(capsys):
    rc = run_cli(["eval", "CCO", "--model", "ANI2DR", "--json"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["smiles"] == "CCO"
    assert "energy_hartree" in data


@pytest.mark.skipif("openmm" not in sys.modules, reason="OpenMM not imported yet")
def test_ala2_md_json(capsys):
    rc = run_cli(["ala2-md", "--steps", "5", "--json", "--report", "5"])  # tiny run
    assert rc == 0
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["steps"] == 5
    assert "final_potential_kjmol" in data
