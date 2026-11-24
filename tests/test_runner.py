import json

import pandas as pd
import pytest
import yaml

from experiments.run import run

SMOKE = {
    "name": "smoke",
    "data": "experiments.e01_concentration:load_points",
    "data_params": {"points": 40, "seed": 0, "synthetic": True},
    "run": "experiments.e01_concentration:run_grid",
    "params": {"qubits": [2], "layers": [1], "s_values": [1.0], "fidelity_max_qubits": 2},
}


def test_runs_a_yaml_config_and_records_provenance(tmp_path):
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(yaml.safe_dump(SMOKE))
    out = run(cfg, out_dir=tmp_path)
    res = pd.read_csv(out)
    assert len(res) == 2 and set(res["kernel"]) == {"projected", "fidelity"}
    prov = json.loads(out.with_suffix(".run.json").read_text())
    assert prov["config"] == SMOKE and "git_commit" in prov and prov["rows"] == 2


def test_rejects_a_config_without_required_keys(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"name": "bad"}))
    with pytest.raises(ValueError, match="run"):
        run(cfg, out_dir=tmp_path)
