"""The pilot script must be safe by default: dry run, no network, results written for later comparison."""
import json

from scripts.qpu_pilot import run_pilot


def test_pilot_dry_run_compares_simulated_tiers_and_writes_results(tmp_path):
    summary = run_pilot(points=4, qubits=3, layers=1, s=1.0, precision=0.05, submit=False, out_dir=tmp_path)
    assert summary["hw"]["dry_run"] is True and summary["hw"]["estimated_qpu_s"] > 0
    assert set(summary["vs_L0"]) == {"L1", "L2"}  # no HW numbers without a real run
    saved = json.loads((tmp_path / f"{summary['run_id']}.json").read_text())
    assert saved["vs_L0"] == summary["vs_L0"] and (tmp_path / f"{summary['run_id']}.npz").exists()
