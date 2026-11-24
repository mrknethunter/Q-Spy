import json

from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from scripts.qpu_pilot import run_pilot
from store.io import read_bloch


def test_pilot_dry_run_stores_simulated_tiers_and_estimates_usage(tmp_path):
    s = run_pilot(points=4, qubits=3, layers=1, precision=0.05, out_dir=tmp_path, store_root=tmp_path / "store")
    assert s["hw"]["dry_run"] is True and s["hw"]["estimated_qpu_s"] > 0
    assert set(s["vs_L0"]) == {"L1", "L2"}
    ids, B, meta = read_bloch(f"{s['run_id']}-L2", root=tmp_path / "store")
    assert B.shape == (4, 3, 3) and len(ids) == 4 and meta["tier"] == "L2"
    assert json.loads((tmp_path / f"{s['run_id']}.json").read_text())["vs_L0"] == s["vs_L0"]


def test_pilot_submission_stores_hardware_run_with_provenance(tmp_path):
    s = run_pilot(points=4, qubits=3, layers=1, precision=0.05, submit=True, device=FakeManilaV2(),
                  out_dir=tmp_path, store_root=tmp_path / "store")
    _, B, meta = read_bloch(f"{s['run_id']}-HW", root=tmp_path / "store")
    assert B.shape == (4, 3, 3) and meta["job_ids"] and meta["calibration"] and "usage_s" in meta
    assert {"HW", "L2"} <= set(s["vs_L0"]) and "HW_vs_L2" in s
    assert s["ledger_row"].startswith("|") and meta["job_ids"][0] in s["ledger_row"]
