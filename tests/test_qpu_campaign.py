import pytest
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from scripts.qpu_campaign import run_campaign
from store.io import read_bloch
from tests.fixtures.synthetic import synthetic_curated, synthetic_features

df = synthetic_curated(n_per_class=6)
X = dict(zip(df["id"], synthetic_features(df)))
ids = list(X)
SUBSETS = {"status": "bound_to_hw_freeze", "subsets": {
    "drift_references": {"ids": ids[:2]}, "main": {"ids": ids[2:8]}, "control": {"ids": ids[8:11]},
    "concentration": [{"ids": ids[11:14]}, {"ids": ids[14:17]}]}}
FROZEN = {"n_qubits": 3, "layers": 1, "s": 1.0, "precision": 0.1, "concentration_qubits": [2, 3], "budget_s": 600}


def features(batch_ids):
    return [X[i] for i in batch_ids]


def test_dry_run_plans_every_experiment_and_stores_emulations(tmp_path):
    s = run_campaign(FROZEN, SUBSETS, features, out_dir=tmp_path, store_root=tmp_path / "store")
    names = [e["experiment"] for e in s["experiments"]]
    assert names == ["drift-start", "main", "control", "concentration-n2", "concentration-n3", "drift-end"]
    assert s["estimated_qpu_s"] == pytest.approx(sum(e["estimated_qpu_s"] for e in s["experiments"]))
    assert [e["mitigation"] for e in s["experiments"]][2] is False
    _, B, meta = read_bloch(f"{s['run_id']}-main-L2", root=tmp_path / "store")
    assert B.shape == (6, 3, 3) and meta["tier"] == "L2"


def test_submission_is_refused_over_budget_before_any_job(tmp_path):
    with pytest.raises(RuntimeError, match="budget"):
        run_campaign({**FROZEN, "budget_s": 1}, SUBSETS, features, submit=True, device=FakeManilaV2(),
                     out_dir=tmp_path, store_root=tmp_path / "store")
    assert not (tmp_path / "store").exists() or not list((tmp_path / "store").glob("*-HW"))


def test_submission_runs_in_one_batch_and_stores_hardware_runs(tmp_path):
    s = run_campaign(FROZEN, SUBSETS, features, submit=True, device=FakeManilaV2(), out_dir=tmp_path,
                     store_root=tmp_path / "store")
    _, B, meta = read_bloch(f"{s['run_id']}-control-HW", root=tmp_path / "store")
    assert B.shape == (3, 3, 3) and meta["mitigation"]["twirling"] is False and meta["job_ids"]
    assert "drift" in s and len(s["ledger_rows"]) == 6


def test_submission_requires_frozen_subsets(tmp_path):
    with pytest.raises(RuntimeError, match="freeze"):
        run_campaign(FROZEN, {**SUBSETS, "status": "draft_pending_l05"}, features, submit=True,
                     device=FakeManilaV2(), out_dir=tmp_path, store_root=tmp_path / "store")
