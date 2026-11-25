"""Independent worker recalculation must bind the real aggregate byte content."""
import pytest

from data.splits import SplitConfig, generate_splits
from experiments import e03_parallel as parallel
from experiments.e03_report import verify_results
from experiments.e03_validation_parallel import snapshot_worker, verify_aggregate
from tests.fixtures.synthetic import synthetic_curated
from tests.test_e03_review import InlinePool


def test_verified_worker_snapshots_survive_cleanup_and_detect_aggregate_tampering(tmp_path, monkeypatch):
    frame = synthetic_curated(n_per_class=120, seed=4)
    splits = tmp_path / "splits"
    generate_splits(frame, output_dir=splits,
                    config=SplitConfig(seeds=(0,), test_per_class=2, val_per_class=2))
    curated = tmp_path / "curated.csv"
    frame.to_csv(curated, index=False, float_format="%.17g")
    snapshots, out = tmp_path / "snapshots", tmp_path / "out"
    real_run = parallel._run_fold

    def observed(data, family, params):
        result = real_run(data, family, params)
        assert snapshot_worker(parallel.Path(params["artifact_dir"]), snapshots / family)
        return result

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(parallel, "_run_fold", observed)
    metrics = parallel.run_parallel(frame, workers=1, seeds=[0],
        held_outs=["Fast_Scattering", "Blip"], budget=1, cv_folds=2,
        n_qubits=7, layers=1, s=.3, max_train=32, max_val=32, max_test=64,
        bootstrap=10, event_bootstrap=10, split_dir=splits, artifact_dir=out, store_bloch=True)
    assert not (out / "partial_rotations").exists()
    receipts = {family: verify_results(snapshots / family, splits, curated)
                for family in ("Fast_Scattering", "Blip")}
    result = verify_aggregate(out, snapshots, receipts)
    assert result["metric_values_recalculated"] == len(metrics) == 80
    assert result["full_frozen_cases_checked"] == 2
    assert result["scalers_refitted_from_selected_training"]
    prediction = out / "predictions/Fast_Scattering.csv.gz"
    prediction.write_bytes(prediction.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_aggregate(out, snapshots, receipts)
