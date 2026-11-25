"""E-03 science-stack regressions for restart, thresholds and fold-local fits.

These require the pinned project runtime; they complement the stdlib integrity
suite and must not be reported as executed when those dependencies are absent.
"""
from concurrent.futures import Future
import json

import numpy as np
import pandas as pd
import pytest

from data.splits import SplitConfig, generate_splits
from experiments import e03_parallel as parallel
from experiments.e03_integrity import atomic_json, sha256_file
from experiments.e03_metrics import paired_event_bootstrap
from experiments.e03_preprocessing import FoldRepresentationCache
from experiments.e03_report import verify_results
from tests.fixtures.synthetic import synthetic_curated


class InlinePool:
    """Exercise parent orchestration deterministically, without spawn overhead."""
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    frame = synthetic_curated(n_per_class=120, seed=4)
    splits = tmp_path / "splits"
    generate_splits(frame, output_dir=splits,
                    config=SplitConfig(seeds=(0,), test_per_class=2, val_per_class=2))
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", InlinePool)
    params = dict(workers=1, seeds=[0], held_outs=["Fast_Scattering", "Blip"],
                  budget=1, cv_folds=2, n_qubits=7, layers=1, s=.3,
                  max_train=32, max_val=32, max_test=64, bootstrap=10,
                  event_bootstrap=10, split_dir=str(splits),
                  artifact_dir=str(tmp_path / "results"), store_bloch=True)
    return frame, splits, params


def test_failed_rotation_can_resume_only_matching_verified_workers(experiment, monkeypatch):
    frame, splits, params = experiment
    real_run = parallel._run_fold

    def fail_second(data, family, arguments):
        if family == "Blip":
            raise RuntimeError("synthetic worker failure")
        return real_run(data, family, arguments)

    monkeypatch.setattr(parallel, "_run_fold", fail_second)
    with pytest.raises(RuntimeError, match="synthetic worker"):
        parallel.run_parallel(frame, **params)
    out = parallel.Path(params["artifact_dir"])
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    assert (out / "partial_rotations/Fast_Scattering/manifest.json").is_file()
    executed = []

    def watched(data, family, arguments):
        executed.append(family)
        return real_run(data, family, arguments)

    monkeypatch.setattr(parallel, "_run_fold", watched)
    metrics = parallel.run_parallel(frame, resume=True, **params)
    assert executed == ["Blip"]
    assert verify_results(out, splits)["metric_values_recalculated"] == len(metrics)
    assert (out / "bloch").is_dir()
    assert not (out / "partial_rotations").exists()
    # A complete, unchanged run is reused without fitting again.
    pd.testing.assert_frame_equal(parallel.run_parallel(frame, resume=True, **params),
                                  pd.read_csv(out / "e03_metrics.csv", float_precision="round_trip"))
    assert executed == ["Blip"]
    with pytest.raises(ValueError, match="incompatible restart"):
        parallel.run_parallel(frame, resume=True, **(params | {"budget": 2}))


def test_independent_verifier_recalculates_validation_threshold(experiment):
    frame, splits, params = experiment
    parallel.run_parallel(frame, **params)
    out = parallel.Path(params["artifact_dir"])
    family = "Fast_Scattering"
    model = "projected_ocsvm"
    audit = json.loads((out / "audit.json").read_text())
    case = next(case for case in audit if case["held_out"] == family)
    changed = case["models"][model]["threshold"] + 100.
    case["models"][model]["threshold"] = changed
    case["models"][model]["validation_fpr"] = 0.
    atomic_json(out / "audit.json", audit)
    for directory in ("predictions", "validation_scores"):
        path = out / directory / f"{family}.csv.gz"
        table = pd.read_csv(path, float_precision="round_trip")
        table[model + "_threshold"] = changed
        table.to_csv(path, index=False, float_format="%.17g")
    manifest = json.loads((out / "manifest.json").read_text())
    for relative in ("audit.json", f"predictions/{family}.csv.gz", f"validation_scores/{family}.csv.gz"):
        manifest["files"][relative] = sha256_file(out / relative)
    atomic_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="validation threshold"):
        verify_results(out, splits)


def test_unregistered_scaler_fit_fails_before_shared_transform_is_called():
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("must not fit")

    cache = FoldRepresentationCache(make_features_fn=forbidden)
    with pytest.raises(ValueError, match="unregistered"):
        cache.fit_transform(np.ones((3, 7)))
    assert not calls


def test_event_pairing_reference_does_not_depend_on_mapping_order():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = {"projected_ocsvm": np.arange(6.), "rbf_ocsvm": -np.arange(6.),
              "lof": np.ones(6)}
    first = paired_event_bootstrap(y, scores, n_resamples=20, seed=4)
    second = paired_event_bootstrap(y, dict(reversed(list(scores.items()))), n_resamples=20, seed=4)
    assert first == second


def test_event_bootstrap_point_and_interval_use_the_same_fpr_limit():
    y = np.array([0] * 10 + [1] * 10)
    scores = np.r_[1., np.zeros(9), np.full(10, .5)]
    rows, _ = paired_event_bootstrap(y, {"projected_ocsvm": scores},
                                     n_resamples=20, seed=4, fpr_limit=.2)
    assert next(row for row in rows if row["metric"] == "tpr_at_1pct_fpr")["value"] == 1.
