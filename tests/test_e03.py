"""E-03 scientific checks: exclusion, operating points, paired inference, L0 integration."""
import gzip
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from data.splits import Split, SplitConfig, generate_splits, load_split
from experiments import e03_novelty as e03
from experiments.e03_report import render_report, verify_results
from experiments.e03_parallel import run_parallel
from kernels.projected import projected_kernel
from tests.fixtures.synthetic import synthetic_curated


def test_subset_preserves_strata_proportions_and_is_order_independent():
    df = synthetic_curated(n_per_class=120)
    first = e03.stratified_subset(df, 80, "test:2026")
    second = e03.stratified_subset(df.sample(frac=1, random_state=55), 80, "test:2026")
    assert first["id"].tolist() == second["id"].tolist()
    assert len(first) == 80 and first["id"].is_unique
    assert set(first.groupby(["label", "ifo"]).groups) == set(df.groupby(["label", "ifo"]).groups)
    assert set(first["id"]) <= set(df["id"])
    assert set(e03.stratified_subset(df, 0, "all")["id"]) == set(df["id"])
    for cap in (-1, True, 1):
        with pytest.raises(ValueError):
            e03.stratified_subset(df, cap, "bad")


def test_proportional_cap_does_not_balance_rare_and_common_strata():
    df = pd.DataFrame({"id": [str(i) for i in range(1000)], "ifo": "H1",
                       "label": ["common"] * 990 + ["rare"] * 10})
    result = e03.stratified_subset(df, 100, "proportion")
    counts = result["label"].value_counts()
    assert counts["common"] >= 97 and 1 <= counts["rare"] <= 3


def test_metrics_orientation_ap_and_attainable_low_fpr():
    perfect, _ = e03.anomaly_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
    assert perfect == {"auroc": 1., "ap": 1., "tpr_at_1pct_fpr": 1.}
    reversed_scores, _ = e03.anomaly_metrics([0, 0, 1, 1], [.9, .8, .2, .1])
    assert reversed_scores["auroc"] == 0.
    # First attained ROC point above zero costs 10% FPR: no invented interpolation.
    y = np.array([0]*10 + [1]*10)
    scores = np.r_[1., np.zeros(9), np.full(10, .5)]
    values, _ = e03.anomaly_metrics(y, scores)
    assert values["tpr_at_1pct_fpr"] == 0.
    tied, _ = e03.anomaly_metrics(y, np.ones(20))
    assert tied == {"auroc": .5, "ap": .5, "tpr_at_1pct_fpr": 0.}


@pytest.mark.parametrize("y,scores", [([0, 0], [1., 2.]), ([0, 1], [np.nan, 2.]),
                                      ([0, 2], [1., 2.]), ([0, 1], [2.])])
def test_metrics_reject_invalid_evaluation(y, scores):
    with pytest.raises(ValueError):
        e03.anomaly_metrics(y, scores)


def test_known_only_quantile_threshold_is_conservative_with_ties():
    for scores in (np.linspace(-3, 3, 2048), np.ones(128), np.arange(32, dtype=float)):
        threshold = e03.calibration_threshold(scores)
        assert np.mean(scores > threshold) <= .01
    with pytest.raises(ValueError):
        e03.calibration_threshold([])


def test_projected_normalization_matches_the_shared_kernel():
    b = np.random.default_rng(2).uniform(-1, 1, (12, 8, 3)).astype(np.float32)
    z = b.reshape(len(b), -1).astype(float) / np.sqrt(8)
    assert np.allclose(rbf_kernel(z, gamma=.7), projected_kernel(b, b, .7), atol=1e-6)


def test_batched_exact_embedding_preserves_rows_and_expectation_values():
    X = np.random.default_rng(2).uniform(-1, 1, (19, 7))
    cfg = e03.EmbedConfig(8, 2, .1778)
    batched = e03.exact_bloch(X, cfg, batch_size=4)
    original = e03.embed(X, cfg, "L0").B
    np.testing.assert_array_equal(batched, original)


def test_full_partition_leakage_guard_runs_before_sampling():
    indexed = pd.DataFrame({"id": ["a", "b", "c", "d", "e"],
                            "label": ["known", "known", "known", "Fast_Scattering", "rotation"]}).set_index("id", drop=False)
    good = Split(("a",), ("b",), ("c", "d", "e"))
    e03.assert_no_leakage(indexed, good, ("Fast_Scattering", "rotation"))
    for bad in (Split(("a", "d"), ("b",), ("c", "e")),
                Split(("a",), ("b", "e"), ("c", "d")),
                Split(("a", "b"), ("b",), ("c", "d", "e")),
                Split(("missing",), ("b",), ("c", "d", "e"))):
        with pytest.raises(ValueError):
            e03.assert_no_leakage(indexed, bad, ("Fast_Scattering", "rotation"))


def test_bootstrap_is_paired_by_seed_not_overlapping_fold_rows():
    rows = []
    for seed in range(10):
        for model in e03.MODELS:
            value = .5 + seed / 100 + (.1 if model == "projected_ocsvm" else 0)
            rows.append(dict(held_out="Fast_Scattering", scope="target_family", metric="auroc",
                             seed=seed, model=model, value=value))
    summary, paired = e03.summarize(pd.DataFrame(rows), bootstrap=500)
    assert len(summary) == 4 and len(paired) == 3
    assert np.allclose(paired[["mean", "ci_low", "ci_high"]], .1)
    assert (summary["n_seeds"] == 10).all()
    one, _ = e03.summarize(pd.DataFrame(rows).query("seed == 0"), bootstrap=10)
    assert one["ci_low"].isna().all()  # a smoke run is not a CI estimate


def test_real_L0_pipeline_and_provenance_on_frozen_synthetic_bundle(tmp_path, monkeypatch):
    frame = synthetic_curated(n_per_class=120, seed=4)
    split_dir = tmp_path / "splits"
    generate_splits(frame, output_dir=split_dir,
                    config=SplitConfig(seeds=(0, 1), test_per_class=2, val_per_class=2))
    fitted_labels = []
    real_make_features = e03.make_features

    def watched_features(df, scaler=None):
        if scaler is None:
            fitted_labels.append(set(df["label"]))
        return real_make_features(df, scaler)

    monkeypatch.setattr(e03, "make_features", watched_features)
    result = e03.run_experiment(frame, seeds=[0, 1], held_outs=["Fast_Scattering", "Blip"],
                                budget=1, cv_folds=2, n_qubits=7, layers=1, s=.3,
                                max_train=32, max_val=32, max_test=64, bootstrap=20,
                                event_bootstrap=40,
                                split_dir=split_dir, artifact_dir=tmp_path / "out", store_bloch=False)
    assert len(result) == 4 * 4 * 2 * 5
    assert set(result["model"]) == set(e03.MODELS)
    assert np.isfinite(result["value"]).all()
    assert result["value"].between(-1e-12, 1 + 1e-12).all()
    assert (result["n_configs_evaluated"] == 1).all()
    assert all("Fast_Scattering" not in labels for labels in fitted_labels)
    assert "Blip" not in fitted_labels[-1]
    out = tmp_path / "out"
    assert json.loads((out / "manifest.json").read_text())["status"] == "complete"
    audit = json.loads((out / "audit.json").read_text())
    assert all(case["full_partition_leakage_checked"] for case in audit)
    assert all(model["validation_fpr"] <= .01 for case in audit for model in case["models"].values())
    assert all(len(model["inner_preprocessing_folds"]) == 2 for case in audit
               for model in case["models"].values())
    assert (out / "e03_event_bootstrap.csv").is_file()
    assert (out / "e03_event_paired_differences.csv").is_file()
    with gzip.open(out / "selected_ids.json.gz", "rt") as file:
        selections = json.load(file)
    for family in ("Fast_Scattering", "Blip"):
        cases = [case for case in selections if case["held_out"] == family]
        assert cases[0]["test"] == cases[1]["test"]
        predictions = pd.read_csv(out / "predictions" / (family + ".csv.gz"))
        for case in cases:
            frozen = load_split("P2", case["seed"], held_out=family, split_dir=split_dir)
            for name, ids in zip(("train", "val", "test"), (frozen.train_ids, frozen.val_ids, frozen.test_ids)):
                assert set(case[name]) <= set(ids)
            assert set(case["train"]).isdisjoint(case["val"])
        assert set(predictions["seed"]) == {0, 1}
    assert (out / "roc_fast_scattering.png").stat().st_size > 0
    assert (out / "roc_rotations.png").stat().st_size > 0
    cached_csv = tmp_path / "curated.csv"
    frame.to_csv(cached_csv, index=False, float_format="%.17g")
    verified = verify_results(out, split_dir, cached_csv)
    assert verified["metric_values_recalculated"] == len(result)
    assert verified["full_frozen_cases_checked"] == 4
    assert verified["scalers_refitted_from_selected_training"]
    assert verified["event_bootstrap_recalculated"]
    assert "Nota fisica" in render_report(out, verified)
    # The report must refuse a changed score artifact, even if summary tables look fine.
    prediction_file = out / "predictions" / "Fast_Scattering.csv.gz"
    prediction_file.write_bytes(prediction_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_results(out, split_dir)
    with pytest.raises(FileExistsError):
        e03.run_experiment(frame, seeds=[0], held_outs=["Fast_Scattering"], split_dir=split_dir,
                           artifact_dir=out, store_bloch=False)


def test_parallel_rotations_preserve_case_provenance_and_paired_metrics(tmp_path):
    frame = synthetic_curated(n_per_class=120, seed=4)
    splits = tmp_path / "splits"
    generate_splits(frame, output_dir=splits,
                    config=SplitConfig(seeds=(0, 1), test_per_class=2, val_per_class=2))
    out = tmp_path / "parallel"
    metrics = run_parallel(frame, workers=2, seeds=[0, 1], held_outs=["Fast_Scattering", "Blip"],
                           budget=1, cv_folds=2, n_qubits=7, layers=1, s=.3,
                           max_train=32, max_val=32, max_test=64, bootstrap=20,
                           event_bootstrap=40,
                           split_dir=str(splits), artifact_dir=str(out), store_bloch=False)
    result = verify_results(out, splits)
    assert result["metric_values_recalculated"] == len(metrics) == 160
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["settings"]["workers"] == 2 and manifest["completed_cases"] == 4
    assert not (out / "partial_rotations").exists()
    assert set(manifest["settings"]["held_outs"]) == {"Fast_Scattering", "Blip"}
