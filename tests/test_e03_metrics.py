"""Synthetic tests for E-03 event-level uncertainty and paired comparisons."""
import numpy as np
import pytest

from experiments.e03_metrics import anomaly_metrics, event_bootstrap_seed, paired_event_bootstrap


def test_event_bootstrap_is_deterministic_stratified_and_paired():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = {
        "projected_ocsvm": np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
        "rbf_ocsvm": np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
        "isolation_forest": np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]),
        "lof": np.array([0.2, 0.3, 0.1, 0.8, 0.9, 0.7]),
    }
    first, first_pairs = paired_event_bootstrap(y, scores, n_resamples=80, seed=61)
    second, second_pairs = paired_event_bootstrap(y, scores, n_resamples=80, seed=61)
    assert first == second
    assert first_pairs == second_pairs
    assert all(row["n_known"] == 3 and row["n_novel"] == 3 for row in first)
    assert all(row["positive_fraction"] == 0.5 for row in first)
    assert all(row["ci_method"] == "paired_stratified_event_bootstrap_fixed_prevalence" for row in first)
    tied_model = [row for row in first if row["model"] in {"projected_ocsvm", "rbf_ocsvm"}]
    tied_pairs = [row for row in first_pairs if "projected_ocsvm minus rbf_ocsvm" == row["comparison"]]
    assert len(tied_model) == 6 and len(tied_pairs) == 3
    assert all(row["value"] == row["ci_low"] == row["ci_high"] == 0.0 for row in tied_pairs)


def test_event_bootstrap_uses_fixed_event_prevalence_and_correct_score_orientation():
    y = np.array([0] * 8 + [1] * 2)
    good = np.arange(10, dtype=float)
    values, _ = anomaly_metrics(y, good)
    assert values["auroc"] == 1.0
    rows, _ = paired_event_bootstrap(y, {"model": good}, n_resamples=50, seed=7)
    auroc = next(row for row in rows if row["metric"] == "auroc")
    assert auroc["value"] == 1.0
    assert auroc["n_known"] == 8 and auroc["n_novel"] == 2
    assert auroc["positive_fraction"] == 0.2


def test_event_bootstrap_seed_is_stable_and_fold_specific():
    assert event_bootstrap_seed("Fast_Scattering", 0) == event_bootstrap_seed("Fast_Scattering", 0)
    assert event_bootstrap_seed("Fast_Scattering", 0) != event_bootstrap_seed("Blip", 0)


@pytest.mark.parametrize("n_resamples,seed", [(0, 1), (True, 1), (2, -1)])
def test_event_bootstrap_rejects_invalid_configuration(n_resamples, seed):
    with pytest.raises(ValueError):
        paired_event_bootstrap([0, 1], {"m": np.array([0.0, 1.0])},
                               n_resamples=n_resamples, seed=seed)
