"""Scientific regressions for the review's shared nested-LOCO correction."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from models.base import BaseNoveltyModel
from models.baselines import build_baseline, novelty_selection_splits
from experiments.e03_parallel import _concat_nonempty
from experiments.e03_tuning import validate_hyperparameters


def known_training():
    rng = np.random.default_rng(94)
    labels = np.repeat(np.array(["Blip", "Tomte", "Koi_Fish"], dtype=object), 20)
    X = np.vstack([rng.normal(center, .3, (20, 7)) for center in (-5., 0., 5.)])
    return X, labels


def test_real_ocsvm_selects_all_nine_median_gamma_factors_and_calibrates_nu():
    X, labels = known_training()
    model = build_baseline("rbf_ocsvm", budget=9, seed=2, cv_folds=2).fit(X, labels)
    np.testing.assert_allclose([row["params"]["gamma_factor"] for row in model.cv_results_],
                                np.logspace(-2, 2, 9))
    assert model.get_budget_report()["tuning_strategy"] == "nested_surrogate_loco_auroc"
    assert len(model.cv_results_) == 9
    assert all(len(row["folds"]) == 6 for row in model.cv_results_)
    assert all(len(fold["calibration"]["nu_trials"]) == 5
               for row in model.cv_results_ for fold in row["folds"])
    expected = np.median(pdist(X, metric="sqeuclidean"))
    assert model.best_estimator_._gamma == pytest.approx(model.best_params_["gamma_factor"] / expected)


def test_surrogate_folds_exclude_the_family_from_fit_and_nu_calibration():
    X, labels = known_training()
    for fold in novelty_selection_splits(labels, seed=3, cv_folds=3):
        assert not set(fold["train"]) & set(fold["calibration"])
        assert not set(fold["train"]) & set(fold["surrogate"])
        assert not set(fold["calibration"]) & set(fold["surrogate"])
        assert set(labels[fold["surrogate"]]) == {fold["family"]}
        assert fold["family"] not in labels[fold["train"]]
        assert fold["family"] not in labels[fold["calibration"]]
    labels[0] = "Fast_Scattering"
    with pytest.raises(ValueError, match="Fast_Scattering"):
        build_baseline("rbf_ocsvm", budget=9).fit(X, labels)


def test_equal_known_fpr_does_not_hide_different_surrogate_discrimination():
    class ControlledEstimator:
        def __init__(self, direction):
            self.direction = direction

        def fit(self, X):
            self.X = X

        def predict(self, X):
            return np.ones(len(X))  # identical known FPR for both candidates

        def decision_function(self, X):
            nearest = np.sqrt(((X[:, None] - self.X[None, :])**2).sum(axis=2).min(axis=1))
            return self.direction * nearest

    class ControlledModel(BaseNoveltyModel):
        model_name = "controlled_discrimination_regression"

        def _param_grid(self):
            return {"direction": [-1, 1]}

        def _build_estimator(self, params):
            return ControlledEstimator(params["direction"])

    X, labels = known_training()
    model = ControlledModel(budget=2, seed=1, cv_folds=2).fit(X, labels)
    assert model.best_params_["direction"] == -1
    assert model.cv_results_[0]["mean_surrogate_auroc"] > .99
    assert model.cv_results_[1]["mean_surrogate_auroc"] < .01
    assert all(fold["calibration"]["observed_fpr"] == 0
               for row in model.cv_results_ for fold in row["folds"])


def test_real_selector_is_deterministic_and_receipt_reproduces_winner():
    X, labels = known_training()
    models = [build_baseline("rbf_ocsvm", budget=9, seed=3, cv_folds=2).fit(X, labels)
              for _ in range(2)]
    assert models[0].best_params_ == models[1].best_params_
    np.testing.assert_array_equal(models[0].score_samples(X[:5]), models[1].score_samples(X[:5]))
    model = models[0]
    receipt = model.get_budget_report() | {"inner_cv": model.cv_results_}
    folds = [dict(surrogate_family=part["family"], fold=part["fold"], n_train=len(part["train"]),
                  n_known=len(part["calibration"]), n_surrogate=len(part["surrogate"]))
             for part in novelty_selection_splits(labels, seed=3, cv_folds=2)]
    candidates = [row["params"] for row in model.cv_results_]
    assert validate_hyperparameters(receipt, candidates, model._loco_param_grid(), folds)["status"] == "pass"


def test_empty_rotation_diagnostics_do_not_emit_futurewarning():
    empty = pd.DataFrame(columns=["seed", "mean_distance"])
    populated = pd.DataFrame({"seed": [0], "mean_distance": [.5]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pd.testing.assert_frame_equal(_concat_nonempty([empty, populated, empty]), populated)
        assert _concat_nonempty([empty, empty]).empty
