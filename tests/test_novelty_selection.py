"""Regression tests for the pre-registered novelty selection procedure."""
import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances
from models.baselines import (RBFOCSVMBaseline, novelty_loco_splits,
                              training_distance_median, build_baseline)


def fixture():
    rng = np.random.default_rng(901)
    X = np.vstack([rng.normal(c, .4, (12, 3)) for c in (-2., 0., 2.)])
    y = np.repeat(['a', 'b', 'c'], 12)
    V = np.vstack([rng.normal(c, .4, (9, 3)) for c in (-2., 0., 2.)])
    return X, y, V, np.repeat(['a', 'b', 'c'], 9)


def test_loco_excludes_entire_surrogate_and_calibration_from_fit():
    X, y, _, _ = fixture()
    for family, fit, cal, surrogate in novelty_loco_splits(y, 2):
        assert set(y[surrogate]) == {family}
        assert family not in y[fit] and family not in y[cal]
        assert not (set(fit) & set(cal) or set(fit) & set(surrogate) or set(cal) & set(surrogate))
        assert set(np.r_[fit, cal, surrogate]) == set(range(len(X)))
    with pytest.raises(ValueError, match='Fast_Scattering'):
        novelty_loco_splits(np.append(y, 'Fast_Scattering'))


def test_auroc_objective_distinguishes_equal_false_alarm_rates():
    class Detector:
        def __init__(self, reverse=False): self.reverse = reverse
        def decision_function(self, X): return (-1 if not self.reverse else 1) * X[:, 0]
    model = RBFOCSVMBaseline()
    known, surrogate = np.zeros((10, 1)), np.ones((10, 1))
    assert model._score_known_validation_fold(Detector(), known, surrogate) == 1.
    assert model._score_known_validation_fold(Detector(True), known, surrogate) == 0.


def test_complete_gamma_grid_medians_and_nu_calibration_are_auditable():
    X, y, V, vy = fixture()
    model = RBFOCSVMBaseline(budget=60, seed=2).fit(X, y, validation=(V, vy))
    audit = model.selection_audit_
    np.testing.assert_allclose(audit['c_grid'], np.logspace(-2, 2, 9))
    expected_median = np.median(euclidean_distances(X, squared=True)[np.triu_indices(len(X), 1)])
    assert training_distance_median(X) == expected_median
    assert model.best_params_['gamma'] == audit['c'] / expected_median
    assert len(model.cv_results_) == 9 and model.n_configs_evaluated_ == 54
    assert audit['preregistered_gamma_grid_complete']
    selected = min(audit['final_nu_calibration'], key=lambda r: (abs(r['native_known_fpr']-.05), r['nu']))
    assert model.best_params_['nu'] == selected['nu']
    assert np.mean(model.predict(V)) <= .05
    splits = novelty_loco_splits(y, 2)
    for row in model.cv_results_:
        for fold, (_, fit, _, _) in zip(row['folds'], splits):
            assert fold['training_distance_median'] == training_distance_median(X[fit])
            assert fold['gamma'] == row['params']['c'] / training_distance_median(X[fit])


def test_no_unknown_labels_permitted_in_external_calibration():
    X, y, V, vy = fixture()
    vy[0] = 'new'
    with pytest.raises(ValueError, match='known training families'):
        RBFOCSVMBaseline(budget=1).fit(X, y, validation=(V, vy))


@pytest.mark.parametrize('name', ['rbf_ocsvm', 'isolation_forest', 'lof'])
def test_unlabelled_api_is_explicitly_untuned(name):
    X, _, _, _ = fixture()
    model = build_baseline(name).fit(X)
    assert model.tuning_mode_ == 'unlabelled_fixed_heuristic_no_loco'
    assert model.get_budget_report()['best_cv_score'] is None
    assert model.n_configs_evaluated_ == 1


def test_collapsed_distance_grid_is_rejected():
    with pytest.raises(ValueError, match='positive training-distance median'):
        training_distance_median(np.zeros((3, 2)))


def test_isolation_forest_warm_start_preserves_the_seeded_score_sequence():
    X, _, V, _ = fixture()
    model = build_baseline('isolation_forest', seed=8)
    params = dict(n_estimators=100, max_samples=.75, contamination=.05,
                  max_features=.5, bootstrap=True)
    growing = model._build_estimator(params).set_params(warm_start=True).fit(X)
    growing.set_params(n_estimators=200).fit(X)
    direct = model._build_estimator(params | dict(n_estimators=200)).fit(X)
    np.testing.assert_array_equal(growing.decision_function(V), direct.decision_function(V))
