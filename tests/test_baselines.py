from __future__ import annotations

import numpy as np
import pytest
from models.baselines import build_baseline

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

FEWSHOT_MODELS = [
    "rbf_svm",
    "laplacian_svm",
    "knn",
    "logistic_regression",
    "hist_gradient_boosting",
]

NOVELTY_MODELS = [
    "rbf_ocsvm",
    "isolation_forest",
    "lof",
]

@pytest.fixture
def fewshot_glitch_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic fixture mirroring Protocol 1: 6 classes, d=7 features, k=5 shots."""
    rng = np.random.default_rng(123)
    n_classes = 6
    n_features = 7
    k_shot = 5
    n_test_per_class = 15

    X_train = np.vstack(
        [rng.normal(loc=c * 1.2, scale=0.7, size=(k_shot, n_features)) for c in range(n_classes)]
    )
    y_train = np.repeat(np.arange(n_classes), k_shot)

    X_test = np.vstack(
        [
            rng.normal(loc=c * 1.2, scale=0.7, size=(n_test_per_class, n_features))
            for c in range(n_classes)
        ]
    )
    y_test = np.repeat(np.arange(n_classes), n_test_per_class)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def novelty_loco_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic fixture mirroring Protocol 2 (LOCO): 6 known classes + 1 unseen O3 class."""
    rng = np.random.default_rng(321)
    n_known_classes = 6
    n_features = 7

    # Known families (inliers) used for training (no labels required)
    X_train_known = np.vstack(
        [rng.normal(loc=c * 1.0, scale=0.6, size=(25, n_features)) for c in range(n_known_classes)]
    )

    # Test pool: held-out known inliers (0) + unseen novel glitch family e.g. Fast_Scattering (1)
    X_test_known = np.vstack(
        [rng.normal(loc=c * 1.0, scale=0.6, size=(10, n_features)) for c in range(n_known_classes)]
    )
    X_test_novel = rng.normal(loc=8.5, scale=0.7, size=(30, n_features))

    X_test = np.vstack([X_test_known, X_test_novel])
    y_test = np.concatenate(
        [np.zeros(len(X_test_known), dtype=int), np.ones(len(X_test_novel), dtype=int)]
    )
    return X_train_known, X_test, y_test


# CONTRACT & ERROR HANDLING TESTS

def test_build_baseline_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        build_baseline("non_existent_model")


@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_invalid_hyperparameters_raise(model_name: str) -> None:
    with pytest.raises(ValueError, match="budget"):
        build_baseline(model_name, budget=0)

    with pytest.raises(ValueError, match="cv_folds"):
        build_baseline(model_name, cv_folds=1)


@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_unfitted_model_raises(model_name: str, fewshot_glitch_fixture) -> None:
    _, _, X_test, _ = fewshot_glitch_fixture
    model = build_baseline(model_name, budget=4, seed=42)

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X_test)

    with pytest.raises(RuntimeError, match="fitted"):
        model.score_samples(X_test)


@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_fewshot_missing_labels_raises(model_name: str, fewshot_glitch_fixture) -> None:
    X_train, _, _, _ = fewshot_glitch_fixture
    model = build_baseline(model_name, budget=4, seed=42)

    with pytest.raises(ValueError, match="requires target labels"):
        model.fit(X_train, y=None)


# FEW-SHOT models tests

@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_fewshot_fit_predict_and_metrics(model_name: str, fewshot_glitch_fixture) -> None:
    X_train, y_train, X_test, y_test = fewshot_glitch_fixture
    n_classes = len(np.unique(y_train))

    model = build_baseline(model_name, budget=10, seed=42, cv_folds=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores = model.score_samples(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = model.get_budget_report()

    print(f"\n--- [{model_name.upper()}] ---")
    print(f"Budget valutato : {report['n_configs_evaluated']}/{report['budget_limit']}")
    print(f"Best params     : {report['best_params']}")
    print(f"CV Macro-F1     : {report['best_cv_score']:.4f}")
    print(f"Test Macro-F1   : {macro_f1:.4f} | Balanced Acc: {bal_acc:.4f}")
    print(f"Inferenza (1 per classe): pred={y_pred[::15].tolist()} vs true={y_test[::15].tolist()}")

    assert y_pred.shape == (len(y_test),)
    assert scores.shape == (len(y_test), n_classes)
    assert macro_f1 > 0.5
    assert bal_acc > 0.5


@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_fewshot_equal_budget_report(model_name: str, fewshot_glitch_fixture) -> None:
    X_train, y_train, _, _ = fewshot_glitch_fixture
    budget = 8

    model = build_baseline(model_name, budget=budget, seed=42, cv_folds=5)
    model.fit(X_train, y_train)
    report = model.get_budget_report()

    assert report["model_name"] == model_name
    assert report["task"] == "fewshot"
    assert report["budget_limit"] == budget
    assert report["n_configs_evaluated"] == budget
    assert isinstance(report["best_params"], dict) and len(report["best_params"]) > 0
    assert np.isfinite(report["best_cv_score"])


def test_fewshot_cross_model_equal_budget(fewshot_glitch_fixture) -> None:
    """Verify all registered few-shot baselines evaluate the exact same config count."""
    X_train, y_train, _, _ = fewshot_glitch_fixture
    budget = 6

    evaluated_counts = [
        build_baseline(name, budget=budget, seed=42)
        .fit(X_train, y_train)
        .get_budget_report()["n_configs_evaluated"]
        for name in FEWSHOT_MODELS
    ]
    assert all(count == budget for count in evaluated_counts)


@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_fewshot_seed_determinism(model_name: str, fewshot_glitch_fixture) -> None:
    X_train, y_train, X_test, _ = fewshot_glitch_fixture

    model_a = build_baseline(model_name, budget=10, seed=7).fit(X_train, y_train)
    model_b = build_baseline(model_name, budget=10, seed=7).fit(X_train, y_train)

    assert model_a.best_params_ == model_b.best_params_
    assert model_a.best_cv_score_ == model_b.best_cv_score_
    np.testing.assert_array_equal(model_a.predict(X_test), model_b.predict(X_test))


@pytest.mark.parametrize("model_name", FEWSHOT_MODELS)
def test_fewshot_edge_cases(model_name: str) -> None:
    """Ensure small k-shot (k=2 < cv_folds) and zero-variance inputs do not crash."""
    # Case 1: k=2 samples per class with cv_folds=5 requested
    rng = np.random.default_rng(99)
    X_small = np.vstack([rng.normal(loc=c * 2.0, scale=0.5, size=(2, 7)) for c in range(4)])
    y_small = np.repeat(np.arange(4), 2)
    X_eval = rng.normal(size=(8, 7))

    model_small = build_baseline(model_name, budget=4, seed=42, cv_folds=5)
    model_small.fit(X_small, y_small)
    assert model_small.predict(X_eval).shape == (8,)

    # Case 2: constant (zero-variance) features
    X_const = np.ones((12, 7), dtype=np.float64)
    y_const = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    model_const = build_baseline(model_name, budget=4, seed=42, cv_folds=4)
    model_const.fit(X_const, y_const)
    assert model_const.predict(np.ones((6, 7))).shape == (6,)


def test_knn_clamps_neighbors_when_fold_size_is_small() -> None:
    """Verify kNN never exceeds inner CV training fold size when k=5 on few classes."""
    rng = np.random.default_rng(77)
    # 2 classes, k=5 -> N=10 -> with cv_folds=5 each inner train fold has 8 samples (< n_neighbors=9)
    X_train = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.5, size=(5, 7)),
            rng.normal(loc=3.0, scale=0.5, size=(5, 7)),
        ]
    )
    y_train = np.repeat([0, 1], 5)
    X_test = rng.normal(loc=0.0, scale=0.5, size=(6, 7))

    model = build_baseline("knn", budget=30, seed=42, cv_folds=5)
    model.fit(X_train, y_train)

    assert model.predict(X_test).shape == (6,)
    assert model.score_samples(X_test).shape == (6, 2)


def test_logistic_regression_tunes_regularization_c(fewshot_glitch_fixture) -> None:
    """Verify LogisticRegression tunes C within budget and converges cleanly."""
    X_train, y_train, X_test, _ = fewshot_glitch_fixture

    model = build_baseline("logistic_regression", budget=12, seed=42, cv_folds=5)
    model.fit(X_train, y_train)

    assert "C" in model.best_params_
    assert model.best_params_["C"] in {1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0}
    assert model.score_samples(X_test).shape == (len(X_test), 6)

def test_hist_gradient_boosting_splits_in_small_k_regime(fewshot_glitch_fixture) -> None:
    """Verify lowered min_samples_leaf enables tree splits when N=30 (6 classes, k=5)."""
    X_train, y_train, X_test, _ = fewshot_glitch_fixture

    model = build_baseline("hist_gradient_boosting", budget=12, seed=42, cv_folds=5)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert model.best_params_["min_samples_leaf"] <= 8
    assert len(np.unique(preds)) > 1
    assert model.score_samples(X_test).shape == (len(X_test), 6)


# NOVELTY DETECTION / LOCO UNIFIED TESTS


@pytest.mark.parametrize("model_name", NOVELTY_MODELS)
def test_novelty_unfitted_model_raises(model_name: str, novelty_loco_fixture) -> None:
    _, X_test, _ = novelty_loco_fixture
    model = build_baseline(model_name, budget=4, seed=42)

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X_test)

    with pytest.raises(RuntimeError, match="fitted"):
        model.score_samples(X_test)


@pytest.mark.parametrize("model_name", NOVELTY_MODELS)
def test_novelty_fit_predict_and_metrics(model_name: str, novelty_loco_fixture) -> None:
    X_train_known, X_test, y_test = novelty_loco_fixture

    model = build_baseline(model_name, budget=10, seed=42, cv_folds=5)
    model.fit(X_train_known)  # Unlabelled known families only

    y_pred = model.predict(X_test)
    anomaly_scores = model.score_samples(X_test)

    auroc = roc_auc_score(y_test, anomaly_scores)
    avg_prec = average_precision_score(y_test, anomaly_scores)
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    tpr_at_1pct_fpr = float(np.interp(0.01, fpr, tpr))
    report = model.get_budget_report()

    print(f"\n--- [NOVELTY: {model_name.upper()}] ---")
    print(f"Budget valutato : {report['n_configs_evaluated']}/{report['budget_limit']}")
    print(f"Best params     : {report['best_params']}")
    print(f"CV Calib Score  : {report['best_cv_score']:.4f}")
    print(f"AUROC: {auroc:.4f} | AP: {avg_prec:.4f} | TPR@1%FPR: {tpr_at_1pct_fpr:.4f}")
    print(
        f"Score medio     : known={anomaly_scores[y_test == 0].mean():.3f} "
        f"vs novel={anomaly_scores[y_test == 1].mean():.3f}"
    )

    assert y_pred.shape == (len(y_test),)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert anomaly_scores.shape == (len(y_test),)
    assert anomaly_scores[y_test == 1].mean() > anomaly_scores[y_test == 0].mean()
    assert auroc > 0.85
    assert avg_prec > 0.80


@pytest.mark.parametrize("model_name", NOVELTY_MODELS)
def test_novelty_equal_budget_report(model_name: str, novelty_loco_fixture) -> None:
    X_train_known, _, _ = novelty_loco_fixture
    budget = 8

    model = build_baseline(model_name, budget=budget, seed=42, cv_folds=5)
    model.fit(X_train_known)
    report = model.get_budget_report()

    assert report["model_name"] == model_name
    assert report["task"] == "novelty"
    assert report["budget_limit"] == budget
    assert report["n_configs_evaluated"] == budget
    assert isinstance(report["best_params"], dict) and len(report["best_params"]) > 0
    assert np.isfinite(report["best_cv_score"])


@pytest.mark.parametrize("model_name", NOVELTY_MODELS)
def test_novelty_seed_determinism(model_name: str, novelty_loco_fixture) -> None:
    X_train_known, X_test, _ = novelty_loco_fixture

    model_a = build_baseline(model_name, budget=10, seed=7).fit(X_train_known)
    model_b = build_baseline(model_name, budget=10, seed=7).fit(X_train_known)

    assert model_a.best_params_ == model_b.best_params_
    assert model_a.best_cv_score_ == model_b.best_cv_score_
    np.testing.assert_array_equal(model_a.predict(X_test), model_b.predict(X_test))
    np.testing.assert_allclose(model_a.score_samples(X_test), model_b.score_samples(X_test))


def test_rbf_ocsvm_tunes_nu_and_gamma(novelty_loco_fixture) -> None:
    X_train_known, _, _ = novelty_loco_fixture

    model = build_baseline("rbf_ocsvm", budget=12, seed=42, cv_folds=5)
    model.fit(X_train_known)

    assert "nu" in model.best_params_
    assert "gamma" in model.best_params_

def test_isolation_forest_tunes_tree_and_contamination_params(novelty_loco_fixture) -> None:
    """Verify IsolationForest tunes ensemble and contamination hyperparameters."""
    X_train_known, _, _ = novelty_loco_fixture

    model = build_baseline("isolation_forest", budget=12, seed=42, cv_folds=5)
    model.fit(X_train_known)

    expected_keys = {
        "n_estimators",
        "max_samples",
        "contamination",
        "max_features",
        "bootstrap",
    }
    assert expected_keys.issubset(model.best_params_.keys())


def test_novelty_cross_model_equal_budget(novelty_loco_fixture) -> None:
    """Verify all 3 registered novelty baselines evaluate the exact same config count."""
    X_train_known, _, _ = novelty_loco_fixture
    budget = 6

    evaluated_counts = [
        build_baseline(name, budget=budget, seed=42)
        .fit(X_train_known)
        .get_budget_report()["n_configs_evaluated"]
        for name in NOVELTY_MODELS
    ]
    assert all(count == budget for count in evaluated_counts)


def test_lof_tunes_neighbors_metric_and_contamination(novelty_loco_fixture) -> None:
    """Verify LOF (novelty=True) tunes neighborhood, metric, and contamination."""
    X_train_known, _, _ = novelty_loco_fixture

    model = build_baseline("lof", budget=12, seed=42, cv_folds=5)
    model.fit(X_train_known)

    assert {"n_neighbors", "metric", "contamination"}.issubset(model.best_params_.keys())


def test_lof_clamps_neighbors_on_small_known_set() -> None:
    """Verify LOF clamps n_neighbors cleanly when training fold size < n_neighbors."""
    rng = np.random.default_rng(88)
    X_train_small = rng.normal(loc=0.0, scale=0.5, size=(12, 7))
    X_test_inlier = rng.normal(loc=0.0, scale=0.5, size=(5, 7))
    X_test_outlier = rng.normal(loc=6.0, scale=0.5, size=(5, 7))
    X_test = np.vstack([X_test_inlier, X_test_outlier])

    model = build_baseline("lof", budget=20, seed=42, cv_folds=4)
    model.fit(X_train_small)

    preds = model.predict(X_test)
    scores = model.score_samples(X_test)

    assert preds.shape == (10,)
    assert scores[5:].mean() > scores[:5].mean()