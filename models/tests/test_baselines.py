from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score
from models.baselines import build_baseline


FEWSHOT_MODELS = [
    "rbf_svm",
    "laplacian_svm",
    "knn",
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