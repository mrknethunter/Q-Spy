#!/usr/bin/env python3
"""Tests for NoveltyOCSVM (M-03): precomputed kernels, known-only tuning, and anti-leakage."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from models.novelty import NoveltyOCSVM, evaluate_novelty, projected_quantum_kernel_matrix

from models.base import BaseBudgetedModel, BaseNoveltyModel
from models.novelty import NoveltyOCSVM, projected_quantum_kernel_matrix


@pytest.fixture
def bloch_loco_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic Bloch-vector fixture (n=8 qubits -> 24 dims) for Protocol 2 (LOCO)."""
    rng = np.random.default_rng(2026)
    n_known_classes = 6
    n_qubits = 8
    dim = n_qubits * 3  # 24 Bloch coordinates in [-1, 1]

    # Known families constrained inside [-1, 1]^{3n}
    centers = rng.uniform(-0.65, 0.65, size=(n_known_classes, dim))
    X_train_known = np.vstack(
        [
            np.clip(rng.normal(loc=centers[c], scale=0.12, size=(25, dim)), -1.0, 1.0)
            for c in range(n_known_classes)
        ]
    )

    X_test_known = np.vstack(
        [
            np.clip(rng.normal(loc=centers[c], scale=0.12, size=(10, dim)), -1.0, 1.0)
            for c in range(n_known_classes)
        ]
    )
    # Held-out novel glitch class occupying a distinct region of Bloch space
    novel_center = np.full(dim, 0.88)
    X_test_novel = np.clip(rng.normal(loc=novel_center, scale=0.08, size=(30, dim)), -1.0, 1.0)

    X_test = np.vstack([X_test_known, X_test_novel])
    y_test = np.concatenate(
        [np.zeros(len(X_test_known), dtype=int), np.ones(len(X_test_novel), dtype=int)]
    )
    return X_train_known, X_test, y_test

@pytest.fixture
def novelty_loco_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic tabular fixture (d=7) for Protocol 2 (LOCO)."""
    rng = np.random.default_rng(321)
    n_known_classes = 6
    n_features = 7

    X_train_known = np.vstack(
        [rng.normal(loc=c * 1.0, scale=0.6, size=(25, n_features)) for c in range(n_known_classes)]
    )
    X_test_known = np.vstack(
        [rng.normal(loc=c * 1.0, scale=0.6, size=(10, n_features)) for c in range(n_known_classes)]
    )
    X_test_novel = rng.normal(loc=8.5, scale=0.7, size=(30, n_features))

    X_test = np.vstack([X_test_known, X_test_novel])
    y_test = np.concatenate(
        [np.zeros(len(X_test_known), dtype=int), np.ones(len(X_test_novel), dtype=int)]
    )
    return X_train_known, X_test, y_test

def test_projected_kernel_matrix_properties() -> None:
    rng = np.random.default_rng(11)
    B_2d = rng.uniform(-1.0, 1.0, size=(10, 24))
    B_3d = B_2d.reshape(10, 8, 3)

    K_2d = projected_quantum_kernel_matrix(B_2d, B_2d, gamma=0.5)
    K_3d = projected_quantum_kernel_matrix(B_3d, B_3d, gamma=0.5)

    assert K_2d.shape == (10, 10)
    np.testing.assert_allclose(K_2d, K_3d)
    np.testing.assert_allclose(K_2d, K_2d.T)
    np.testing.assert_allclose(np.diag(K_2d), 1.0)
    assert np.all((K_2d > 0.0) & (K_2d <= 1.0))


def test_novelty_ocsvm_projected_kernel_auroc_above_09(bloch_loco_fixture) -> None:
    X_train_known, X_test, y_test = bloch_loco_fixture

    model = NoveltyOCSVM(kernel="projected", n_qubits=8, budget=12, seed=42, cv_folds=5)
    assert isinstance(model, BaseBudgetedModel)
    assert isinstance(model, BaseNoveltyModel)

    model.fit(X_train_known)
    scores = model.score_samples(X_test)
    preds = model.predict(X_test)
    auroc = roc_auc_score(y_test, scores)

    print(f"\n--- [NOVELTY_OCSVM (projected, n=8)] ---")
    print(f"Best params : {model.best_params_}")
    print(f"CV score    : {model.best_cv_score_:.4f} | Test AUROC: {auroc:.4f}")

    assert preds.shape == (len(y_test),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert auroc > 0.90


def test_novelty_ocsvm_tabular_fixture_auroc_above_09(novelty_loco_fixture) -> None:
    """Acceptance criterion 1: clearly separated synthetic class reaches AUROC > 0.9."""
    X_train_known, X_test, y_test = novelty_loco_fixture

    model = NoveltyOCSVM(kernel="rbf", budget=12, seed=42, cv_folds=5)
    model.fit(X_train_known)

    scores = model.score_samples(X_test)
    auroc = roc_auc_score(y_test, scores)

    assert auroc > 0.90

def test_no_heldout_sample_used_in_model_selection(bloch_loco_fixture) -> None:
    """Acceptance criterion 2: model selection depends strictly on known families."""
    X_train_known, X_test, y_test = bloch_loco_fixture
    X_heldout = X_test[y_test == 1]

    seen_training_rows: list[int] = []
    is_fitting = True

    def spy_projected_kernel(B_a: np.ndarray, B_b: np.ndarray, gamma: float) -> np.ndarray:
        if is_fitting:
            seen_training_rows.append(len(B_b))
            # Ensure no held-out sample ever appears inside B_a or B_b during fit()
            assert not np.any(np.isclose(B_a[:, 0], 999.0))
            assert not np.any(np.isclose(B_b[:, 0], 999.0))
        return projected_quantum_kernel_matrix(B_a, B_b, gamma=gamma, n_qubits=8)

    # Poisoned held-out set that would corrupt tuning if leaked
    X_heldout_poisoned = np.full_like(X_heldout, 999.0)
    X_test_poisoned = np.vstack([X_test[y_test == 0], X_heldout_poisoned])

    model_clean = NoveltyOCSVM(kernel="projected", n_qubits=8, budget=10, seed=7).fit(
        X_train_known
    )
    model_spy = NoveltyOCSVM(kernel_fn=spy_projected_kernel, budget=10, seed=7).fit(
        X_train_known
    )
    is_fitting = False

    assert max(seen_training_rows) == len(X_train_known)
    assert model_clean.best_params_ == model_spy.best_params_
    assert model_clean.best_cv_score_ == model_spy.best_cv_score_
    np.testing.assert_array_equal(
        model_clean.best_estimator_.ocsvm_.support_,
        model_spy.best_estimator_.ocsvm_.support_,
    )

    # Scoring poisoned held-out samples after fit works without affecting fitted state
    scores_poisoned = model_spy.score_samples(X_test_poisoned)
    assert scores_poisoned.shape == (len(X_test_poisoned),)

def test_novelty_ocsvm_equal_budget_and_determinism(bloch_loco_fixture) -> None:
    X_train_known, X_test, _ = bloch_loco_fixture
    budget = 8

    m1 = NoveltyOCSVM(kernel="projected", n_qubits=8, budget=budget, seed=42).fit(X_train_known)
    m2 = NoveltyOCSVM(kernel="projected", n_qubits=8, budget=budget, seed=42).fit(X_train_known)

    report = m1.get_budget_report()
    assert report["model_name"] == "novelty_ocsvm"
    assert report["task"] == "novelty"
    assert report["n_configs_evaluated"] == budget
    assert m1.best_params_ == m2.best_params_
    np.testing.assert_allclose(m1.score_samples(X_test), m2.score_samples(X_test))


def test_novelty_ocsvm_invalid_bloch_dimension_raises() -> None:
    X_invalid = np.ones((15, 7))  # 7 is not divisible by 3n
    model = NoveltyOCSVM(kernel="projected", budget=4)
    with pytest.raises(ValueError, match="divisible by 3"):
        model.fit(X_invalid)

def test_evaluate_novelty_metrics_exact_and_partial_separation() -> None:
    # Case 1: perfect separation -> all RQ4 metrics equal 1.0
    y_true = np.array([0] * 100 + [1] * 20)
    perfect_scores = np.concatenate([np.linspace(-1.0, 0.0, 100), np.linspace(0.5, 2.0, 20)])

    metrics_perfect = evaluate_novelty(y_true, perfect_scores)
    assert set(metrics_perfect.keys()) == {"auroc", "average_precision", "tpr_at_1pct_fpr"}
    assert metrics_perfect["auroc"] == pytest.approx(1.0)
    assert metrics_perfect["average_precision"] == pytest.approx(1.0)
    assert metrics_perfect["tpr_at_1pct_fpr"] == pytest.approx(1.0)

    # Case 2: 2 of 100 inliers outrank half of the 20 novel triggers -> at FPR=1% (1 inlier), TPR=0.5
    y_partial = np.array([0] * 100 + [1] * 20)
    scores_partial = np.zeros(120, dtype=np.float64)
    scores_partial[:98] = np.linspace(0.0, 0.4, 98)     # 98 known below threshold
    scores_partial[100:110] = 0.6                       # 10 novel at score 0.6
    scores_partial[98] = 0.7                            # 1st false positive (FPR = 1%)
    scores_partial[110:120] = 0.8                       # 10 novel above 1st FP (TPR = 10/20 = 0.5)
    scores_partial[99] = 0.9                            # 2nd false positive (FPR = 2%)

    metrics_partial = evaluate_novelty(y_partial, scores_partial)
    assert 0.5 < metrics_partial["auroc"] < 1.0
    assert 0.5 < metrics_partial["average_precision"] < 1.0
    assert metrics_partial["tpr_at_1pct_fpr"] == pytest.approx(0.5)


def test_evaluate_novelty_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluate_novelty(np.array([0, 1, 0]), np.array([0.1, 0.9]))

    with pytest.raises(ValueError, match="both known"):
        evaluate_novelty(np.zeros(10, dtype=int), np.linspace(0.0, 1.0, 10))


def test_novelty_ocsvm_end_to_end_evaluate(bloch_loco_fixture) -> None:
    X_train_known, X_test, y_test = bloch_loco_fixture

    model = NoveltyOCSVM(kernel="projected", n_qubits=8, budget=12, seed=42, cv_folds=5)
    model.fit(X_train_known)
    metrics = model.evaluate(X_test, y_test)

    print(
        f"\n--- [RQ4 METRICS: NOVELTY_OCSVM] ---\n"
        f"AUROC={metrics['auroc']:.4f} | "
        f"AP={metrics['average_precision']:.4f} | "
        f"TPR@1%FPR={metrics['tpr_at_1pct_fpr']:.4f}"
    )

    assert metrics["auroc"] > 0.90
    assert metrics["average_precision"] > 0.90
    assert metrics["tpr_at_1pct_fpr"] >= 0.80