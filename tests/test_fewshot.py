#!/usr/bin/env python3
"""Tests for FewShotSVC (M-02): k-shot regimes, equal budget across Q/C, and RQ3 metrics."""

from __future__ import annotations

import numpy as np
import pytest

from models.base import BaseBudgetedModel, BaseFewShotModel
from models.baselines import BASELINE_REGISTRY, build_baseline
from models.fewshot import FewShotSVC, assert_equal_search_budget, evaluate_fewshot
from models.novelty import projected_quantum_kernel_matrix

GLITCH_CLASSES = ["Blip", "Koi_Fish", "LFB", "Scattered_Light", "Tomte", "Whistle"]


def _make_bloch_kshot_split(
    k_shot: int,
    n_classes: int = 6,
    n_qubits: int = 8,
    n_test_per_class: int = 20,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic Bloch vectors in [-1, 1]^{3n} for Protocol 1 with k shots per class."""
    rng = np.random.default_rng(seed + k_shot)
    dim = n_qubits * 3
    centers = rng.uniform(-0.7, 0.7, size=(n_classes, dim))

    X_train = np.vstack(
        [
            np.clip(rng.normal(loc=centers[c], scale=0.18, size=(k_shot, dim)), -1.0, 1.0)
            for c in range(n_classes)
        ]
    )
    y_train = np.repeat(np.arange(n_classes), k_shot)

    X_test = np.vstack(
        [
            np.clip(rng.normal(loc=centers[c], scale=0.18, size=(n_test_per_class, dim)), -1.0, 1.0)
            for c in range(n_classes)
        ]
    )
    y_test = np.repeat(np.arange(n_classes), n_test_per_class)
    return X_train, y_train, X_test, y_test


@pytest.mark.parametrize("k_shot", [5, 10, 20, 50])
def test_fewshot_svc_runs_for_every_k(k_shot: int) -> None:
    """Acceptance criterion 1: runs on synthetic fixtures for every k in {5, 10, 20, 50}."""
    X_train, y_train, X_test, y_test = _make_bloch_kshot_split(k_shot=k_shot, n_qubits=8)

    model = FewShotSVC(kernel="projected", n_qubits=8, budget=10, seed=42, cv_folds=5)
    assert isinstance(model, BaseBudgetedModel)
    assert isinstance(model, BaseFewShotModel)

    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test, target_names=GLITCH_CLASSES)
    scores = model.score_samples(X_test)

    print(
        f"\n--- [FEWSHOT_SVC (projected, k={k_shot})] ---\n"
        f"Best params : {model.best_params_} | CV macro-F1: {model.best_cv_score_:.4f}\n"
        f"Test macro-F1: {metrics['macro_f1']:.4f} | "
        f"Balanced Acc: {metrics['balanced_accuracy']:.4f}"
    )

    assert scores.shape == (len(y_test), len(GLITCH_CLASSES))
    assert metrics["macro_f1"] > 0.85
    assert metrics["balanced_accuracy"] > 0.85
    assert set(metrics["per_class"].keys()) == set(GLITCH_CLASSES)


def test_fewshot_svc_precomputed_gram_matrix_mode() -> None:
    """Verify FewShotSVC works on precomputed square/rectangular Gram matrices (e.g., k_F)."""
    X_train, y_train, X_test, y_test = _make_bloch_kshot_split(k_shot=10, n_qubits=8)
    K_train = projected_quantum_kernel_matrix(X_train, X_train, gamma=0.25, n_qubits=8)
    K_test = projected_quantum_kernel_matrix(X_test, X_train, gamma=0.25, n_qubits=8)

    budget = 12
    model = FewShotSVC(kernel="precomputed", budget=budget, seed=42, cv_folds=5)
    model.fit(K_train, y_train)
    metrics = model.evaluate(K_test, y_test, target_names=GLITCH_CLASSES)

    assert model.get_budget_report()["n_configs_evaluated"] == budget
    assert "C" in model.best_params_ and "gamma" not in model.best_params_
    assert metrics["macro_f1"] > 0.85
    assert metrics["balanced_accuracy"] > 0.85


def test_search_budget_identical_across_quantum_and_classical_models() -> None:
    """Acceptance criterion 2: search budget is identical across quantum and classical models."""
    budget = 8
    X_bloch_tr, y_tr, _, _ = _make_bloch_kshot_split(k_shot=10, n_qubits=8)
    K_tr = projected_quantum_kernel_matrix(X_bloch_tr, X_bloch_tr, gamma=0.1, n_qubits=8)

    # Tabular d=7 fixture for the 5 classical baselines
    rng = np.random.default_rng(77)
    X_tab_tr = np.vstack([rng.normal(loc=c * 1.2, scale=0.7, size=(10, 7)) for c in range(6)])

    q_projected = FewShotSVC(kernel="projected", n_qubits=8, budget=budget, seed=42).fit(
        X_bloch_tr, y_tr
    )
    q_precomputed = FewShotSVC(kernel="precomputed", budget=budget, seed=42).fit(K_tr, y_tr)

    classical_fewshot_names = [
        name for name, cls in BASELINE_REGISTRY.items() if cls.task == "fewshot"
    ]
    assert len(classical_fewshot_names) == 5

    classical_models = [
        build_baseline(name, budget=budget, seed=42).fit(X_tab_tr, y_tr)
        for name in classical_fewshot_names
    ]

    all_models = [q_projected, q_precomputed, *classical_models]
    verified_budget = assert_equal_search_budget(all_models, expected_budget=budget)
    assert verified_budget == budget


def test_assert_equal_search_budget_detects_mismatch() -> None:
    X_tr, y_tr, _, _ = _make_bloch_kshot_split(k_shot=5, n_qubits=4)
    m_8 = FewShotSVC(kernel="projected", n_qubits=4, budget=8, seed=42).fit(X_tr, y_tr)
    m_12 = FewShotSVC(kernel="projected", n_qubits=4, budget=12, seed=42).fit(X_tr, y_tr)

    with pytest.raises(AssertionError, match="Equal-budget violation"):
        assert_equal_search_budget([m_8, m_12])


def test_evaluate_fewshot_metrics_and_per_class_report() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])

    res = evaluate_fewshot(y_true, y_pred, target_names=["Blip", "Tomte", "Whistle"])

    assert set(res.keys()) == {"macro_f1", "balanced_accuracy", "per_class"}
    # Balanced accuracy: (1.0 + 0.5 + 1.0) / 3 = 0.8333...
    assert res["balanced_accuracy"] == pytest.approx(5.0 / 6.0)
    # Macro-F1: (1.0 + 2/3 + 0.8) / 3 = 0.8222...
    assert res["macro_f1"] == pytest.approx((1.0 + (2.0 / 3.0) + 0.8) / 3.0)
    assert res["per_class"]["Blip"]["f1-score"] == pytest.approx(1.0)
    assert res["per_class"]["Tomte"]["recall"] == pytest.approx(0.5)
    assert res["per_class"]["Whistle"]["support"] == 2