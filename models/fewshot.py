#!/usr/bin/env python3
"""Few-shot classification model (M-02) using SVC with precomputed kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.svm import SVC

try:
    from kernels.classical import laplacian_kernel, rbf_kernel
except ImportError:
    from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel

from models.base import GAMMA_GRID, BaseBudgetedModel, BaseFewShotModel, resolve_gamma

KernelFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class _PrecomputedKernelSVC(BaseEstimator, ClassifierMixin):
    """SVC(kernel='precomputed') wrapper supporting Z representations and precomputed Gram matrices."""

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | str = "scale",
        kernel: Literal["rbf", "projected", "laplacian", "precomputed"] = "rbf",
        n_qubits: int | None = None,
        kernel_fn: KernelFn | None = None,
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.kernel_fn = kernel_fn
        self.random_state = random_state

    def _compute_gram(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        if self.kernel_fn is not None:
            return np.asarray(self.kernel_fn(X_a, X_b, self.gamma_), dtype=np.float64)
        if self.kernel in ("rbf", "projected"):
            return rbf_kernel(X_a, X_b, gamma=self.gamma_)
        if self.kernel == "laplacian":
            return laplacian_kernel(X_a, X_b, gamma=self.gamma_)
        raise ValueError(f"Unsupported kernel: {self.kernel!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> _PrecomputedKernelSVC:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.kernel == "precomputed" and self.kernel_fn is None:
            K_train = X_arr
        else:
            self.X_train_ = X_arr
            self.gamma_ = resolve_gamma(
                self.X_train_,
                self.gamma,
                n_qubits=self.n_qubits if self.kernel == "projected" else None,
            )
            K_train = self._compute_gram(self.X_train_, self.X_train_)

        self.svc_ = SVC(
            kernel="precomputed",
            C=float(self.C),
            decision_function_shape="ovr",
            random_state=self.random_state,
        )
        self.svc_.fit(K_train, y)
        self.classes_ = self.svc_.classes_
        return self

    def _eval_gram(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.kernel == "precomputed" and self.kernel_fn is None:
            return X_arr
        return self._compute_gram(X_arr, self.X_train_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict(self._eval_gram(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.decision_function(self._eval_gram(X))


class FewShotSVC(BaseFewShotModel):
    """Primary Q-Spy few-shot classifier (M-02) with precomputed kernel and nested CV."""

    model_name = "fewshot_svc"

    def __init__(
        self,
        kernel: Literal["rbf", "projected", "laplacian", "precomputed"] = "rbf",
        n_qubits: int | None = None,
        kernel_fn: KernelFn | None = None,
        c_grid: list[float] | None = None,
        gamma_grid: list[float | str] | None = None,
        budget: int = 16,
        seed: int = 42,
        cv_folds: int = 5,
    ) -> None:
        super().__init__(budget=budget, seed=seed, cv_folds=cv_folds)
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.kernel_fn = kernel_fn
        self.c_grid = c_grid
        self.gamma_grid = gamma_grid

    def _param_grid(self) -> dict[str, list[Any]]:
        if self._is_precomputed():
            c_values = (
                list(self.c_grid)
                if self.c_grid is not None
                else np.logspace(-3, 3, num=max(self.budget, 30)).tolist()
            )
            return {"C": c_values}

        return {
            "C": list(self.c_grid) if self.c_grid is not None else [1e-2, 1e-1, 1.0, 10.0, 100.0],
            "gamma": list(self.gamma_grid) if self.gamma_grid is not None else list(GAMMA_GRID),
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _PrecomputedKernelSVC(
            C=params["C"],
            gamma=params.get("gamma", "scale"),
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            kernel_fn=self.kernel_fn,
            random_state=self.seed,
        )

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Predict on `X` and compute RQ3 metrics (macro-F1, balanced accuracy, per-class)."""
        return evaluate_fewshot(y_true=y, y_pred=self.predict(X), target_names=target_names)


def evaluate_fewshot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute RQ3 few-shot metrics: macro-F1, balanced accuracy, and per-class report."""
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if y_true_arr.shape != y_pred_arr.shape or len(y_true_arr) == 0:
        raise ValueError(
            f"Invalid shapes for evaluation: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}."
        )

    raw_report = classification_report(
        y_true_arr,
        y_pred_arr,
        target_names=list(target_names) if target_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    summary_keys = {"accuracy", "macro avg", "weighted avg"}
    per_class = {
        str(label): {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1-score": float(metrics["f1-score"]),
            "support": int(metrics["support"]),
        }
        for label, metrics in raw_report.items()
        if label not in summary_keys
    }

    return {
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "per_class": per_class,
    }


def assert_equal_search_budget(
    models: Sequence[BaseBudgetedModel],
    expected_budget: int | None = None,
) -> int:
    """Assert that every quantum and classical model evaluated the exact same config budget."""
    if not models:
        raise ValueError("models sequence must not be empty.")

    evaluated = {
        f"{m.get_budget_report()['model_name']}#{i}": int(m.get_budget_report()["n_configs_evaluated"])
        for i, m in enumerate(models)
    }
    if any(count < 1 for count in evaluated.values()):
        raise AssertionError(f"Unfitted model found in budget check: {evaluated}")

    unique_counts = set(evaluated.values())
    if len(unique_counts) != 1:
        raise AssertionError(f"Equal-budget violation across models: {evaluated}")

    common_budget = next(iter(unique_counts))
    if expected_budget is not None and common_budget != int(expected_budget):
        raise AssertionError(
            f"Expected search budget {expected_budget}, got {common_budget}: {evaluated}"
        )
    return common_budget