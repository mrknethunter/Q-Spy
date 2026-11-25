#!/usr/bin/env python3
"""Few-shot classification model (M-02) using SVC with precomputed kernels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from models.base import BaseBudgetedModel, BaseFewShotModel
from models.novelty import KernelFn, _normalize_bloch_input, projected_quantum_kernel_matrix


class _PrecomputedKernelSVC(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible SVC(kernel='precomputed') wrapper with OvR strategy."""

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | str = "scale",
        kernel: Literal["projected", "precomputed", "rbf"] = "projected",
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

    def _prepare_features(self, X: np.ndarray) -> tuple[np.ndarray, int | None]:
        if self.kernel == "projected" and self.kernel_fn is None:
            return _normalize_bloch_input(X, n_qubits=self.n_qubits)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        return arr, self.n_qubits

    def _resolve_gamma(self, X_train: np.ndarray) -> float:
        if isinstance(self.gamma, (int, float)):
            gamma_val = float(self.gamma)
            if gamma_val <= 0.0:
                raise ValueError(f"gamma must be > 0, got {gamma_val}")
            return gamma_val

        x_var = float(np.var(X_train))
        if self.kernel == "projected":
            if self.gamma == "auto":
                return 1.0
            if self.gamma == "scale":
                return 1.0 / x_var if x_var > 0.0 else 1.0

        n_features = X_train.shape[1]
        if self.gamma == "auto":
            return 1.0 / n_features
        if self.gamma == "scale":
            return 1.0 / (n_features * x_var) if x_var > 0.0 else 1.0 / n_features

        raise ValueError(f"Unsupported gamma specification: {self.gamma!r}")

    def _compute_gram(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        if self.kernel_fn is not None:
            return np.asarray(self.kernel_fn(X_a, X_b, self.gamma_), dtype=np.float64)

        if self.kernel == "projected":
            return projected_quantum_kernel_matrix(
                X_a, X_b, gamma=self.gamma_, n_qubits=self.resolved_qubits_
            )

        if self.kernel == "rbf":
            sq_dists = euclidean_distances(X_a, X_b, squared=True)
            return np.exp(-self.gamma_ * sq_dists)

        raise ValueError(
            f"Unsupported kernel '{self.kernel}'. Expected 'projected', 'precomputed', or 'rbf'."
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> _PrecomputedKernelSVC:
        if self.kernel == "precomputed":
            K_train = np.asarray(X, dtype=np.float64)
            if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
                raise ValueError(
                    f"Precomputed training kernel must be a square 2D matrix, got {K_train.shape}."
                )
            self.n_train_samples_ = int(K_train.shape[0])
        else:
            X_train, resolved_qubits = self._prepare_features(X)
            self.X_train_ = X_train
            self.resolved_qubits_ = resolved_qubits
            self.gamma_ = self._resolve_gamma(self.X_train_)
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
        if self.kernel == "precomputed":
            K_eval = np.asarray(X, dtype=np.float64)
            if K_eval.ndim != 2 or K_eval.shape[1] != self.n_train_samples_:
                raise ValueError(
                    f"Expected precomputed Gram matrix with {self.n_train_samples_} columns, "
                    f"got {K_eval.shape}."
                )
            return K_eval

        X_eval, _ = self._prepare_features(X)
        return self._compute_gram(X_eval, self.X_train_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.predict(self._eval_gram(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.svc_.decision_function(self._eval_gram(X))


class FewShotSVC(BaseFewShotModel):
    """Primary Q-Spy few-shot classifier (M-02) with precomputed kernel and nested CV.

    Supports:
    - `kernel="projected"`: Bloch vectors `(N, 3n)` or `(N, n, 3)` with post-hoc `gamma` tuning.
    - `kernel="precomputed"`: externally computed Gram matrices (e.g., fidelity kernel `k_F`).
    - `kernel="rbf"` or custom `kernel_fn`: tabular feature fixtures or product-map ablations.
    """

    model_name = "fewshot_svc"

    def __init__(
        self,
        kernel: Literal["projected", "precomputed", "rbf"] = "projected",
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
        if self.kernel == "precomputed" and self.kernel_fn is None:
            # Ensure equal-budget compliance when only C is tuned (no post-hoc gamma)
            c_values = (
                list(self.c_grid)
                if self.c_grid is not None
                else np.logspace(-3, 3, num=max(self.budget, 30)).tolist()
            )
            return {"C": c_values}

        c_values = (
            list(self.c_grid)
            if self.c_grid is not None
            else [1e-2, 1e-1, 1.0, 10.0, 100.0]
        )
        gamma_values = (
            list(self.gamma_grid)
            if self.gamma_grid is not None
            else ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]
        )
        return {
            "C": c_values,
            "gamma": gamma_values,
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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FewShotSVC:
        if self.kernel != "precomputed" or self.kernel_fn is not None:
            super().fit(X, y)
            return self

        if y is None:
            raise ValueError("Few-shot classification requires target labels `y`.")

        K_arr = np.asarray(X, dtype=np.float64)
        if K_arr.ndim != 2 or K_arr.shape[0] != K_arr.shape[1]:
            raise ValueError(
                f"Precomputed training kernel must be a square 2D matrix, got {K_arr.shape}."
            )
        y_arr = np.asarray(y)

        configs = self._sample_configurations()
        n_splits = self._resolve_n_splits(y_arr)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = list(skf.split(np.zeros(len(y_arr)), y_arr))

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        self.cv_results_ = []

        for params in configs:
            fold_scores: list[float] = []
            for train_idx, val_idx in splits:
                est = self._build_estimator(params)
                est.fit(K_arr[np.ix_(train_idx, train_idx)], y_arr[train_idx])
                preds = est.predict(K_arr[np.ix_(val_idx, train_idx)])
                fold_scores.append(
                    float(f1_score(y_arr[val_idx], preds, average="macro", zero_division=0))
                )

            mean_score = float(np.mean(fold_scores))
            self.cv_results_.append({"params": params, "mean_macro_f1": mean_score})

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.n_configs_evaluated_ = len(configs)
        self.best_params_ = dict(best_params)
        self.best_cv_score_ = best_score

        final_estimator = self._build_estimator(self.best_params_)
        final_estimator.fit(K_arr, y_arr)
        self.best_estimator_ = final_estimator
        return self

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Predict on `X` and compute RQ3 metrics (macro-F1, balanced accuracy, per-class)."""
        y_pred = self.predict(X)
        return evaluate_fewshot(y_true=y, y_pred=y_pred, target_names=target_names)


def evaluate_fewshot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute RQ3 few-shot metrics: macro-F1, balanced accuracy, and per-class report."""
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true_arr.shape}, y_pred has {y_pred_arr.shape}."
        )
    if len(y_true_arr) == 0:
        raise ValueError("Cannot evaluate empty prediction arrays.")

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

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
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "per_class": per_class,
    }


def assert_equal_search_budget(
    models: Sequence[BaseBudgetedModel],
    expected_budget: int | None = None,
) -> int:
    """Assert that every quantum and classical model evaluated the exact same config budget."""
    if not models:
        raise ValueError("models sequence must not be empty.")

    evaluated: dict[str, int] = {}
    for idx, model in enumerate(models):
        report = model.get_budget_report()
        key = f"{report['model_name']}#{idx}"
        count = int(report["n_configs_evaluated"])
        if count < 1:
            raise AssertionError(f"Model '{key}' has not been fitted (n_configs_evaluated={count}).")
        evaluated[key] = count

    unique_counts = set(evaluated.values())
    if len(unique_counts) != 1:
        raise AssertionError(
            f"Equal-budget violation across models: {evaluated}"
        )

    common_budget = next(iter(unique_counts))
    if expected_budget is not None and common_budget != int(expected_budget):
        raise AssertionError(
            f"Expected search budget {expected_budget}, but models evaluated {common_budget}: {evaluated}"
        )

    return common_budget