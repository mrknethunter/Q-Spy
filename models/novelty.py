#!/usr/bin/env python3
"""Novelty detection model (M-03) using One-Class SVM with precomputed kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM

try:
    from kernels.classical import laplacian_kernel, rbf_kernel
except ImportError:
    from sklearn.metrics.pairwise import laplacian_kernel, rbf_kernel

from models.base import GAMMA_GRID, BaseNoveltyModel, resolve_gamma

KernelFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


class _PrecomputedKernelOCSVM(BaseEstimator):
    """OneClassSVM(kernel='precomputed') wrapper supporting Z representations and Gram matrices."""

    def __init__(
        self,
        nu: float = 0.05,
        gamma: float | str = "scale",
        kernel: Literal["rbf", "projected", "laplacian", "precomputed"] = "rbf",
        n_qubits: int | None = None,
        kernel_fn: KernelFn | None = None,
    ) -> None:
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.kernel_fn = kernel_fn

    def _compute_gram(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        if self.kernel_fn is not None:
            return np.asarray(self.kernel_fn(X_a, X_b, self.gamma_), dtype=np.float64)
        if self.kernel in ("rbf", "projected"):
            return rbf_kernel(X_a, X_b, gamma=self.gamma_)
        if self.kernel == "laplacian":
            return laplacian_kernel(X_a, X_b, gamma=self.gamma_)
        raise ValueError(f"Unsupported kernel: {self.kernel!r}")

    def fit(self, X: np.ndarray, y: Any = None) -> _PrecomputedKernelOCSVM:
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

        self.ocsvm_ = OneClassSVM(kernel="precomputed", nu=float(self.nu))
        self.ocsvm_.fit(K_train)
        return self

    def _eval_gram(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.kernel == "precomputed" and self.kernel_fn is None:
            return X_arr
        return self._compute_gram(X_arr, self.X_train_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ocsvm_.predict(self._eval_gram(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.ocsvm_.decision_function(self._eval_gram(X))


class NoveltyOCSVM(BaseNoveltyModel):
    """Primary Q-Spy novelty detector (M-03) with precomputed kernel and equal-budget tuning."""

    model_name = "novelty_ocsvm"

    def __init__(
        self,
        kernel: Literal["rbf", "projected", "laplacian", "precomputed"] = "rbf",
        n_qubits: int | None = None,
        kernel_fn: KernelFn | None = None,
        nu_grid: list[float] | None = None,
        gamma_grid: list[float | str] | None = None,
        budget: int = 16,
        seed: int = 42,
        cv_folds: int = 5,
    ) -> None:
        super().__init__(budget=budget, seed=seed, cv_folds=cv_folds)
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.kernel_fn = kernel_fn
        self.nu_grid = nu_grid
        self.gamma_grid = gamma_grid

    def _param_grid(self) -> dict[str, list[Any]]:
        if self._is_precomputed():
            nu_values = (
                list(self.nu_grid)
                if self.nu_grid is not None
                else np.linspace(0.01, 0.30, num=max(self.budget, 30)).tolist()
            )
            return {"nu": nu_values}

        return {
            "nu": list(self.nu_grid) if self.nu_grid is not None else [0.01, 0.05, 0.1, 0.15, 0.2],
            "gamma": list(self.gamma_grid) if self.gamma_grid is not None else list(GAMMA_GRID),
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _PrecomputedKernelOCSVM(
            nu=params["nu"],
            gamma=params.get("gamma", "scale"),
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            kernel_fn=self.kernel_fn,
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Score `X` and compute RQ4 novelty detection metrics against binary labels `y`."""
        return evaluate_novelty(y_true=y, anomaly_scores=self.score_samples(X))


def evaluate_novelty(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """Compute RQ4 novelty detection metrics: AUROC, Average Precision, and TPR@1%FPR."""
    y_arr = np.asarray(y_true, dtype=int).ravel()
    scores_arr = np.asarray(anomaly_scores, dtype=np.float64).ravel()

    if y_arr.shape != scores_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_arr.shape}, anomaly_scores has {scores_arr.shape}."
        )
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}.")
    if set(np.unique(y_arr).tolist()) != {0, 1}:
        raise ValueError("y_true must contain both known (0) and novel (1) labels.")

    fpr, tpr, _ = roc_curve(y_arr, scores_arr)
    return {
        "auroc": float(roc_auc_score(y_arr, scores_arr)),
        "average_precision": float(average_precision_score(y_arr, scores_arr)),
        "tpr_at_1pct_fpr": float(np.interp(target_fpr, fpr, tpr)),
    }