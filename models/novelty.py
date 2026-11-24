#!/usr/bin/env python3
"""Novelty detection model (M-03) using One-Class SVM with precomputed kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import OneClassSVM
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from models.base import BaseNoveltyModel

KernelFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def _normalize_bloch_input(X: np.ndarray, n_qubits: int | None = None) -> tuple[np.ndarray, int]:
    """Flatten (N, n_qubits, 3) or validate (N, 3 * n_qubits) Bloch-vector arrays."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[2] != 3:
            raise ValueError(f"3D Bloch vectors must have shape (N, n_qubits, 3), got {arr.shape}.")
        inferred_qubits = int(arr.shape[1])
        if n_qubits is not None and inferred_qubits != n_qubits:
            raise ValueError(f"Expected n_qubits={n_qubits}, got {inferred_qubits}.")
        return arr.reshape(arr.shape[0], inferred_qubits * 3), inferred_qubits

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D input array, got shape {arr.shape}.")

    if n_qubits is not None:
        expected_cols = n_qubits * 3
        if arr.shape[1] != expected_cols:
            raise ValueError(
                f"Expected {expected_cols} columns for n_qubits={n_qubits}, got {arr.shape[1]}."
            )
        return arr, int(n_qubits)

    if arr.shape[1] % 3 != 0 or arr.shape[1] == 0:
        raise ValueError(
            f"Projected kernel requires feature dimension divisible by 3 (3n), got {arr.shape[1]}."
        )
    return arr, int(arr.shape[1] // 3)


def projected_quantum_kernel_matrix(
    B_a: np.ndarray,
    B_b: np.ndarray,
    gamma: float,
    n_qubits: int | None = None,
) -> np.ndarray:
    """Compute k_P(x, x') = exp(-gamma / n * sum_q ||b_q(x) - b_q(x')||_2^2)."""
    flat_a, qubits_a = _normalize_bloch_input(B_a, n_qubits=n_qubits)
    flat_b, qubits_b = _normalize_bloch_input(B_b, n_qubits=qubits_a)
    if qubits_a != qubits_b:
        raise ValueError(f"Qubit count mismatch: {qubits_a} vs {qubits_b}.")

    sq_dists = euclidean_distances(flat_a, flat_b, squared=True)
    return np.exp(-(float(gamma) / float(qubits_a)) * sq_dists)


class _PrecomputedKernelOCSVM(BaseEstimator):
    """Scikit-learn compatible OneClassSVM(kernel='precomputed') wrapper."""

    def __init__(
        self,
        nu: float = 0.05,
        gamma: float | str = "scale",
        kernel: Literal["projected", "rbf"] = "projected",
        n_qubits: int | None = None,
        kernel_fn: KernelFn | None = None,
    ) -> None:
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.kernel_fn = kernel_fn

    def _prepare_features(self, X: np.ndarray) -> tuple[np.ndarray, int | None]:
        if self.kernel == "projected" and self.kernel_fn is None:
            return _normalize_bloch_input(X, n_qubits=self.n_qubits)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)
        return arr, self.n_qubits

    def _resolve_gamma(self, X_train: np.ndarray, resolved_qubits: int | None) -> float:
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

        raise ValueError(f"Unsupported kernel '{self.kernel}'. Expected 'projected' or 'rbf'.")

    def fit(self, X: np.ndarray, y: Any = None) -> _PrecomputedKernelOCSVM:
        X_train, resolved_qubits = self._prepare_features(X)
        self.X_train_ = X_train
        self.resolved_qubits_ = resolved_qubits
        self.gamma_ = self._resolve_gamma(self.X_train_, self.resolved_qubits_)

        K_train = self._compute_gram(self.X_train_, self.X_train_)
        self.ocsvm_ = OneClassSVM(kernel="precomputed", nu=float(self.nu))
        self.ocsvm_.fit(K_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_eval, _ = self._prepare_features(X)
        K_eval = self._compute_gram(X_eval, self.X_train_)
        return self.ocsvm_.predict(K_eval)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X_eval, _ = self._prepare_features(X)
        K_eval = self._compute_gram(X_eval, self.X_train_)
        return self.ocsvm_.decision_function(K_eval)


class NoveltyOCSVM(BaseNoveltyModel):
    """Primary Q-Spy novelty detector (M-03) with precomputed kernel and equal-budget tuning.

    Trains strictly on known glitch families (`fit(X_known)`) and selects `(nu, gamma)`
    on known-only inner validation splits to prevent any held-out class leakage.
    """

    model_name = "novelty_ocsvm"

    def __init__(
        self,
        kernel: Literal["projected", "rbf"] = "projected",
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
        self.nu_grid = nu_grid or [0.01, 0.05, 0.1, 0.15, 0.2]
        self.gamma_grid = gamma_grid or ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "nu": list(self.nu_grid),
            "gamma": list(self.gamma_grid),
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _PrecomputedKernelOCSVM(
            nu=params["nu"],
            gamma=params["gamma"],
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            kernel_fn=self.kernel_fn,
        )
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Score `X` and compute RQ4 novelty detection metrics against binary labels `y`."""
        scores = self.score_samples(X)
        return evaluate_novelty(y_true=y, anomaly_scores=scores)


def evaluate_novelty(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """Compute RQ4 novelty detection metrics: AUROC, Average Precision, and TPR@1%FPR.

    Parameters
    ----------
    y_true:
        Binary ground-truth array where `0` denotes known families (inliers)
        and `1` denotes the held-out novel family (outliers).
    anomaly_scores:
        Continuous anomaly scores where higher values indicate higher novelty
        (as returned by `BaseNoveltyModel.score_samples`).
    target_fpr:
        False positive rate operating point for TPR interpolation (default `0.01` = 1%).
    """
    y_arr = np.asarray(y_true, dtype=int).ravel()
    scores_arr = np.asarray(anomaly_scores, dtype=np.float64).ravel()

    if y_arr.shape != scores_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_arr.shape}, anomaly_scores has {scores_arr.shape}."
        )
    if not (0.0 <= target_fpr <= 1.0):
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}.")

    unique_labels = set(np.unique(y_arr).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(
            f"y_true must contain both known (0) and novel (1) labels, got {sorted(unique_labels)}."
        )

    auroc = float(roc_auc_score(y_arr, scores_arr))
    avg_precision = float(average_precision_score(y_arr, scores_arr))
    fpr, tpr, _ = roc_curve(y_arr, scores_arr)
    tpr_at_target_fpr = float(np.interp(target_fpr, fpr, tpr))

    return {
        "auroc": auroc,
        "average_precision": avg_precision,
        "tpr_at_1pct_fpr": tpr_at_target_fpr,
    }