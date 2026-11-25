#!/usr/bin/env python3
"""Shared model interfaces and equal-budget hyperparameter tuning (M-02..M-04)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold

try:
    from kernels.classical import GAMMA_GRID
except ImportError:
    GAMMA_GRID: list[float | str] = ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0]


def as_2d_float64(X: np.ndarray) -> np.ndarray:
    """Ensure representation Z is a 2D float64 array (flattening 3D Bloch tensors if passed)."""
    arr = np.asarray(X, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1) if arr.ndim == 3 else arr


def resolve_gamma(
    X: np.ndarray,
    gamma: float | str | None = "scale",
    *,
    n_qubits: int | None = None,
) -> float:
    """Resolve `'scale'`, `'auto'`, or numeric gamma into a positive float64 scalar."""
    if isinstance(gamma, (int, float)) and not isinstance(gamma, bool):
        val = float(gamma)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"gamma must be finite and > 0, got {gamma!r}")
        return val / float(n_qubits) if n_qubits is not None else val

    n_features = float(X.shape[1])
    if gamma is None or gamma == "auto":
        return 1.0 / n_features
    if gamma == "scale":
        x_var = float(np.var(X))
        return 1.0 / (n_features * x_var) if x_var > 0.0 else 1.0 / n_features
    raise ValueError(f"Unsupported gamma specification: {gamma!r}")


class BaseBudgetedModel(ABC):
    """Common interface and equal-budget hyperparameter tuning across quantum and classical models."""

    task: str  # "fewshot" or "novelty"
    model_name: str

    def __init__(self, budget: int = 16, seed: int = 42, cv_folds: int = 5) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")

        self.budget = int(budget)
        self.seed = int(seed)
        self.cv_folds = int(cv_folds)

        self.best_estimator_: BaseEstimator | None = None
        self.best_params_: dict[str, Any] = {}
        self.best_cv_score_: float = -np.inf
        self.n_configs_evaluated_: int = 0
        self.cv_results_: list[dict[str, Any]] = []

    @abstractmethod
    def _param_grid(self) -> dict[str, list[Any]]:
        """Return candidate hyperparameter search space."""

    @abstractmethod
    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        """Instantiate a fresh underlying estimator for a given parameter dict."""

    def _is_precomputed(self) -> bool:
        return getattr(self, "kernel", None) == "precomputed" and getattr(self, "kernel_fn", None) is None

    def _sample_configurations(self) -> list[dict[str, Any]]:
        """Deterministically select up to `self.budget` hyperparameter configurations."""
        all_configs = list(ParameterGrid(self._param_grid()))
        if not all_configs:
            return [{}]
        if len(all_configs) <= self.budget:
            return all_configs

        rng = np.random.default_rng(self.seed)
        chosen_idx = rng.choice(len(all_configs), size=self.budget, replace=False)
        return [all_configs[int(i)] for i in sorted(chosen_idx)]

    def _resolve_n_splits(self, y: np.ndarray) -> int:
        _, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min())
        if min_class_count < 2:
            raise ValueError(f"Need at least 2 samples per class for inner CV, got {min_class_count}.")
        return min(self.cv_folds, min_class_count)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseBudgetedModel:
        """Tune hyperparameters within `self.budget` and fit on full training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (few-shot) or binary anomaly flags 0/1 (novelty)."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return class scores (few-shot) or anomaly scores where higher = more anomalous (novelty)."""

    def get_budget_report(self) -> dict[str, Any]:
        """Return auditable metadata proving equal-budget compliance."""
        return {
            "model_name": self.model_name,
            "task": self.task,
            "budget_limit": self.budget,
            "n_configs_evaluated": self.n_configs_evaluated_,
            "seed": self.seed,
            "best_params": dict(self.best_params_),
            "best_cv_score": float(self.best_cv_score_),
        }


class BaseFewShotModel(BaseBudgetedModel, ABC):
    """Base class for few-shot classifiers (both classical baselines and FewShotSVC)."""

    task = "fewshot"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseFewShotModel:
        if y is None:
            raise ValueError("Few-shot classification requires target labels `y`.")

        X_arr = as_2d_float64(X)
        y_arr = np.asarray(y)
        precomputed = self._is_precomputed()
        if precomputed and (X_arr.ndim != 2 or X_arr.shape[0] != X_arr.shape[1]):
            raise ValueError(f"Precomputed kernel must be square 2D, got {X_arr.shape}.")

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
                X_tr = X_arr[np.ix_(train_idx, train_idx)] if precomputed else X_arr[train_idx]
                X_va = X_arr[np.ix_(val_idx, train_idx)] if precomputed else X_arr[val_idx]
                est.fit(X_tr, y_arr[train_idx])
                preds = est.predict(X_va)
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
        final_estimator.fit(X_arr, y_arr)
        self.best_estimator_ = final_estimator
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self.best_estimator_.predict(as_2d_float64(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling score_samples().")
        X_arr = as_2d_float64(X)
        if hasattr(self.best_estimator_, "decision_function"):
            return self.best_estimator_.decision_function(X_arr)
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X_arr)
        raise NotImplementedError(f"{self.model_name} does not expose continuous scores.")


class BaseNoveltyModel(BaseBudgetedModel, ABC):
    """Base class for novelty detectors (higher score_samples = more anomalous)."""

    task = "novelty"

    def _score_known_validation_fold(
        self,
        estimator: BaseEstimator,
        X_val: np.ndarray,
        params: dict[str, Any],
    ) -> float:
        raw_preds = estimator.predict(X_val)  # sklearn: +1 inlier, -1 outlier
        anomaly_scores = -estimator.decision_function(X_val)
        if float(np.std(anomaly_scores)) < 1e-12:
            return -1.0

        val_fpr = float(np.mean(raw_preds == -1))
        target_fpr = float(params.get("nu", params.get("contamination", 0.05)))
        excess_fpr_penalty = 0.5 * max(0.0, val_fpr - 0.25)
        return float(1.0 - abs(val_fpr - target_fpr) - excess_fpr_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseNoveltyModel:
        X_arr = as_2d_float64(X)
        if len(X_arr) < 2:
            raise ValueError(f"Need at least 2 samples for inner CV, got {len(X_arr)}.")
        precomputed = self._is_precomputed()
        if precomputed and (X_arr.ndim != 2 or X_arr.shape[0] != X_arr.shape[1]):
            raise ValueError(f"Precomputed kernel must be square 2D, got {X_arr.shape}.")

        configs = self._sample_configurations()
        n_splits = min(self.cv_folds, len(X_arr))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = list(kf.split(np.zeros(len(X_arr))))

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        self.cv_results_ = []

        for params in configs:
            fold_scores: list[float] = []
            for train_idx, val_idx in splits:
                est = self._build_estimator(params)
                X_tr = X_arr[np.ix_(train_idx, train_idx)] if precomputed else X_arr[train_idx]
                X_va = X_arr[np.ix_(val_idx, train_idx)] if precomputed else X_arr[val_idx]
                est.fit(X_tr)
                fold_scores.append(self._score_known_validation_fold(est, X_va, params))

            mean_score = float(np.mean(fold_scores))
            self.cv_results_.append({"params": params, "mean_calibration_score": mean_score})
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.n_configs_evaluated_ = len(configs)
        self.best_params_ = dict(best_params)
        self.best_cv_score_ = best_score

        final_estimator = self._build_estimator(self.best_params_)
        final_estimator.fit(X_arr)
        self.best_estimator_ = final_estimator
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        raw_preds = self.best_estimator_.predict(as_2d_float64(X))
        return (raw_preds == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling score_samples().")
        return -np.asarray(
            self.best_estimator_.decision_function(as_2d_float64(X)), dtype=np.float64
        )


BaseFewShotBaseline = BaseFewShotModel
BaseNoveltyBaseline = BaseNoveltyModel