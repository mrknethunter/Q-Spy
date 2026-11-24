#!/usr/bin/env python3
"""Shared model interfaces and equal-budget hyperparameter tuning for FewShotSVC, NoveltyOCSVM and baseline models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold


class BaseBudgetedModel(ABC):
    """Common interface and equal-budget hyperparameter tuning.

    Every model receives the exact same maximum number of hyperparameter
    configurations (`budget`) and a deterministic `seed` for reproducibility.
    """

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
        """Return the candidate hyperparameter search space."""

    @abstractmethod
    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        """Instantiate a fresh underlying estimator for a given parameter dict."""

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
        """Ensure StratifiedKFold never fails on small k-shot regimes (e.g., k=5)."""
        _, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min())
        if min_class_count < 2:
            raise ValueError(
                f"Need at least 2 samples per class for inner CV, got {min_class_count}."
            )
        return min(self.cv_folds, min_class_count)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseBudgetedModel:
        """Tune hyperparameters within `self.budget` and fit on full training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (few-shot) or inlier/outlier flags (novelty)."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return decision scores (class margins for few-shot, anomaly scores for novelty)."""

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


# -----------------------------------------------------------------------------------------------
# Few-Shot base model
# -----------------------------------------------------------------------------------------------


class BaseFewShotModel(BaseBudgetedModel, ABC):
    """Base class for few-shot classifiers (both classical baselines and QSVC)."""

    task = "fewshot"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseFewShotModel:
        if y is None:
            raise ValueError("Few-shot classification requires target labels `y`.")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        configs = self._sample_configurations()
        n_splits = self._resolve_n_splits(y_arr)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = list(skf.split(X_arr, y_arr))

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        self.cv_results_ = []

        for params in configs:
            fold_scores: list[float] = []
            for train_idx, val_idx in splits:
                est = self._build_estimator(params)
                est.fit(X_arr[train_idx], y_arr[train_idx])
                preds = est.predict(X_arr[val_idx])
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
        return self.best_estimator_.predict(np.asarray(X, dtype=np.float64))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling score_samples().")
        X_arr = np.asarray(X, dtype=np.float64)
        if hasattr(self.best_estimator_, "decision_function"):
            return self.best_estimator_.decision_function(X_arr)
        if hasattr(self.best_estimator_, "predict_proba"):
            return self.best_estimator_.predict_proba(X_arr)
        raise NotImplementedError(f"{self.model_name} does not expose continuous scores.")


# -----------------------------------------------------------------------------------------------
# Novelty Detection base model
# -----------------------------------------------------------------------------------------------


class BaseNoveltyModel(BaseBudgetedModel, ABC):
    """Base class for novelty detectors (both classical baselines and NoveltyOCSVM).

    Conventions:
    - `fit(X, y=None)` expects `X` to contain ONLY known glitch families (no outliers).
    - `predict(X)` returns `0` for known (inlier) and `1` for novel/anomalous (outlier).
    - `score_samples(X)` returns continuous anomaly scores where higher = more anomalous.
    """

    task = "novelty"

    def _resolve_novelty_n_splits(self, n_samples: int) -> int:
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples for inner CV, got {n_samples}.")
        return min(self.cv_folds, n_samples)

    def _score_known_validation_fold(
        self,
        estimator: BaseEstimator,
        X_val: np.ndarray,
        params: dict[str, Any],
    ) -> float:
        """Evaluate hyperparameter calibration on a known-only validation fold."""
        raw_preds = estimator.predict(X_val)  # sklearn: +1 inlier, -1 outlier
        anomaly_scores = -estimator.decision_function(X_val)

        # Reject collapsed/constant kernels that cannot rank anomalies
        if float(np.std(anomaly_scores)) < 1e-12:
            return -1.0

        val_fpr = float(np.mean(raw_preds == -1))
        target_fpr = float(params.get("nu", params.get("contamination", 0.05)))
        excess_fpr_penalty = 0.5 * max(0.0, val_fpr - 0.25)
        return float(1.0 - abs(val_fpr - target_fpr) - excess_fpr_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseNoveltyModel:
        X_arr = np.asarray(X, dtype=np.float64)
        configs = self._sample_configurations()
        n_splits = self._resolve_novelty_n_splits(len(X_arr))

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = list(kf.split(X_arr))

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        self.cv_results_ = []

        for params in configs:
            fold_scores: list[float] = []
            for train_idx, val_idx in splits:
                est = self._build_estimator(params)
                est.fit(X_arr[train_idx])
                fold_scores.append(
                    self._score_known_validation_fold(est, X_arr[val_idx], params)
                )

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
        raw_preds = self.best_estimator_.predict(np.asarray(X, dtype=np.float64))
        return (raw_preds == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling score_samples().")
        X_arr = np.asarray(X, dtype=np.float64)
        return -np.asarray(self.best_estimator_.decision_function(X_arr), dtype=np.float64)


# Backwards-compatible aliases used by models/baselines.py
BaseFewShotBaseline = BaseFewShotModel
BaseNoveltyBaseline = BaseNoveltyModel