#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import laplacian_kernel


class BaseBudgetedModel(ABC):
    """Common interface and equal-budget hyperparameter tuning.

    Every model receives the exact same maximum number of hyperparameter
    configurations and a deterministic seed for reproducibility.
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

        # Populated during fit()
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


class BaseFewShotBaseline(BaseBudgetedModel, ABC):
    """Base class for few-shot classifiers."""

    task = "fewshot"

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseFewShotBaseline:
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


class RBFSVMBaseline(BaseFewShotBaseline):
    """Support Vector Classifier with RBF kernel and One-vs-Rest strategy."""

    model_name = "rbf_svm"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            decision_function_shape="ovr",
            random_state=self.seed,
        )

# Scikit-learn doesn't natively support Laplacian, this is a wrapper class for the Laplacian kernel

class _LaplacianSVC(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible SVC wrapper using an exact Laplacian kernel."""

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | str | None = "scale",
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

    def _resolve_gamma(self, X: np.ndarray) -> float:
        n_features = X.shape[1]
        if self.gamma is None or self.gamma == "auto":
            return 1.0 / n_features
        if self.gamma == "scale":
            x_var = float(np.var(X))
            return 1.0 / (n_features * x_var) if x_var > 0.0 else 1.0 / n_features
        return float(self.gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _LaplacianSVC:
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.gamma_ = self._resolve_gamma(self.X_train_)
        K_train = laplacian_kernel(self.X_train_, self.X_train_, gamma=self.gamma_)
        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            decision_function_shape="ovr",
            random_state=self.random_state,
        )
        self.svc_.fit(K_train, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_test = laplacian_kernel(
            np.asarray(X, dtype=np.float64), self.X_train_, gamma=self.gamma_
        )
        return self.svc_.predict(K_test)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K_test = laplacian_kernel(
            np.asarray(X, dtype=np.float64), self.X_train_, gamma=self.gamma_
        )
        return self.svc_.decision_function(K_test)


class LaplacianSVMBaseline(BaseFewShotBaseline):
    """Support Vector Classifier with Laplacian kernel and One-vs-Rest strategy."""

    model_name = "laplacian_svm"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _LaplacianSVC(
            C=params["C"],
            gamma=params["gamma"],
            random_state=self.seed,
        )


BASELINE_REGISTRY: dict[str, type[BaseBudgetedModel]] = {
    "rbf_svm": RBFSVMBaseline,
    "laplacian_svm": LaplacianSVMBaseline,
}


def build_baseline(
    name: str,
    budget: int = 16,
    seed: int = 42,
    cv_folds: int = 5,
) -> BaseBudgetedModel:
    """Factory function invoked by the YAML experiment runner."""
    key = name.strip().lower()
    if key not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY.keys()))
        raise ValueError(f"Unknown baseline '{name}'. Available: [{available}]")
    return BASELINE_REGISTRY[key](budget=budget, seed=seed, cv_folds=cv_folds)