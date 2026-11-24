#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest


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


#-----------------------------------------------------------------------------------------------
# Few-Shot classifiers
#-----------------------------------------------------------------------------------------------

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


# SVM with RBF kernel

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

# SVM with Laplacian kernel
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


# k-Nearest Neighbors (kNN)


class _SafeKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """KNeighborsClassifier wrapper that clamps n_neighbors to the training fold size."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SafeKNeighborsClassifier:
        X_arr = np.asarray(X, dtype=np.float64)
        effective_k = max(1, min(int(self.n_neighbors), len(X_arr)))
        self.knn_ = KNeighborsClassifier(
            n_neighbors=effective_k,
            weights=self.weights,
            metric=self.metric,
        )
        self.knn_.fit(X_arr, y)
        self.classes_ = self.knn_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.knn_.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.knn_.predict_proba(np.asarray(X, dtype=np.float64))


class KNNBaseline(BaseFewShotBaseline):
    """k-Nearest Neighbors classifier with fold-safe neighborhood bounds."""

    model_name = "knn"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "n_neighbors": [1, 3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "chebyshev"],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _SafeKNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"],
        )


# Logistic Regression

class LogisticRegressionBaseline(BaseFewShotBaseline):
    """Multinomial Logistic Regression with L2 regularization C in the search budget."""

    model_name = "logistic_regression"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
            "class_weight": [None, "balanced"],
            "fit_intercept": [True, False],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            fit_intercept=params["fit_intercept"],
            max_iter=1000,
            random_state=self.seed,
        )


# Histogram Gradient Boosting

class HistGradientBoostingBaseline(BaseFewShotBaseline):
    """Histogram-based Gradient Boosting with low min_samples_leaf for k-shot regimes."""

    model_name = "hist_gradient_boosting"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "learning_rate": [0.03, 0.1, 0.2],
            "max_iter": [50, 100, 200],
            "max_leaf_nodes": [7, 15, 31],
            "min_samples_leaf": [1, 2, 4, 8],
            "l2_regularization": [0.0, 1.0],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            early_stopping=False,
            random_state=self.seed,
        )


#-----------------------------------------------------------------------------------------------
# NOVELTY classifiers
#-----------------------------------------------------------------------------------------------


class BaseNoveltyBaseline(BaseBudgetedModel, ABC):
    """Base class for all 3 classical novelty detectors (Protocol 2).

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

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseNoveltyBaseline:
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


# RBF One-Class SVM

class RBFOCSVMBaseline(BaseNoveltyBaseline):
    """One-Class SVM with RBF kernel tuning nu and gamma on known families."""

    model_name = "rbf_ocsvm"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "nu": [0.01, 0.05, 0.1, 0.15, 0.2],
            "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return OneClassSVM(
            kernel="rbf",
            nu=params["nu"],
            gamma=params["gamma"],
        )


# Isolation Forest

class IsolationForestBaseline(BaseNoveltyBaseline):
    """Isolation Forest novelty detector tuned strictly on known families."""

    model_name = "isolation_forest"

    def _param_grid(self) -> dict[str, list[Any]]:
        # >>> MODIFICA QUI I PARAMETRI PER GIOCARE CON ISOLATION FOREST <<<
        return {
            "n_estimators": [100, 200, 300],
            "max_samples": ["auto", 0.5, 0.75, 1.0],
            "contamination": [0.01, 0.05, 0.1, 0.15],
            "max_features": [0.5, 0.75, 1.0],
            "bootstrap": [False, True],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return IsolationForest(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            contamination=params["contamination"],
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            random_state=self.seed,
        )


# Local Outlier Factor

class _SafeLocalOutlierFactor(BaseEstimator):
    """LocalOutlierFactor (novelty=True) wrapper clamping n_neighbors to fold size."""

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "euclidean",
        contamination: float = 0.05,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.contamination = contamination

    def fit(self, X: np.ndarray, y: Any = None) -> _SafeLocalOutlierFactor:
        X_arr = np.asarray(X, dtype=np.float64)
        effective_k = max(1, min(int(self.n_neighbors), len(X_arr) - 1))
        self.lof_ = LocalOutlierFactor(
            n_neighbors=effective_k,
            metric=self.metric,
            contamination=self.contamination,
            novelty=True,
        )
        self.lof_.fit(X_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lof_.predict(np.asarray(X, dtype=np.float64))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.lof_.decision_function(np.asarray(X, dtype=np.float64))


class LOFBaseline(BaseNoveltyBaseline):
    """Local Outlier Factor novelty detector tuned strictly on known families."""

    model_name = "lof"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "n_neighbors": [5, 10, 15, 20, 30],
            "metric": ["euclidean", "manhattan", "chebyshev"],
            "contamination": [0.01, 0.05, 0.1, 0.15],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _SafeLocalOutlierFactor(
            n_neighbors=params["n_neighbors"],
            metric=params["metric"],
            contamination=params["contamination"],
        )


BASELINE_REGISTRY: dict[str, type[BaseBudgetedModel]] = {
    # Few-shot classifiers
    "rbf_svm": RBFSVMBaseline,
    "laplacian_svm": LaplacianSVMBaseline,
    "knn": KNNBaseline,
    "logistic_regression": LogisticRegressionBaseline,
    "hist_gradient_boosting": HistGradientBoostingBaseline,
    # Novelty detectors
    "rbf_ocsvm": RBFOCSVMBaseline,
    "isolation_forest": IsolationForestBaseline,
    "lof": LOFBaseline,
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