#!/usr/bin/env python3
"""Classical baseline models and factory registry (M-04)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import OneClassSVM, SVC

try:
    from kernels.classical import laplacian_kernel
except ImportError:
    from sklearn.metrics.pairwise import laplacian_kernel

from models.base import (
    GAMMA_GRID,
    BaseBudgetedModel,
    BaseFewShotBaseline,
    BaseNoveltyBaseline,
    resolve_gamma,
)

# -----------------------------------------------------------------------------------------------
# Few-Shot classifiers
# -----------------------------------------------------------------------------------------------


class RBFSVMBaseline(BaseFewShotBaseline):
    """Support Vector Classifier with RBF kernel and One-vs-Rest strategy."""

    model_name = "rbf_svm"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
            "gamma": list(GAMMA_GRID),
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            decision_function_shape="ovr",
            random_state=self.seed,
        )


class _LaplacianSVC(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible SVC wrapper using `kernels.classical.laplacian_kernel`."""

    def __init__(
        self,
        C: float = 1.0,
        gamma: float | str | None = "scale",
        random_state: int | None = None,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> _LaplacianSVC:
        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.gamma_ = resolve_gamma(self.X_train_, self.gamma)
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
        K_test = laplacian_kernel(np.asarray(X, dtype=np.float64), self.X_train_, gamma=self.gamma_)
        return self.svc_.predict(K_test)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K_test = laplacian_kernel(np.asarray(X, dtype=np.float64), self.X_train_, gamma=self.gamma_)
        return self.svc_.decision_function(K_test)


class LaplacianSVMBaseline(BaseFewShotBaseline):
    """Support Vector Classifier with Laplacian kernel and One-vs-Rest strategy."""

    model_name = "laplacian_svm"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "C": [1e-2, 1e-1, 1.0, 10.0, 100.0],
            "gamma": list(GAMMA_GRID),
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return _LaplacianSVC(
            C=params["C"],
            gamma=params["gamma"],
            random_state=self.seed,
        )


class _SafeKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """KNeighborsClassifier wrapper clamping n_neighbors to the training fold size."""

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


# -----------------------------------------------------------------------------------------------
# Novelty detectors
# -----------------------------------------------------------------------------------------------


class RBFOCSVMBaseline(BaseNoveltyBaseline):
    """One-Class SVM with RBF kernel tuning nu and gamma on known families."""

    model_name = "rbf_ocsvm"

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "nu": [0.01, 0.05, 0.1, 0.15, 0.2],
            "gamma": list(GAMMA_GRID),
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        return OneClassSVM(
            kernel="rbf",
            nu=params["nu"],
            gamma=params["gamma"],
        )


class IsolationForestBaseline(BaseNoveltyBaseline):
    """Isolation Forest novelty detector tuned strictly on known families."""

    model_name = "isolation_forest"

    def _param_grid(self) -> dict[str, list[Any]]:
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
    "rbf_svm": RBFSVMBaseline,
    "laplacian_svm": LaplacianSVMBaseline,
    "knn": KNNBaseline,
    "logistic_regression": LogisticRegressionBaseline,
    "hist_gradient_boosting": HistGradientBoostingBaseline,
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