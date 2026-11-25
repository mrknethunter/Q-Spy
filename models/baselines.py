#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import OneClassSVM, SVC

from models.base import BaseBudgetedModel, BaseFewShotBaseline, BaseNoveltyBaseline

#-----------------------------------------------------------------------------------------------
# Few-Shot classifiers
#-----------------------------------------------------------------------------------------------

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

# RBF One-Class SVM

def novelty_selection_splits(labels, *, seed, cv_folds):
    """Nested known-only folds, holding each known family out as a surrogate.

    Returns (fit, known-calibration, surrogate-anomaly) indices. Fast_Scattering
    is categorically forbidden. The real outer validation/test are not inputs.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(np.unique(labels)) < 3:
        raise ValueError("surrogate LOCO requires at least three known families")
    if any(str(label) == "Fast_Scattering" for label in labels):
        raise ValueError("Fast_Scattering is forbidden in hyperparameter selection")
    splits = []
    for family in np.unique(labels):
        others = np.flatnonzero(labels != family)
        surrogate = np.flatnonzero(labels == family)
        folds = KFold(n_splits=min(cv_folds, len(others)), shuffle=True, random_state=seed)
        for number, (fit, calibration) in enumerate(folds.split(others)):
            splits.append(dict(family=str(family), fold=number, train=others[fit],
                               calibration=others[calibration], surrogate=surrogate))
    return splits


def novelty_final_calibration_split(n_samples, *, seed, cv_folds):
    """Reserve known-training rows for final nu selection before full refitting."""
    return next(KFold(n_splits=min(cv_folds, n_samples), shuffle=True,
                      random_state=seed).split(np.zeros(n_samples)))


def _selection_estimator_metadata(estimator):
    while hasattr(estimator, "estimator"):
        estimator = estimator.estimator
    return dict(effective_gamma=getattr(estimator, "_gamma", None),
                median_squared_distance=getattr(estimator, "median_squared_distance_", None))


def _calibrate_nu(model, params, X_fit, X_known):
    """Choose nu using known-only FPR proximity to the preregistered 5% target.

    Surrogate anomaly scores are intentionally unavailable here. Finite-grid
    calibration error is reported, not hidden by a test-selected sign/offset.
    """
    nu_values = [0.01, 0.05, 0.1, 0.15, 0.2] if "gamma_factor" in params else [params.get("nu")]
    trials = []
    for nu in nu_values:
        resolved = dict(params)
        if nu is not None:
            resolved["nu"] = nu
        estimator = model._build_estimator(resolved)
        estimator.fit(X_fit)
        fpr = float(np.mean(estimator.predict(X_known) == -1))
        trials.append((abs(fpr - .05), float(nu or 0), estimator, resolved, fpr))
    chosen = min(trials, key=lambda trial: (trial[0], trial[1]))
    return chosen[2], chosen[3], dict(target_fpr=.05, observed_fpr=chosen[4],
        nu_trials=[dict(nu=trial[3].get("nu"), fpr=trial[4]) for trial in trials],
        **_selection_estimator_metadata(chosen[2]))


def novelty_candidate_configurations(model):
    """Use the shared deterministic budget sampler on the labelled novelty grid."""
    grid = model._loco_param_grid() if hasattr(model, "_loco_param_grid") else model._param_grid()
    all_configs = list(ParameterGrid(grid))
    if len(all_configs) > model.budget:
        positions = np.random.default_rng(model.seed).choice(len(all_configs), model.budget, replace=False)
        configs = [all_configs[int(index)] for index in sorted(positions)]
    else:
        configs = all_configs
    return configs, grid


def fit_surrogate_loco(model, X, labels):
    """Select by mean surrogate AUROC; preserve the shared unlabelled API.

    RBF OCSVM candidates are median-scaled gamma factors, with nu calibrated
    separately on known rows. Other detectors retain their shared grids, but
    use the same surrogate-discrimination objective and outer family folds.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if len(X) != len(labels):
        raise ValueError("known-family labels must align with training rows")
    splits = novelty_selection_splits(labels, seed=model.seed, cv_folds=model.cv_folds)
    configs, grid = novelty_candidate_configurations(model)
    model.cv_results_ = []
    for params in configs:
        scores, folds = [], []
        for split in splits:
            estimator, calibrated, calibration = _calibrate_nu(
                model, params, X[split["train"]], X[split["calibration"]])
            known = -np.asarray(estimator.decision_function(X[split["calibration"]]))
            novel = -np.asarray(estimator.decision_function(X[split["surrogate"]]))
            score = float(roc_auc_score(np.r_[np.zeros(len(known)), np.ones(len(novel))],
                                        np.r_[known, novel]))
            if not np.isfinite(score):
                raise ValueError("nonfinite surrogate LOCO AUROC")
            scores.append(score)
            folds.append(dict(surrogate_family=split["family"], fold=split["fold"],
                              n_train=len(split["train"]), n_known=len(known), n_surrogate=len(novel),
                              auroc=score, calibrated_params=calibrated, calibration=calibration))
        model.cv_results_.append(dict(params=params, mean_surrogate_auroc=float(np.mean(scores)), folds=folds))
    # Explicit deterministic tie break towards smaller gamma factor, then
    # canonical shared-grid order. No test data participates in this choice.
    best = min(range(len(configs)), key=lambda i: (-model.cv_results_[i]["mean_surrogate_auroc"],
               configs[i].get("gamma_factor", float("inf")), i))
    final_fit, final_cal = novelty_final_calibration_split(len(X), seed=model.seed, cv_folds=model.cv_folds)
    _, selected, calibration = _calibrate_nu(model, configs[best], X[final_fit], X[final_cal])
    model.best_params_ = selected
    model.best_cv_score_ = model.cv_results_[best]["mean_surrogate_auroc"]
    model.n_configs_evaluated_ = len(configs)
    model.tuning_strategy_ = "nested_surrogate_loco_auroc"
    model.selection_audit_ = dict(objective="mean_surrogate_auroc", nu_target_fpr=.05,
        final_nu_calibration=calibration, best_candidate=best, grid=grid,
        surrogate_families=sorted(map(str, np.unique(labels))), inner_splits=len(splits),
        tie_rule="smaller_gamma_factor_then_grid_order",
        configuration_count=len(configs), nu_calibration_fits=sum(
            len(fold["calibration"]["nu_trials"]) for row in model.cv_results_ for fold in row["folds"]))
    model.best_estimator_ = model._build_estimator(selected)
    model.best_estimator_.fit(X)
    model.selection_audit_["final_refit"] = _selection_estimator_metadata(model.best_estimator_)
    return model


class _MedianGammaOCSVM(BaseEstimator):
    """RBF OCSVM with gamma=c/median(off-diagonal squared fit distances)."""
    def __init__(self, nu, gamma_factor, distance_cache):
        self.nu, self.gamma_factor, self.distance_cache = nu, gamma_factor, distance_cache

    def fit(self, X, y=None):
        import hashlib
        X = np.asarray(X, dtype=np.float64)
        key = hashlib.sha256(str(X.shape).encode() + X.tobytes()).digest()
        if key not in self.distance_cache:
            distances = pdist(X, metric="sqeuclidean")
            self.distance_cache[key] = float(np.median(distances))
        self.median_squared_distance_ = self.distance_cache[key]
        if not np.isfinite(self.median_squared_distance_) or self.median_squared_distance_ <= 0:
            raise ValueError("median squared training distance is zero/nonfinite; gamma is undefined")
        self._gamma = float(self.gamma_factor / self.median_squared_distance_)
        self.model_ = OneClassSVM(nu=self.nu, gamma=self._gamma).fit(X)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)

class RBFOCSVMBaseline(BaseNoveltyBaseline):
    """One-Class SVM with RBF kernel tuning nu and gamma on known families."""

    model_name = "rbf_ocsvm"

    def _loco_param_grid(self):
        return {"gamma_factor": np.logspace(-2, 2, 9).tolist()}

    def _param_grid(self) -> dict[str, list[Any]]:
        return {
            "nu": [0.01, 0.05, 0.1, 0.15, 0.2],
            "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
        }

    def _build_estimator(self, params: dict[str, Any]) -> BaseEstimator:
        if "gamma_factor" in params:
            if not hasattr(self, "_distance_cache"):
                self._distance_cache = {}
            return _MedianGammaOCSVM(params["nu"], params["gamma_factor"], self._distance_cache)
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
