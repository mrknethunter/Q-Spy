#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics.pairwise import laplacian_kernel, euclidean_distances
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
            "best_cv_score": float(self.best_cv_score_) if np.isfinite(self.best_cv_score_) else None,
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


def novelty_loco_splits(y, seed=42):
    """Indices for surrogate-family LOCO and independent known calibration.

    The true unknown family is never permitted here. Splits use labels/seed,
    never feature values or test outcomes. No preprocessing is fitted here.
    """
    labels = np.asarray(y)
    if labels.ndim != 1 or "Fast_Scattering" in set(map(str, labels)):
        raise ValueError("novelty tuning requires known families only; Fast_Scattering is forbidden")
    families = np.unique(labels)
    if len(families) < 2:
        raise ValueError("LOCO requires at least two known families")
    rng = np.random.default_rng(seed)
    splits = []
    for family in families:
        fit, calibration = [], []
        for known in families[families != family]:
            indices = rng.permutation(np.flatnonzero(labels == known))
            if len(indices) < 2:
                raise ValueError("LOCO requires at least two rows per known family")
            n_cal = max(1, int(np.ceil(.2 * len(indices))))
            calibration.extend(indices[:n_cal])
            fit.extend(indices[n_cal:])
        splits.append((str(family), np.asarray(fit), np.asarray(calibration),
                       np.flatnonzero(labels == family)))
    return splits


def training_distance_median(X):
    """Median of strictly upper-triangular squared training distances."""
    distances = euclidean_distances(X, squared=True)
    median = float(np.median(distances[np.triu_indices(len(X), 1)]))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("positive training-distance median required for the gamma grid")
    return median


class BaseNoveltyBaseline(BaseBudgetedModel, ABC):
    """Shared known-family LOCO tuning. Larger scores mean more anomalous.

    Pass family labels for supervised *surrogate* tuning, never true anomalies.
    validation=(X_known, y_known) calibrates the final operating point. The
    optional inner_representations contain independently preprocessed inner
    fit/calibration/surrogate arrays, keyed by family, in the splitter order.
    Feature-only fit remains an explicitly untuned compatibility heuristic.
    """
    task = "novelty"
    target_fpr = .05

    def _score_known_validation_fold(self, estimator, X_val, X_surrogate):
        scores = -estimator.decision_function(np.concatenate([X_val, X_surrogate]))
        labels = np.r_[np.zeros(len(X_val)), np.ones(len(X_surrogate))]
        return float(roc_auc_score(labels, scores))

    def _calibrate_nu(self, X_fit, X_known, gamma, nu_grid):
        records, estimators = [], []
        for nu in nu_grid:
            estimator = self._build_estimator(dict(gamma=float(gamma), nu=float(nu)))
            estimator.fit(X_fit)
            fpr = float(np.mean(estimator.predict(X_known) == -1))
            records.append(dict(nu=float(nu), native_known_fpr=fpr))
            estimators.append(estimator)
        # A deterministic tie favours smaller nu; anomaly labels never enter.
        chosen = min(range(len(records)), key=lambda i: (
            abs(records[i]["native_known_fpr"] - self.target_fpr), records[i]["nu"]))
        return estimators[chosen], records[chosen], records

    def fit(self, X, y=None, *, validation=None, inner_representations=None):
        X = np.asarray(X, dtype=np.float64)
        if len(X) < 2 or X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("finite two-dimensional training data with at least two rows required")
        self.cv_results_ = []
        if y is None:
            # Without family labels there is no honest surrogate CV objective.
            params = next(iter(ParameterGrid(self._param_grid())))
            if self.model_name == "rbf_ocsvm":
                params = dict(gamma=1 / training_distance_median(X), nu=.05)
            elif self.model_name == "isolation_forest":
                params = dict(n_estimators=100, max_samples="auto", contamination=.05,
                              max_features=1., bootstrap=False)
            else:
                params = dict(n_neighbors=20, metric="euclidean", contamination=.05)
            self.best_params_, self.best_cv_score_ = params, -np.inf
            self.best_estimator_ = self._build_estimator(params).fit(X)
            self.n_configs_evaluated_ = 1
            self.tuning_mode_ = "unlabelled_fixed_heuristic_no_loco"
            self.selection_audit_ = dict(mode=self.tuning_mode_, target_known_fpr=self.target_fpr)
            self.operating_threshold_ = 0.
            return self
        y = np.asarray(y)
        if len(y) != len(X):
            raise ValueError("family labels and training rows differ")
        splits = novelty_loco_splits(y, self.seed)
        inner_X = X
        if validation is not None:
            known_X, known_y = validation
            known_X, known_y = np.asarray(known_X), np.asarray(known_y)
            if len(known_X) != len(known_y) or not len(known_X) or not set(known_y) <= set(y):
                raise ValueError("validation must contain known training families only")
        else:
            # Compatibility path: final calibration holds out known rows and
            # does NOT refit on them. Production E03 supplies external validation.
            _, fit_idx, cal_idx, _ = splits[0]
            # Include the first family among final fit rows, never calibration.
            fit_idx = np.r_[fit_idx, splits[0][3]]
            known_X, X = X[cal_idx], X[fit_idx]
        prepared = []
        for family, fit_idx, cal_idx, pseudo_idx in splits:
            arrays = ((np.asarray(inner_representations[family][0]),
                       np.asarray(inner_representations[family][1]),
                       np.asarray(inner_representations[family][2]))
                      if inner_representations is not None else
                      (inner_X[fit_idx], inner_X[cal_idx], inner_X[pseudo_idx]))
            if tuple(map(len, arrays)) != (len(fit_idx), len(cal_idx), len(pseudo_idx)):
                raise ValueError("inner representation sizes do not match the shared LOCO split")
            prepared.append((family, *arrays))
        self.tuning_mode_ = "nested_known_family_loco_auroc"
        if self.model_name == "rbf_ocsvm":
            nu_grid = np.array([.001, .005, .01, .05, .1, .2])
            c_grid = np.logspace(-2, 2, 9)
            if self.budget < 54:
                # Small synthetic smoke tests / explicitly exploratory budgets.
                nu_grid = np.array([.05])
                c_grid = c_grid[np.unique(np.linspace(0, 8, min(9, self.budget), dtype=int))]
            medians = [training_distance_median(fit) for _, fit, _, _ in prepared]
            for c in c_grid:
                folds = []
                for (family, fit, known, surrogate), median in zip(prepared, medians):
                    estimator, chosen, calibration = self._calibrate_nu(fit, known, c / median, nu_grid)
                    folds.append(dict(surrogate_family=family, gamma=float(c / median),
                                      training_distance_median=median,
                                      surrogate_auroc=self._score_known_validation_fold(estimator, known, surrogate),
                                      **chosen, nu_calibration=calibration))
                self.cv_results_.append(dict(params=dict(c=float(c)),
                    mean_surrogate_auroc=float(np.mean([f["surrogate_auroc"] for f in folds])), folds=folds))
            best = max(self.cv_results_, key=lambda r: (r["mean_surrogate_auroc"], -r["params"]["c"]))
            median = training_distance_median(X)
            gamma = best["params"]["c"] / median
            estimator, chosen, final_calibration = self._calibrate_nu(X, known_X, gamma, nu_grid)
            params = dict(gamma=gamma, nu=chosen["nu"])
            self.n_configs_evaluated_ = len(c_grid) * len(nu_grid)
            extra = dict(c=best["params"]["c"], training_distance_median=median,
                         c_grid=c_grid.tolist(), nu_grid=nu_grid.tolist(),
                         final_nu_calibration=final_calibration,
                         final_native_known_fpr=chosen["native_known_fpr"],
                         preregistered_gamma_grid_complete=bool(len(c_grid) == 9 and len(nu_grid) == 6))
        else:
            configs = self._sample_configurations()
            forest_group, forest_folds = None, {}
            for params in configs:
                if self.model_name == "isolation_forest":
                    group = tuple((key, value) for key, value in params.items() if key != "n_estimators")
                    if group != forest_group:
                        forest_group, forest_folds = group, {}
                folds = []
                for family, fit, known, surrogate in prepared:
                    if self.model_name == "isolation_forest":
                        # ParameterGrid order groups identical forest settings.
                        # Warm-start extends the same seeded tree sequence exactly;
                        # discard the cache between groups to bound memory.
                        estimator = forest_folds.get(family)
                        if estimator is None or estimator.n_estimators >= params["n_estimators"]:
                            estimator = self._build_estimator(params).set_params(warm_start=True)
                        else:
                            estimator.set_params(n_estimators=params["n_estimators"])
                        estimator.fit(fit)
                        forest_folds[family] = estimator
                    else:
                        estimator = self._build_estimator(params).fit(fit)
                    folds.append(dict(surrogate_family=family,
                        surrogate_auroc=self._score_known_validation_fold(estimator, known, surrogate)))
                self.cv_results_.append(dict(params=params,
                    mean_surrogate_auroc=float(np.mean([f["surrogate_auroc"] for f in folds])), folds=folds))
            best = max(self.cv_results_, key=lambda r: r["mean_surrogate_auroc"])
            params = best["params"]
            estimator = self._build_estimator(params).fit(X)
            self.n_configs_evaluated_ = len(configs)
            extra = {}
        ranked = sorted((row["mean_surrogate_auroc"] for row in self.cv_results_), reverse=True)
        extra["best_minus_runner_up_auroc"] = float(ranked[0] - ranked[1]) if len(ranked) > 1 else None
        extra["candidates_within_0_01_of_best"] = sum(ranked[0] - value < .01 for value in ranked)
        self.best_params_ = dict(params)
        self.best_cv_score_ = best["mean_surrogate_auroc"]
        self.best_estimator_ = estimator
        known_scores = self.score_samples(known_X)
        self.operating_threshold_ = float(np.quantile(known_scores, .95, method="higher"))
        self.selection_audit_ = dict(mode=self.tuning_mode_, target_known_fpr=.05,
            calibrated_known_fpr=float(np.mean(known_scores > self.operating_threshold_)),
            operating_threshold=self.operating_threshold_, surrogate_families=[s[0] for s in splits],
            configuration_budget_is_ceiling=True, **extra)
        return self

    def predict(self, X):
        return (self.score_samples(X) > self.operating_threshold_).astype(int)

    def score_samples(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Model must be fitted before calling score_samples().")
        return -np.asarray(self.best_estimator_.decision_function(np.asarray(X, dtype=np.float64)), dtype=np.float64)


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
            "contamination": [0.05],
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
            "contamination": [0.05],
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