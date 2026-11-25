"""E-03-only fold-local feature and embedding adapter for shared M-04 models."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from features.transforms import FEATURE_COLUMNS, make_features


def _matrix_hash(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
    digest.update(matrix.tobytes())
    return digest.hexdigest()


def _id_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


class FoldRepresentationCache:
    """Fit the shared scaler on each model-training fold and cache its features.

    The cache is scoped to one outer case and one model. Its expected ID hashes
    are populated from the same deterministic KFold membership used by M-04;
    test/validation IDs are never registered as scaler-fit rows.
    """

    def __init__(self, *, representation="classical", n_qubits=8,
                 embed_fn=None, make_features_fn=None):
        if representation not in {"classical", "projected"}:
            raise ValueError("representation must be 'classical' or 'projected'")
        if representation == "projected" and embed_fn is None:
            raise ValueError("projected representation requires an L0 embedding function")
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError("n_qubits must be a positive integer")
        self.representation = representation
        self.n_qubits = n_qubits
        self.embed_fn = embed_fn
        self.make_features_fn = make_features_fn or make_features
        self.expected_fit_ids = {}
        self.fit_audit = []
        self._fit_cache = {}
        self._evaluation_cache = {}

    @staticmethod
    def _raw_matrix(values):
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"raw input must have shape (N, {len(FEATURE_COLUMNS)})")
        if not len(matrix) or not np.isfinite(matrix).all() or (matrix <= 0).any():
            raise ValueError("raw Omicron features must be finite and positive")
        return np.ascontiguousarray(matrix)

    def register_training_ids(self, values, ids, *, role: str):
        """Bind one exact raw training matrix to its allowed ID membership."""
        matrix = self._raw_matrix(values)
        id_list = list(map(str, ids))
        if len(id_list) != len(matrix) or len(set(id_list)) != len(id_list):
            raise ValueError("registered IDs must be unique and aligned with raw rows")
        if role not in {"inner_cv_training", "outer_training"}:
            raise ValueError("only training IDs may be registered for scaler fitting")
        key = _matrix_hash(matrix)
        record = {"role": role, "n_rows": len(matrix), "ids_sha256": _id_hash(id_list)}
        previous = self.expected_fit_ids.get(key)
        if previous is not None and previous != record:
            raise ValueError("identical raw matrices map to conflicting training-ID memberships")
        self.expected_fit_ids[key] = record
        return key

    def _as_frame(self, matrix):
        return pd.DataFrame(matrix, columns=FEATURE_COLUMNS)

    def _represent(self, transformed):
        if self.representation == "classical":
            return transformed, None
        bloch = np.asarray(self.embed_fn(transformed), dtype=np.float64)
        expected = (len(transformed), self.n_qubits, 3)
        if bloch.shape != expected or not np.isfinite(bloch).all():
            raise ValueError(f"L0 embedding must return finite Bloch vectors of shape {expected}")
        if (np.abs(bloch) > 1.0 + 1e-7).any():
            raise ValueError("Bloch components must lie in [-1, 1]")
        return bloch.reshape(len(bloch), -1) / np.sqrt(self.n_qubits), bloch

    def fit_transform(self, values):
        """Fit training-only preprocessing and return the model representation."""
        matrix = self._raw_matrix(values)
        key = _matrix_hash(matrix)
        if key not in self.expected_fit_ids:
            raise ValueError("scaler fit requested for unregistered/non-training rows")
        if key not in self._fit_cache:
            transformed, scaler = self.make_features_fn(self._as_frame(matrix))
            representation, bloch = self._represent(np.asarray(transformed, dtype=np.float64))
            membership = self.expected_fit_ids[key]
            if membership["n_rows"] != len(matrix):
                raise ValueError("registered scaler-fit row count differs from its raw matrix")
            record = membership | {
                "input_sha256": key,
                "scaler_center": np.asarray(scaler.center_, dtype=np.float64).tolist(),
                "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
            }
            self.fit_audit.append(record)
            self._fit_cache[key] = (scaler, representation, bloch)
        return self._fit_cache[key][1], key

    def transform(self, training_key: str, values):
        """Transform unseen rows with the scaler fitted for `training_key`."""
        if training_key not in self._fit_cache:
            raise ValueError("evaluation requested before its training-fold scaler was fitted")
        matrix = self._raw_matrix(values)
        evaluation_key = _matrix_hash(matrix)
        cache_key = (training_key, evaluation_key)
        if cache_key not in self._evaluation_cache:
            scaler = self._fit_cache[training_key][0]
            transformed, _ = self.make_features_fn(self._as_frame(matrix), scaler)
            representation, _ = self._represent(np.asarray(transformed, dtype=np.float64))
            self._evaluation_cache[cache_key] = representation
        return self._evaluation_cache[cache_key]


class FoldAwareEstimator(BaseEstimator):
    """Adapt one shared M-04 estimator without changing its search contract."""

    def __init__(self, estimator, representation_cache: FoldRepresentationCache):
        self.estimator = estimator
        self.representation_cache = representation_cache

    def fit(self, X, y=None):
        training, self.training_key_ = self.representation_cache.fit_transform(X)
        self.estimator.fit(training)
        return self

    def predict(self, X):
        return self.estimator.predict(self.representation_cache.transform(self.training_key_, X))

    def decision_function(self, X):
        return self.estimator.decision_function(
            self.representation_cache.transform(self.training_key_, X)
        )

    @property
    def _gamma(self):
        return getattr(self.estimator, "_gamma", None)


def attach_fold_preprocessing(model, representation_cache: FoldRepresentationCache):
    """Wrap shared candidate estimators with E-03 fold-local preprocessing.

    M-04 still owns candidate generation, KFold membership, model fitting,
    known-only calibration scoring and the equal-budget report. The wrapper
    ensures each candidate estimator fits a scaler (and, for R3, an exact
    embedding) only on the rows M-04 supplied to that estimator's `fit` call.
    """
    original_builder = model._build_estimator

    def build_fold_estimator(params):
        return FoldAwareEstimator(original_builder(params), representation_cache)

    model._build_estimator = build_fold_estimator
    model._e03_preprocessing_cache = representation_cache
    return model
