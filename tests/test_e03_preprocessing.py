"""Synthetic checks that E-03 scaling is fitted inside each M-04 training fold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from features.transforms import FEATURE_COLUMNS, make_features
from kernels.projected import projected_kernel
from models.baselines import build_baseline
from experiments.e03_preprocessing import FoldRepresentationCache, _matrix_hash, attach_fold_preprocessing


def _ids_hash(ids):
    import hashlib
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def test_scalers_fit_only_their_inner_training_ids_and_outer_train():
    rng = np.random.default_rng(91)
    raw = np.exp(rng.normal(size=(24, len(FEATURE_COLUMNS))))
    ids = np.array([f"event-{i:03d}" for i in range(len(raw))])
    folds = list(KFold(n_splits=3, shuffle=True, random_state=17).split(raw))
    fitted_rows = []

    def watched_make_features(frame, scaler=None):
        if scaler is None:
            fitted_rows.append(frame.to_numpy(dtype=np.float64).copy())
        return make_features(frame, scaler)

    cache = FoldRepresentationCache(make_features_fn=watched_make_features)
    expected_by_hash = {}
    for train_indices, _ in folds:
        key = cache.register_training_ids(raw[train_indices], ids[train_indices], role="inner_cv_training")
        expected_by_hash[key] = set(train_indices.tolist())
    outer_key = cache.register_training_ids(raw, ids, role="outer_training")
    expected_by_hash[outer_key] = set(range(len(raw)))

    model = build_baseline("lof", budget=3, seed=17, cv_folds=3)
    attach_fold_preprocessing(model, cache)
    model.fit(raw)
    validation_scores = model.score_samples(raw[:4])
    assert validation_scores.shape == (4,)

    assert len(fitted_rows) == 4  # three inner training folds plus one outer refit
    for matrix in fitted_rows:
        key = _matrix_hash(matrix)
        assert key in expected_by_hash
    assert {record["role"] for record in cache.fit_audit} == {"inner_cv_training", "outer_training"}
    assert {record["ids_sha256"] for record in cache.fit_audit
            if record["role"] == "inner_cv_training"} == {
        _ids_hash(ids[train_indices]) for train_indices, _ in folds
    }
    assert next(record for record in cache.fit_audit if record["role"] == "outer_training")["ids_sha256"] == _ids_hash(ids)


def test_projected_representation_matches_contractual_kernel_normalization():
    rng = np.random.default_rng(5)
    raw = np.exp(rng.normal(size=(12, len(FEATURE_COLUMNS))))
    ids = [f"quantum-{i}" for i in range(len(raw))]

    def exact_fixture_embedding(transformed):
        bloch = np.zeros((len(transformed), 8, 3), dtype=np.float64)
        bloch[:, :7, 0] = transformed
        bloch[:, 7, 0] = transformed[:, 0]
        bloch[:, :, 2] = 0.5
        return bloch

    cache = FoldRepresentationCache(representation="projected", n_qubits=8,
                                     embed_fn=exact_fixture_embedding)
    training_key = cache.register_training_ids(raw[:8], ids[:8], role="outer_training")
    train_rep, observed_key = cache.fit_transform(raw[:8])
    eval_rep = cache.transform(training_key, raw[8:])
    assert observed_key == training_key
    B_train = cache._fit_cache[training_key][2]
    transformed_eval, _ = make_features(pd.DataFrame(raw[8:], columns=FEATURE_COLUMNS),
                                        cache._fit_cache[training_key][0])
    B_eval = exact_fixture_embedding(transformed_eval)
    np.testing.assert_allclose(
        np.exp(-0.7 * ((train_rep[:, None, :] - eval_rep[None, :, :]) ** 2).sum(axis=2)),
        projected_kernel(B_train, B_eval, gamma=0.7),
        rtol=1e-12, atol=1e-12,
    )
