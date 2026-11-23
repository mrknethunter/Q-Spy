"""L-02 contract tests. A test is skipped while its module is still a stub, and enforced once implemented."""
from __future__ import annotations

import numpy as np
import pytest

from tests.contracts.model_contract import check_fewshot_model
from tests.fixtures.synthetic import (CURATED_COLUMNS, FEATURES, HELD_OUT, synthetic_bloch, synthetic_curated,
                                      synthetic_features, synthetic_split)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"stub not implemented yet: {exc}")


@pytest.fixture(scope="module")
def curated():
    return synthetic_curated()


# --- fixtures honour the contract themselves --------------------------------------------------
def test_fixtures_follow_contract(curated):
    assert tuple(curated.columns) == CURATED_COLUMNS and curated["id"].is_unique
    assert set(curated.loc[curated["label"] == HELD_OUT, "run"]) <= {"O3a", "O3b"}
    X = synthetic_features(curated)
    assert X.shape == (len(curated), len(FEATURES)) and X.dtype == np.float64 and np.abs(X).max() <= 1.0
    s = synthetic_split(curated)
    assert not (set(s.train_ids) & set(s.val_ids) or set(s.train_ids) & set(s.test_ids) or set(s.val_ids) & set(s.test_ids))
    assert np.all(np.linalg.norm(synthetic_bloch(50, 6), axis=-1) <= 1.0 + 1e-12)


# --- quantum embedding --------------------------------------------------------------------------
@pytest.mark.parametrize("entangle", [True, False])
def test_embed_l0(curated, entangle):
    from quantum.embed import EmbedConfig, embed
    cfg = EmbedConfig(n_qubits=4, layers=2, s=1.0, entangle=entangle)
    res = _call(embed, synthetic_features(curated)[:20], cfg, "L0")
    assert res.B.shape == (20, 4, 3) and np.isfinite(res.B).all()
    assert np.all(np.linalg.norm(res.B, axis=-1) <= 1.0 + 1e-6)
    assert res.meta["tier"] == "L0" and res.meta["embed_config"]["n_qubits"] == 4


# --- store ----------------------------------------------------------------------------------------
def test_store_round_trip_and_hw_protection(tmp_path):
    from store.io import read_bloch, write_bloch
    ids, B = [f"syn-{i}" for i in range(7)], synthetic_bloch(7, 3)
    meta = {"tier": "HW", "embed_config": {"n_qubits": 3, "layers": 1, "s": 1.0, "entangle": True}}
    _call(write_bloch, "run-a", ids, B, meta, root=tmp_path)
    ids2, B2, meta2 = read_bloch("run-a", root=tmp_path)
    assert list(ids2) == ids and np.array_equal(B2, B) and meta2["tier"] == "HW"
    with pytest.raises(FileExistsError):
        write_bloch("run-a", ids, B, meta, root=tmp_path)


# --- kernels ----------------------------------------------------------------------------------------
def test_projected_kernel():
    from kernels.projected import projected_kernel
    B1, B2 = synthetic_bloch(9, 4, seed=1), synthetic_bloch(5, 4, seed=2)
    K = _call(projected_kernel, B1, B2, 0.7)
    d2 = ((B1[:, None] - B2[None]) ** 2).sum(axis=(-1, -2))
    assert K.shape == (9, 5) and np.allclose(K, np.exp(-0.7 / 4 * d2))
    K11 = projected_kernel(B1, B1, 0.7)
    assert np.allclose(K11, K11.T) and np.allclose(np.diag(K11), 1.0)


def test_rbf_kernel(curated):
    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    from kernels.classical import rbf_kernel
    X = synthetic_features(curated)[:30]
    assert np.allclose(_call(rbf_kernel, X[:10], X, 0.3), sk_rbf(X[:10], X, gamma=0.3))


# --- models (activates when M-04 is merged) ---------------------------------------------------------
def test_baseline_follows_model_contract(curated):
    baselines = pytest.importorskip("models.baselines")
    X, s = synthetic_features(curated), synthetic_split(curated)
    pos = {i: n for n, i in enumerate(curated["id"])}
    tr, te = [pos[i] for i in s.train_ids], [pos[i] for i in s.test_ids]
    y = curated["label"].to_numpy()
    check_fewshot_model(baselines.RBFSVMBaseline(budget=4, seed=0), X[tr], y[tr], X[te])
