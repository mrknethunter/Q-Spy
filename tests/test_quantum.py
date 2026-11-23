from __future__ import annotations

import numpy as np
import pytest

from quantum.embed import EmbedConfig, embed
from tests.fixtures.synthetic import synthetic_curated, synthetic_features

X = synthetic_features(synthetic_curated())[:12]


def numpy_reference_bloch(X, cfg):
    """Independent dense-statevector implementation, so a Qiskit ordering or sign error cannot pass unnoticed."""
    n, d = cfg.n_qubits, X.shape[1]
    paulis = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]).astype(complex)]
    idx = np.arange(2 ** n)
    bit = lambda q: (idx >> q) & 1  # qubit q = bit q (little-endian, as in Qiskit)
    cz = np.prod([np.where(bit(q) & bit(q + 1), -1, 1) for q in range(n - 1)], axis=0) if n > 1 else 1
    out = []
    for x in X:
        psi = np.zeros(2 ** n, complex); psi[0] = 1
        for _ in range(cfg.layers):
            for q in range(n):
                t = cfg.s * np.pi * x[q % d]
                ry = np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
                u = np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)]) @ ry
                m = np.moveaxis(psi.reshape([2] * n), n - 1 - q, 0)
                psi = np.moveaxis((u @ m.reshape(2, -1)).reshape(m.shape), 0, n - 1 - q).reshape(-1)
            if cfg.entangle:
                psi = cz * psi
        rows = []
        for q in range(n):
            m = np.moveaxis(psi.reshape([2] * n), n - 1 - q, 0).reshape(2, -1)
            rho = m @ m.conj().T
            rows.append([np.real(np.trace(rho @ p)) for p in paulis])
        out.append(rows)
    return np.array(out)


@pytest.mark.parametrize("entangle, n_cz", [(True, 2 * 3), (False, 0)])
def test_feature_map_gates(entangle, n_cz):
    from quantum.feature_map import feature_map
    qc = feature_map(n_qubits=4, layers=2, d=7, entangle=entangle)
    ops = qc.count_ops()
    assert ops.get("cz", 0) == n_cz and ops["ry"] == 8 and ops["rz"] == 8
    assert qc.num_parameters == 4


@pytest.mark.parametrize("n", [2, 7, 8, 14])
def test_feature_map_cz_chain_has_two_moments_per_layer(n):
    from quantum.feature_map import feature_map
    qc = feature_map(n_qubits=n, layers=3)
    assert qc.depth(lambda inst: inst.operation.num_qubits == 2) <= 2 * 3


def test_feature_map_reuploads_features_cyclically():
    from quantum.feature_map import feature_map
    qc = feature_map(n_qubits=9, layers=1, d=7)
    assert qc.num_parameters == 7


def test_bloch_of_zero_state_is_plus_z():
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorEstimator
    from quantum.observables import bloch_observables, evs_to_bloch
    evs = StatevectorEstimator().run([(QuantumCircuit(3), bloch_observables(3))]).result()[0].data.evs
    assert np.allclose(evs_to_bloch(evs, 3), [[[0, 0, 1]] * 3])


def test_embed_l0_product_state_matches_closed_form():
    cfg = EmbedConfig(n_qubits=3, layers=1, s=0.8, entangle=False)
    t = 0.8 * np.pi * X[:, :3]
    expected = np.stack([np.sin(t) * np.cos(t), np.sin(t) * np.sin(t), np.cos(t)], axis=-1)
    assert np.allclose(embed(X, cfg, "L0").B, expected, atol=1e-10)


@pytest.mark.parametrize("n, layers", [(4, 2), (9, 1)])
def test_embed_l0_matches_independent_statevector(n, layers):
    cfg = EmbedConfig(n_qubits=n, layers=layers, s=1.0)
    assert np.allclose(embed(X[:5], cfg, "L0").B, numpy_reference_bloch(X[:5], cfg), atol=1e-10)


def test_embed_rejects_unknown_tier():
    with pytest.raises(ValueError):
        embed(X, EmbedConfig(2, 1, 1.0), "L9")


def test_embed_l1_adds_unbiased_shot_noise_of_requested_size():
    cfg = EmbedConfig(n_qubits=3, layers=1, s=1.0)
    exact = embed(X, cfg, "L0").B
    diff = embed(X, cfg, "L1", precision=0.05, seed=1).B - exact
    assert 0.02 < diff.std() < 0.1 and abs(diff.mean()) < 0.03


def test_embed_l2_compiles_for_fake_kingston_and_stays_close_to_exact():
    cfg = EmbedConfig(n_qubits=3, layers=1, s=1.0)
    res = embed(X[:4], cfg, "L2", precision=0.05, seed=1)
    assert res.B.shape == (4, 3, 3)
    assert res.meta["backend_name"] == "fake_kingston" and len(res.meta["layout"]) == 3
    assert isinstance(res.meta["two_qubit_depth"], int)
    assert np.corrcoef(res.B.ravel(), embed(X[:4], cfg, "L0").B.ravel())[0, 1] > 0.8


def test_estimate_usage_matches_ibm_formula():
    from quantum.embed import estimate_usage
    # 120 points x 3 measurement settings x ceil(1/0.03^2)=1112 shots = 400 320 executions
    assert estimate_usage(n_points=120, precision=0.03) == pytest.approx(1.3 * (2 + 0.35e-3 * 400_320))


def test_embed_hw_is_a_dry_run_unless_submit(monkeypatch):
    import qiskit_ibm_runtime

    def no_network(*a, **k):
        raise AssertionError("dry run must not contact IBM")
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", no_network)
    res = embed(X, EmbedConfig(4, 2, 1.0), "HW", precision=0.03)
    assert res.B is None and res.meta["dry_run"] is True
    assert res.meta["estimated_qpu_s"] > 0


def test_embed_hw_submit_path_runs_locally_on_a_small_fake_device():
    """Same code path as IBM hardware, on a fake device small enough to simulate."""
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    cfg = EmbedConfig(n_qubits=3, layers=1, s=1.0)
    res = embed(X[:4], cfg, "HW", precision=0.05, submit=True, device=FakeManilaV2(), seed=1)
    assert res.B.shape == (4, 3, 3) and res.meta["dry_run"] is False
    assert res.meta["job_ids"] and len(res.meta["layout"]) == 3
    assert np.corrcoef(res.B.ravel(), embed(X[:4], cfg, "L0").B.ravel())[0, 1] > 0.7
