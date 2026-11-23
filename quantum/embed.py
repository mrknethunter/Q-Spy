"""Quantum embedding (L-02 contract; Q-03 exact tier). See docs/contracts.md."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from quantum.feature_map import feature_map
from quantum.observables import bloch_observables, evs_to_bloch


@dataclass(frozen=True)
class EmbedConfig:
    n_qubits: int
    layers: int
    s: float
    entangle: bool = True


@dataclass
class EmbedResult:
    B: np.ndarray  # (N, n_qubits, 3): <X_q>, <Y_q>, <Z_q>
    meta: dict = field(default_factory=dict)


def embed(X_tilde: np.ndarray, cfg: EmbedConfig, backend: str = "L0", **backend_options) -> EmbedResult:
    """Bloch vectors of U_{s,L}(x)|0> for every row of X_tilde, as one broadcast Estimator job."""
    X_tilde = np.asarray(X_tilde, dtype=np.float64)
    qc = feature_map(cfg.n_qubits, cfg.layers, X_tilde.shape[1], cfg.entangle)
    angles = cfg.s * np.pi * X_tilde[:, : qc.num_parameters]
    meta = {"tier": backend, "embed_config": asdict(cfg), "n_points": len(X_tilde)}
    if backend == "L0":
        # Aer computes exact expectation values from the statevector (no sampling): ~70x faster than StatevectorEstimator.
        est = AerEstimator(options={"backend_options": {"method": "statevector"}, "run_options": {"approximation": True}})
        evs = est.run([(qc, bloch_observables(cfg.n_qubits), angles)]).result()[0].data.evs
        return EmbedResult(evs_to_bloch(evs, cfg.n_qubits), meta)
    raise ValueError(f"unknown tier {backend!r}: expected L0, L1, L2 or HW")
