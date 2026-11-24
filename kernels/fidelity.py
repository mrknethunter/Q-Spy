from __future__ import annotations

import numpy as np
from qiskit_aer import AerSimulator

from quantum.embed import EmbedConfig, circuit_and_angles


def fidelity_kernel(X1: np.ndarray, X2: np.ndarray, cfg: EmbedConfig) -> np.ndarray:
    S1, S2 = _states(X1, cfg), _states(X2, cfg)
    return np.abs(S1.conj() @ S2.T) ** 2


def _states(X, cfg):
    qc, angles = circuit_and_angles(X, cfg)
    qc = qc.copy()
    qc.save_statevector()
    # one Aer run for all points: ~7x faster than building a Statevector per point
    binds = [{p: angles[:, i] for i, p in enumerate(qc.parameters)}]
    res = AerSimulator(method="statevector").run(qc, parameter_binds=binds).result()
    return np.array([np.asarray(res.get_statevector(i)) for i in range(len(angles))])
