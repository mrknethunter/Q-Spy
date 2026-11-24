from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from quantum.embed import EmbedConfig, circuit_and_angles


def fidelity_kernel(X1: np.ndarray, X2: np.ndarray, cfg: EmbedConfig) -> np.ndarray:
    S1, S2 = _states(X1, cfg), _states(X2, cfg)
    return np.abs(S1.conj() @ S2.T) ** 2


def _states(X, cfg):
    qc, angles = circuit_and_angles(X, cfg)
    return np.array([Statevector(qc.assign_parameters(a)).data for a in angles])
