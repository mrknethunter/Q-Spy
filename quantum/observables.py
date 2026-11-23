"""Q-02: single-qubit Pauli observables and reshaping of Estimator output into Bloch vectors."""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def bloch_observables(n_qubits: int, layout=None) -> list[list[SparsePauliOp]]:
    """Nested list of shape (3n, 1): X_0, Y_0, Z_0, X_1, ... ; `layout` maps them onto a transpiled circuit."""
    obs = [SparsePauliOp.from_sparse_list([(p, [q], 1.0)], n_qubits) for q in range(n_qubits) for p in "XYZ"]
    if layout is not None:
        obs = [o.apply_layout(layout) for o in obs]
    return [[o] for o in obs]  # shape (3n, 1) broadcasts against parameter values of shape (N, d)


def evs_to_bloch(evs: np.ndarray, n_qubits: int) -> np.ndarray:
    """Estimator evs of shape (3n, N) -> Bloch vectors of shape (N, n, 3)."""
    return np.asarray(evs, dtype=np.float64).T.reshape(-1, n_qubits, 3)
