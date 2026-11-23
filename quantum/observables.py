from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def bloch_observables(n_qubits: int, layout=None) -> list[list[SparsePauliOp]]:
    obs = [SparsePauliOp.from_sparse_list([(p, [q], 1.0)], n_qubits) for q in range(n_qubits) for p in "XYZ"]
    if layout is not None:
        obs = [o.apply_layout(layout) for o in obs]
    return [[o] for o in obs]  # not np.array(dtype=object): numpy would unpack each SparsePauliOp


def evs_to_bloch(evs: np.ndarray, n_qubits: int) -> np.ndarray:
    return np.asarray(evs, dtype=np.float64).T.reshape(-1, n_qubits, 3)
