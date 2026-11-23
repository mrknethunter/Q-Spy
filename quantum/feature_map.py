"""Q-01: hardware-efficient re-uploading feature map U_{s,L}(x) of the proposal (Figure 2)."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def feature_map(n_qubits: int, layers: int, d: int = 7, entangle: bool = True) -> QuantumCircuit:
    """Per layer: R_Y(x_j) R_Z(x_j) on qubit q with j = q mod d, then a CZ chain (q, q+1).

    The circuit takes angles, not raw features: bind theta = s * pi * X_tilde (embed does this).
    With n_qubits < d only the first n_qubits features are used.
    """
    x = ParameterVector("x", min(n_qubits, d))
    qc = QuantumCircuit(n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(x[q % d], q)
            qc.rz(x[q % d], q)
        if entangle:  # CZs commute: even pairs then odd pairs gives the same state in two moments
            for q in [*range(0, n_qubits - 1, 2), *range(1, n_qubits - 1, 2)]:
                qc.cz(q, q + 1)
    return qc
