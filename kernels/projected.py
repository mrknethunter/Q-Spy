from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def projected_kernel(B1: np.ndarray, B2: np.ndarray, gamma: float) -> np.ndarray:
    # an RBF kernel on flattened Bloch vectors, averaged over qubits (gamma / n)
    return rbf_kernel(B1.reshape(len(B1), -1), B2.reshape(len(B2), -1), gamma=gamma / B1.shape[1])
