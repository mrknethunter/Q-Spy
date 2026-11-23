"""Classical kernel contract (L-02). Implemented by M-01; see docs/contracts.md."""
from __future__ import annotations

import numpy as np


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma * ||X1[i] - X2[j]||^2), shape (N1, N2)."""
    raise NotImplementedError("M-01")
