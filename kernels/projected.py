"""Projected quantum kernel contract (L-02). Implemented by M-01; see docs/contracts.md."""
from __future__ import annotations

import numpy as np


def projected_kernel(B1: np.ndarray, B2: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma / n * sum_q ||B1[i, q] - B2[j, q]||^2), shape (N1, N2)."""
    raise NotImplementedError("M-01")
