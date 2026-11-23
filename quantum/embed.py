"""Quantum embedding contract (L-02). Implemented by Q-01..Q-05; see docs/contracts.md."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    """Return Bloch vectors of U_{s,L}(x)|0> for every row of X_tilde on the given tier."""
    raise NotImplementedError("Q-03 (L0), Q-04 (L1/L2), Q-05 (HW)")
