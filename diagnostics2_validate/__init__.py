"""Diagnostiche per le matrici kernel del progetto Q-Spy."""

from .alignment import kernel_target_alignment
from .concentration import (
    effective_rank,
    kernel_spectrum,
    off_diagonal_variance,
)

__all__ = [
    "off_diagonal_variance",
    "kernel_spectrum",
    "effective_rank",
    "kernel_target_alignment",
]
