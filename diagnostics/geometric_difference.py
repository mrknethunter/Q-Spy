"""Geometric difference between classical and quantum Gram matrices.

This module consumes already computed kernel matrices.  It does not build
kernels or depend on any quantum, data, or machine-learning component.
"""

from __future__ import annotations

from collections.abc import Iterable
import warnings
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class GeometricDifferenceSweepResult(TypedDict):
    """JSON-safe, row-aligned output from :func:`geometric_difference_sweep`."""

    regularizations: list[float | None]
    values: list[float | None]
    valid: list[bool]
    errors: list[str | None]
    condition_numbers: list[float | None]

_SYMMETRY_RTOL = 1e-7
_SYMMETRY_ATOL = 1e-10
_PSD_RTOL = 1e-10
_PSD_ATOL = 1e-12
_TRACE_RTOL = 1e-7
_TRACE_ATOL = 1e-10

DEFAULT_REGULARIZATIONS: tuple[float, ...] = tuple(10.0**power for power in range(-10, -1))


def _as_kernel_matrix(matrix: ArrayLike, name: str) -> FloatArray:
    """Convert and validate one finite, symmetric PSD kernel matrix."""
    raw = np.asarray(matrix)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc

    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.allclose(
        result, result.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL
    ):
        raise ValueError(f"{name} must be symmetric.")

    # Symmetrization removes only differences already accepted by the explicit
    # symmetry tolerance; it is not used to repair structural asymmetry.
    result = (result + result.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(result)
    scale = float(np.max(np.abs(eigenvalues)))
    psd_tolerance = _PSD_ATOL + _PSD_RTOL * scale
    if float(eigenvalues[0]) < -psd_tolerance:
        raise ValueError(
            f"{name} is not positive semidefinite within tolerance "
            f"({_PSD_ATOL:g} + {_PSD_RTOL:g} * spectral scale)."
        )
    return result


def _warn_if_trace_not_normalized(K_C: FloatArray, K_Q: FloatArray) -> None:
    """Warn when the trace convention assumed by Huang et al. is not met."""
    n = K_C.shape[0]
    discrepancies = []
    for name, matrix in (("K_C", K_C), ("K_Q", K_Q)):
        trace = float(np.trace(matrix))
        if not np.isclose(trace, n, rtol=_TRACE_RTOL, atol=_TRACE_ATOL):
            discrepancies.append(f"Tr({name})={trace:.12g} (expected N={n})")
    if discrepancies:
        warnings.warn(
            "Huang et al.'s trace normalization assumption Tr(K_C) = "
            f"Tr(K_Q) = N is not met: {'; '.join(discrepancies)}. "
            "The matrices are used without automatic normalization.",
            RuntimeWarning,
            stacklevel=3,
        )


def _principal_psd_sqrt(matrix: FloatArray) -> FloatArray:
    """Return the principal PSD square root of an already validated matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def _geometric_difference_from_validated(
    classical: FloatArray, quantum_root: FloatArray, ridge: float
) -> float:
    """Compute ``g`` from validated inputs and a precomputed quantum root.

    The caller is responsible for validating both kernels, checking the trace
    convention, computing ``quantum_root``, and validating ``ridge``.
    """
    regularized = classical.copy()
    regularized.flat[:: regularized.shape[0] + 1] += ridge
    if not np.isfinite(regularized).all():
        raise np.linalg.LinAlgError("K_C + regularization * I is not finite.")
    classical_values = np.linalg.eigvalsh(regularized)
    if float(classical_values[0]) <= 0.0:
        raise ValueError(
            "K_C + regularization * I must be positive definite; "
            "use a positive regularization for singular K_C."
        )

    solved = np.linalg.solve(regularized, quantum_root)
    comparison = quantum_root @ solved
    # The theoretical product is symmetric PSD; symmetrize only this computed
    # product to remove solve-level roundoff, after all input structure checks.
    comparison = (comparison + comparison.T) * 0.5
    comparison_values = np.linalg.eigvalsh(comparison)
    comparison_scale = float(np.max(np.abs(comparison_values)))
    if float(comparison_values[0]) < -(_PSD_ATOL + _PSD_RTOL * comparison_scale):
        raise np.linalg.LinAlgError(
            "The computed comparison matrix is not PSD within numerical tolerance."
        )
    spectral_norm = max(abs(float(comparison_values[0])), abs(float(comparison_values[-1])))
    result = float(np.sqrt(max(0.0, spectral_norm)))
    if not np.isfinite(result):
        raise np.linalg.LinAlgError("The geometric difference is not finite.")
    return result


def geometric_difference(
    K_C: ArrayLike, K_Q: ArrayLike, regularization: float = 0.0
) -> float:
    r"""Compute the ridge-regularized geometric difference ``g(K_C || K_Q)``.

    The order of the inputs is significant.  This evaluates

    .. math::

       g_\lambda = \sqrt{\|K_Q^{1/2}(K_C + \lambda I)^{-1}
                                  K_Q^{1/2}\|_\infty}.

    The principal PSD square root of ``K_Q`` is constructed with ``eigh``.
    The linear system is solved directly; no explicit matrix inverse is
    formed.  Ridge regularization is applied only to ``K_C``.

    ``regularization`` is an absolute, non-negative ridge parameter in the
    units of the kernel entries.  At zero, ``K_C`` must be numerically
    invertible.  A non-normalized trace emits ``RuntimeWarning`` and the
    supplied matrices are left unchanged.

    Args:
        K_C: Classical ``N x N`` real symmetric PSD Gram matrix.
        K_Q: Quantum ``N x N`` real symmetric PSD Gram matrix.
        regularization: Absolute ridge value :math:`\lambda`, default 0.

    Returns:
        A finite non-negative ``float`` geometric difference.

    Raises:
        ValueError: If matrices are malformed, non-PSD, incompatible, or
            regularization is negative/non-finite; also if the regularized
            classical matrix is singular.
        numpy.linalg.LinAlgError: If an eigensystem or linear solve fails.
    """
    try:
        ridge = float(regularization)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("regularization must be a finite non-negative number.") from exc
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("regularization must be a finite non-negative number.")

    classical = _as_kernel_matrix(K_C, "K_C")
    quantum = _as_kernel_matrix(K_Q, "K_Q")
    if classical.shape != quantum.shape:
        raise ValueError("K_C and K_Q must have identical shapes.")
    _warn_if_trace_not_normalized(classical, quantum)

    # Tiny negative eigenvalues accepted by the PSD tolerance are clipped only
    # while constructing the principal PSD square root.
    quantum_root = _principal_psd_sqrt(quantum)
    return _geometric_difference_from_validated(classical, quantum_root, ridge)


def geometric_difference_sweep(
    K_C: ArrayLike,
    K_Q: ArrayLike,
    regularizations: Iterable[float] | None = None,
) -> GeometricDifferenceSweepResult:
    r"""Compute geometric difference across an explicit ridge sweep.

    By default, the absolute values ``1e-10`` through ``1e-2`` are used.
    This log-spaced proposal is not an automatically selected optimum.  Its
    useful range depends on the kernel eigenvalue scale; inspect the returned
    condition numbers and consider a scale-aware sweep when kernel scales
    differ substantially.

    Each requested value has an aligned row in the returned dictionary.
    A failed individual calculation is represented by ``valid=False``,
    ``value=None``, and a human-readable ``error``; malformed shared kernels
    raise ``ValueError`` before the sweep begins. Finite negative lambdas are
    retained as floats, while unconvertible or non-finite entries are stored
    as ``None``. All returned values are safe for strict JSON serialization.
    An empty sweep is rejected.

    Returns:
        A dictionary with JSON-friendly lists: ``regularizations``,
        ``values``, ``valid``, ``errors``, and ``condition_numbers``. The
        condition number is that of ``K_C + lambda I`` when calculable.

    Raises:
        ValueError: If the shared matrices are invalid or no values were
            supplied.
    """
    requested = DEFAULT_REGULARIZATIONS if regularizations is None else regularizations
    if isinstance(requested, (str, bytes)):
        raise ValueError("regularizations must be a non-empty sequence of numbers.")
    try:
        values = list(requested)
    except TypeError as exc:
        raise ValueError("regularizations must be a non-empty sequence of numbers.") from exc
    if not values:
        raise ValueError("regularizations must not be empty.")

    classical = _as_kernel_matrix(K_C, "K_C")
    quantum = _as_kernel_matrix(K_Q, "K_Q")
    if classical.shape != quantum.shape:
        raise ValueError("K_C and K_Q must have identical shapes.")
    _warn_if_trace_not_normalized(classical, quantum)

    # Validate once and reuse the same principal PSD root for every lambda.
    quantum_root = _principal_psd_sqrt(quantum)

    result: GeometricDifferenceSweepResult = {
        "regularizations": [],
        "values": [],
        "valid": [],
        "errors": [],
        "condition_numbers": [],
    }
    for raw_ridge in values:
        try:
            ridge = float(raw_ridge)
        except (TypeError, ValueError, OverflowError):
            ridge = None
        if ridge is not None and not np.isfinite(ridge):
            ridge = None
        result["regularizations"].append(ridge)
        if ridge is None or ridge < 0.0:
            result["values"].append(None)
            result["valid"].append(False)
            result["errors"].append(
                "regularization must be convertible to a finite non-negative number."
            )
            result["condition_numbers"].append(None)
            continue

        condition: float | None = None
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                regularized = classical.copy()
                regularized.flat[:: regularized.shape[0] + 1] += ridge
            if np.isfinite(regularized).all():
                eigs = np.linalg.eigvalsh(regularized)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    condition = (
                        float(eigs[-1] / eigs[0]) if eigs[0] > 0.0 else None
                    )
                if condition is not None and not np.isfinite(condition):
                    condition = None
            value = _geometric_difference_from_validated(
                classical, quantum_root, ridge
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            result["values"].append(None)
            result["valid"].append(False)
            result["errors"].append(str(exc))
            result["condition_numbers"].append(condition)
        else:
            result["values"].append(value)
            result["valid"].append(True)
            result["errors"].append(None)
            result["condition_numbers"].append(condition)
    return result


__all__ = [
    "DEFAULT_REGULARIZATIONS",
    "geometric_difference",
    "geometric_difference_sweep",
]
