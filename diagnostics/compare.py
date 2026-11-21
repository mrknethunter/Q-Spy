"""Compare Bloch-vector runs, projected kernels, and campaign drift.

This module consumes results that have already been loaded. In particular,
it does not import the Bloch store or construct a projected kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BlochArray = NDArray[np.float32]
BlochResult = tuple[ArrayLike, ArrayLike, Mapping[str, Any]]
KernelResult = tuple[ArrayLike, ArrayLike]


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Return JSON-compatible builtins, rejecting lossy/unknown values."""
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key.")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    raise ValueError(f"{path} contains a value that is not JSON serializable.")


def _as_ids(ids: ArrayLike, *, size: int, name: str) -> list[Any]:
    """Validate a one-dimensional sequence of unique JSON-compatible IDs."""
    if isinstance(ids, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional sequence of IDs.")
    if isinstance(ids, np.ndarray) and ids.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    try:
        values = list(ids)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence of IDs.") from exc

    if len(values) != size:
        raise ValueError(f"{name} must contain one ID per sample.")
    if not values:
        raise ValueError(f"{name} cannot be empty.")

    normalized: list[Any] = []
    seen: set[Any] = set()
    for index, sample_id in enumerate(values):
        if isinstance(sample_id, np.generic):
            sample_id = sample_id.item()
        if sample_id is None or (isinstance(sample_id, str) and not sample_id):
            raise ValueError(f"{name} contains a missing sample ID.")
        json_id = _json_safe(sample_id, path=f"{name}[{index}]")
        if isinstance(json_id, (dict, list)):
            raise ValueError(f"{name}[{index}] must be a scalar sample ID.")
        try:
            if json_id in seen:
                raise ValueError(f"{name} contains duplicate sample IDs.")
            seen.add(json_id)
        except TypeError as exc:
            raise ValueError(f"{name}[{index}] must be a hashable sample ID.") from exc
        normalized.append(json_id)
    return normalized


def _indices_in_reference_order(
    reference_ids: list[Any], other_ids: list[Any], *, context: str
) -> list[int]:
    """Return indices that order ``other_ids`` like ``reference_ids``."""
    try:
        other_index = {sample_id: index for index, sample_id in enumerate(other_ids)}
        if len(reference_ids) != len(other_ids) or set(reference_ids) != set(other_ids):
            raise ValueError(f"{context} must contain the same sample IDs.")
        return [other_index[sample_id] for sample_id in reference_ids]
    except TypeError as exc:
        raise ValueError(f"{context} IDs must be hashable.") from exc


def _as_bloch(B: ArrayLike, *, name: str) -> BlochArray:
    """Validate and preserve an L-02 float32 Bloch array."""
    try:
        raw = np.asarray(B)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if np.iscomplexobj(raw) or raw.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must contain real float32 values per L-02.")
    if raw.ndim != 3:
        raise ValueError(f"{name} must have shape (N, n_qubits, 3).")
    if raw.shape[0] == 0 or raw.shape[1] == 0 or raw.shape[2] != 3:
        raise ValueError(f"{name} must have non-empty shape (N, n_qubits, 3).")
    if not np.isfinite(raw).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if np.any(raw < -1.0) or np.any(raw > 1.0):
        raise ValueError(f"{name} values must be in [-1, 1] per L-02.")
    return raw


def _as_kernel(K: ArrayLike, *, name: str) -> FloatArray:
    """Validate a real, finite, symmetric Gram matrix."""
    try:
        raw = np.asarray(K)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric matrix.") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values.")
    try:
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{name} must be symmetric.")
    # Remove only the tiny asymmetry admitted by the numerical tolerance.
    return (matrix + matrix.T) * 0.5


def _as_meta(meta: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise ValueError(f"{name} must be a metadata dictionary per L-02.")
    converted = _json_safe(dict(meta), path=name)
    return converted


def _unpack_bloch_result(
    result: BlochResult, *, name: str
) -> tuple[list[Any], BlochArray, dict[str, Any]]:
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError(f"{name} must have the read_bloch() form (ids, B, meta).")
    ids_raw, B_raw, meta_raw = result
    B = _as_bloch(B_raw, name=f"{name}.B")
    ids = _as_ids(ids_raw, size=B.shape[0], name=f"{name}.ids")
    meta = _as_meta(meta_raw, name=f"{name}.meta")
    return ids, B, meta


def _unpack_kernel_result(
    result: KernelResult, *, name: str
) -> tuple[list[Any], FloatArray]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError(f"{name} must have the form (sample_ids, K).")
    ids_raw, K_raw = result
    K = _as_kernel(K_raw, name=f"{name}.K")
    ids = _as_ids(ids_raw, size=K.shape[0], name=f"{name}.ids")
    return ids, K


def _center_kernel(matrix: FloatArray) -> FloatArray:
    """Apply H K H without materializing the centering matrix H."""
    row_mean = np.mean(matrix, axis=1, keepdims=True)
    column_mean = np.mean(matrix, axis=0, keepdims=True)
    overall_mean = float(np.mean(matrix))
    return matrix - row_mean - column_mean + overall_mean


def bloch_frobenius_error(
    ids_a: ArrayLike,
    B_a: ArrayLike,
    ids_b: ArrayLike,
    B_b: ArrayLike,
) -> dict[str, Any]:
    """Compare two L-02 Bloch arrays after matching their sample IDs.

    ``B_a`` and ``B_b`` must be float32 arrays shaped ``(N, n_qubits, 3)``.
    If the ID sequences contain the same IDs in different orders, rows of
    ``B_b`` are explicitly reordered to the order in ``ids_a``.

    Returns a JSON-serializable dictionary containing the matched IDs, RMSE
    for each qubit, global RMSE, and the unnormalized global Frobenius norm.
    Raises ``ValueError`` for invalid IDs, shapes, values, or qubit counts.
    """
    array_a = _as_bloch(B_a, name="B_a")
    array_b = _as_bloch(B_b, name="B_b")
    sample_ids_a = _as_ids(ids_a, size=array_a.shape[0], name="ids_a")
    sample_ids_b = _as_ids(ids_b, size=array_b.shape[0], name="ids_b")
    if array_a.shape[1:] != array_b.shape[1:]:
        raise ValueError("Bloch arrays must have the same number of qubits and 3 axes.")
    indices_b = _indices_in_reference_order(
        sample_ids_a, sample_ids_b, context="ids_a and ids_b"
    )
    difference = array_a.astype(np.float64) - array_b[indices_b].astype(np.float64)
    squared = np.square(difference)
    rmse_per_qubit = np.sqrt(np.mean(squared, axis=(0, 2)))
    rmse_global = float(np.sqrt(np.mean(squared)))
    frobenius_global = float(np.linalg.norm(difference.ravel(), ord=2))
    return {
        "sample_ids": sample_ids_a,
        "rmse_per_qubit": [float(value) for value in rmse_per_qubit],
        "rmse_global": rmse_global,
        "frobenius_global": frobenius_global,
    }


def kernel_alignment(
    K_a: ArrayLike,
    K_b: ArrayLike,
    ids_a: ArrayLike,
    ids_b: ArrayLike,
) -> float:
    """Return centered normalized Frobenius alignment for two Gram matrices.

    The matrices must be square and symmetric, have finite values, and be
    indexed by unique, matching sample IDs. If their ID order differs, rows
    and columns of ``K_b`` are reordered explicitly to match ``ids_a``.
    This is a centered alignment convention selected by G-03; neither the
    task nor the project proposal prescribes a more detailed formula.

    Raises ``ValueError`` for incompatible shapes/IDs, invalid matrices, or
    a zero norm after centering. This function compares provided matrices;
    it never constructs a projected kernel.
    """
    matrix_a = _as_kernel(K_a, name="K_a")
    matrix_b = _as_kernel(K_b, name="K_b")
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("K_a and K_b must have the same shape.")
    sample_ids_a = _as_ids(ids_a, size=matrix_a.shape[0], name="ids_a")
    sample_ids_b = _as_ids(ids_b, size=matrix_b.shape[0], name="ids_b")
    indices_b = _indices_in_reference_order(
        sample_ids_a, sample_ids_b, context="ids_a and ids_b"
    )
    matrix_b = matrix_b[np.ix_(indices_b, indices_b)]

    centered_a = _center_kernel(matrix_a)
    centered_b = _center_kernel(matrix_b)
    norm_a = float(np.linalg.norm(centered_a, ord="fro"))
    norm_b = float(np.linalg.norm(centered_b, ord="fro"))
    denominator = norm_a * norm_b
    if denominator == 0.0:
        raise ValueError(
            "Kernel alignment is undefined: a centered matrix has zero norm."
        )
    numerator = float(np.sum(centered_a * centered_b))
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def _validate_tier(tier: str, *, name: str) -> str:
    if not isinstance(tier, str) or not tier.strip():
        raise ValueError(f"{name} must be a non-empty tier label.")
    return tier


def _kernel_comparison(
    kernel_a: KernelResult | None,
    kernel_b: KernelResult | None,
    *,
    ids_a: list[Any],
    ids_b: list[Any],
) -> dict[str, Any] | None:
    if kernel_a is None and kernel_b is None:
        return None
    if kernel_a is None or kernel_b is None:
        raise ValueError("Provide both kernels to compare kernel alignment.")
    kernel_ids_a, matrix_a = _unpack_kernel_result(kernel_a, name="kernel_a")
    kernel_ids_b, matrix_b = _unpack_kernel_result(kernel_b, name="kernel_b")
    _indices_in_reference_order(
        ids_a, kernel_ids_a, context="run A and kernel_a"
    )
    _indices_in_reference_order(
        ids_b, kernel_ids_b, context="run B and kernel_b"
    )
    alignment = kernel_alignment(
        matrix_a, matrix_b, kernel_ids_a, kernel_ids_b
    )
    return {
        "sample_ids": kernel_ids_a,
        "centered_normalized_frobenius_alignment": alignment,
    }


def compare_runs(
    run_a: BlochResult,
    run_b: BlochResult,
    *,
    tier_a: str,
    tier_b: str,
    kernel_a: KernelResult | None = None,
    kernel_b: KernelResult | None = None,
) -> dict[str, Any]:
    """Compare two tier results already loaded using the L-02 data contract.

    Args:
        run_a, run_b: ``(ids, B, meta)`` results matching ``read_bloch()``.
        tier_a, tier_b: Explicit labels such as ``"L0"`` and ``"HW"``.
        kernel_a, kernel_b: Optional ``(sample_ids, K)`` pairs. Supply both
            or neither. Matrices must already be computed by the caller.

    Returns:
        A JSON-serializable report with tier labels, matched Bloch IDs and
        errors, optional kernel alignment, and the metadata supplied by both
        runs. Different ID ordering is aligned explicitly by sample ID.
    """
    tier_a = _validate_tier(tier_a, name="tier_a")
    tier_b = _validate_tier(tier_b, name="tier_b")
    ids_a, B_a, meta_a = _unpack_bloch_result(run_a, name="run_a")
    ids_b, B_b, meta_b = _unpack_bloch_result(run_b, name="run_b")
    bloch_error = bloch_frobenius_error(ids_a, B_a, ids_b, B_b)
    kernel_result = _kernel_comparison(
        kernel_a, kernel_b, ids_a=ids_a, ids_b=ids_b
    )
    return {
        "comparison_type": "tier",
        "tier_a": tier_a,
        "tier_b": tier_b,
        "bloch_vector_error": bloch_error,
        "kernel_comparison": kernel_result,
        "meta_a": meta_a,
        "meta_b": meta_b,
    }


def _resolve_run_id(run_id: Any, meta: Mapping[str, Any], *, name: str) -> Any:
    if run_id is None:
        if "run_id" not in meta:
            raise ValueError(
                f"{name} must be supplied or available as meta['run_id']."
            )
        run_id = meta["run_id"]
    resolved = _json_safe(run_id, path=name)
    if resolved is None or resolved == "":
        raise ValueError(f"{name} must identify a run.")
    if isinstance(resolved, (dict, list)):
        raise ValueError(f"{name} must be a scalar run identifier.")
    return resolved


def drift_report(
    run_start: BlochResult,
    run_end: BlochResult,
    *,
    run_start_id: Any = None,
    run_end_id: Any = None,
    kernel_start: KernelResult | None = None,
    kernel_end: KernelResult | None = None,
) -> dict[str, Any]:
    """Report Bloch-vector and optional kernel drift between reference runs.

    Args:
        run_start, run_end: Start/end ``(ids, B, meta)`` read results.
        run_start_id, run_end_id: Explicit run identifiers. When omitted,
            the function uses the corresponding ``meta['run_id']`` value;
            it raises if neither source identifies a run.
        kernel_start, kernel_end: Optional ``(sample_ids, K)`` pairs of
            already-computed kernels. Supply both or neither.

    Returns:
        A JSON-serializable drift report naming both runs, the matched
        reference IDs, per-qubit/global Bloch errors, optional centered
        kernel alignment, and both metadata dictionaries. No drift threshold
        or qualitative acceptability judgment is applied.
    """
    ids_start, B_start, meta_start = _unpack_bloch_result(
        run_start, name="run_start"
    )
    ids_end, B_end, meta_end = _unpack_bloch_result(run_end, name="run_end")
    start_id = _resolve_run_id(run_start_id, meta_start, name="run_start_id")
    end_id = _resolve_run_id(run_end_id, meta_end, name="run_end_id")
    bloch_drift = bloch_frobenius_error(
        ids_start, B_start, ids_end, B_end
    )
    kernel_result = _kernel_comparison(
        kernel_start, kernel_end, ids_a=ids_start, ids_b=ids_end
    )
    return {
        "comparison_type": "drift",
        "run_start_id": start_id,
        "run_end_id": end_id,
        "reference_ids": bloch_drift["sample_ids"],
        "bloch_drift": bloch_drift,
        "kernel_alignment": kernel_result,
        "meta_start": meta_start,
        "meta_end": meta_end,
    }


__all__ = [
    "bloch_frobenius_error",
    "kernel_alignment",
    "compare_runs",
    "drift_report",
]
