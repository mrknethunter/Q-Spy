"""Mathematical and input-contract tests for G-02."""

import json

import numpy as np
import pytest

from diagnostics.geometric_difference import (
    DEFAULT_REGULARIZATIONS,
    geometric_difference,
    geometric_difference_sweep,
)


def test_identical_positive_definite_kernels_have_unit_difference():
    kernel = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert geometric_difference(kernel, kernel) == pytest.approx(1.0)


def test_diagonal_matrices_match_analytic_unregularized_formula():
    classical = np.diag([0.4, 1.6])
    quantum = np.diag([1.6, 0.4])
    expected = np.sqrt(max(1.6 / 0.4, 0.4 / 1.6))
    assert geometric_difference(classical, quantum) == pytest.approx(expected)


def test_result_is_nonnegative():
    assert geometric_difference(np.eye(2), np.diag([0.0, 2.0])) >= 0.0


def test_geometric_difference_is_asymmetric():
    classical = np.diag([1.5, 0.5])
    quantum = np.eye(2)
    forward = geometric_difference(classical, quantum)
    reverse = geometric_difference(quantum, classical)
    assert forward == pytest.approx(np.sqrt(2.0))
    assert reverse == pytest.approx(np.sqrt(1.5))
    assert forward != pytest.approx(reverse)


def test_regularized_diagonal_formula():
    classical = np.diag([0.5, 1.5])
    quantum = np.diag([1.6, 0.4])
    ridge = 0.5
    expected = np.sqrt(max(1.6 / 1.0, 0.4 / 2.0))
    assert geometric_difference(classical, quantum, ridge) == pytest.approx(expected)


def test_identical_kernel_with_ridge_is_not_forced_to_one():
    kernel = np.diag([0.4, 1.6])
    ridge = 0.2
    expected = np.sqrt(max(0.4 / 0.6, 1.6 / 1.8))
    assert geometric_difference(kernel, kernel, ridge) == pytest.approx(expected)
    assert expected < 1.0


def test_singular_classical_kernel_requires_positive_ridge():
    classical = np.diag([2.0, 0.0])
    quantum = np.eye(2)
    with pytest.raises(ValueError, match="positive definite"):
        geometric_difference(classical, quantum)
    assert np.isfinite(geometric_difference(classical, quantum, 1e-6))


def test_nearly_singular_kernel_has_finite_regularized_sweep():
    classical = np.diag([2.0, 2e-14])
    quantum = np.eye(2)
    result = geometric_difference_sweep(classical, quantum, [0.0, 1e-10, 1e-6])
    assert result["valid"] == [True, True, True]
    assert all(np.isfinite(value) for value in result["values"])
    assert result["condition_numbers"][2] < result["condition_numbers"][1]


def test_sweep_keeps_each_lambda_and_reports_invalid_calculations():
    classical = np.diag([2.0, 0.0])
    result = geometric_difference_sweep(classical, np.eye(2), [0.0, 0.1, -1.0])
    assert result["regularizations"] == [0.0, 0.1, -1.0]
    assert result["valid"] == [False, True, False]
    assert result["values"][0] is None
    assert result["values"][1] is not None
    assert result["errors"][0]
    assert result["errors"][2]


def test_default_sweep_is_log_spaced_proposal():
    assert DEFAULT_REGULARIZATIONS == tuple(10.0**power for power in range(-10, -1))
    result = geometric_difference_sweep(np.eye(2), np.eye(2))
    assert result["regularizations"] == list(DEFAULT_REGULARIZATIONS)
    assert all(result["valid"])


def test_non_normalized_traces_warn_without_changing_inputs():
    classical = np.diag([2.0, 2.0])
    quantum = np.diag([1.0, 1.0])
    classical_before, quantum_before = classical.copy(), quantum.copy()
    with pytest.warns(RuntimeWarning, match="trace normalization assumption"):
        geometric_difference(classical, quantum, 0.1)
    np.testing.assert_array_equal(classical, classical_before)
    np.testing.assert_array_equal(quantum, quantum_before)


@pytest.mark.parametrize(
    "classical,quantum",
    [
        (np.ones((2, 3)), np.eye(2)),
        (np.eye(2), np.eye(3)),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), np.eye(2)),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), np.eye(2)),
        (np.array([[1.0, 0.2], [0.1, 1.0]]), np.eye(2)),
        (np.diag([1.0, -0.1]), np.eye(2)),
    ],
)
def test_invalid_kernel_inputs_raise_value_error(classical, quantum):
    with pytest.raises(ValueError):
        geometric_difference(classical, quantum)


@pytest.mark.parametrize("ridge", [-1.0, np.nan, np.inf])
def test_invalid_regularization_is_rejected(ridge):
    with pytest.raises(ValueError, match="regularization"):
        geometric_difference(np.eye(2), np.eye(2), ridge)


def test_empty_sweep_is_rejected():
    with pytest.raises(ValueError, match="not be empty"):
        geometric_difference_sweep(np.eye(2), np.eye(2), [])


def test_results_are_reproducible():
    classical = np.array([[1.0, 0.25], [0.25, 1.0]])
    quantum = np.array([[1.0, -0.1], [-0.1, 1.0]])
    one = geometric_difference_sweep(classical, quantum, [0.0, 1e-6, 1e-3])
    two = geometric_difference_sweep(classical, quantum, [0.0, 1e-6, 1e-3])
    assert one == two


def test_sweep_is_strict_json_serializable_for_invalid_regularizations():
    result = geometric_difference_sweep(
        np.eye(2), np.eye(2), [0.1, -0.1, "not-a-number", np.nan, np.inf, -np.inf]
    )
    encoded = json.dumps(result, allow_nan=False)
    assert encoded
    assert result["regularizations"] == [0.1, -0.1, None, None, None, None]
    assert result["valid"] == [True, False, False, False, False, False]
    assert all(value is None for value in result["values"][1:])
    assert all(error is not None for error in result["errors"][1:])
    assert all(number is None for number in result["condition_numbers"][1:])


def test_geometric_difference_is_monotone_in_ridge_for_noncommuting_kernels():
    classical = np.array([[1.3, 0.4], [0.4, 0.7]])
    quantum = np.array([[0.6, 0.25], [0.25, 1.4]])
    assert not np.allclose(classical @ quantum, quantum @ classical)

    lambdas = [0.0, 1e-6, 1e-3, 0.1, 1.0, 10.0]
    values = [geometric_difference(classical, quantum, lam) for lam in lambdas]
    assert all(left >= right - 1e-12 for left, right in zip(values, values[1:]))


def test_noncommuting_result_matches_independent_svd_reference():
    classical = np.array([[1.3, 0.4], [0.4, 0.7]])
    quantum = np.array([[0.6, 0.25], [0.25, 1.4]])
    ridge = 0.15

    # Independent reference: construct the square root, then use the largest
    # singular value rather than the implementation's symmetric eigenspectrum.
    eigenvalues, eigenvectors = np.linalg.eigh(quantum)
    quantum_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    comparison = quantum_root @ np.linalg.solve(
        classical + ridge * np.eye(classical.shape[0]), quantum_root
    )
    expected = np.sqrt(np.linalg.svd(comparison, compute_uv=False)[0])
    assert geometric_difference(classical, quantum, ridge) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_sweep_emits_only_one_trace_normalization_warning():
    with pytest.warns(
        RuntimeWarning, match="trace normalization assumption"
    ) as trace_warnings:
        result = geometric_difference_sweep(
            np.diag([2.0, 2.0]), np.eye(2), [0.0, 0.1, 1.0, 10.0]
        )

    assert len(trace_warnings) == 1
    assert result["valid"] == [True, True, True, True]
