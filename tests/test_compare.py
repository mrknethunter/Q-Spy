"""Unit and synthetic acceptance tests for G-03 comparisons."""

import json

import numpy as np
import pytest

from diagnostics.compare import (
    bloch_frobenius_error,
    compare_runs,
    drift_report,
    kernel_alignment,
)


def _bloch(n_samples=3, n_qubits=2):
    return np.zeros((n_samples, n_qubits, 3), dtype=np.float32)


def _centered_test_kernels():
    # H = I - 11^T/3 and C has zero row sums and zero trace, hence H:C = 0.
    ones = np.ones((3, 3), dtype=np.float64)
    H = np.eye(3) - ones / 3.0
    C = np.array(
        [[1.0, 0.0, -1.0], [0.0, -1.0, 1.0], [-1.0, 1.0, 0.0]]
    )
    return ones + H, ones + C


def test_identical_bloch_arrays_have_zero_error():
    ids = ["a", "b", "c"]
    B = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 20

    result = bloch_frobenius_error(ids, B, ids, B.copy())

    assert result["sample_ids"] == ids
    assert result["rmse_per_qubit"] == [0.0, 0.0]
    assert result["rmse_global"] == 0.0
    assert result["frobenius_global"] == 0.0


def test_deterministic_perturbation_matches_analytic_metrics():
    ids = [1, 2, 3, 4]
    ideal = _bloch(4, 2)
    perturbed = ideal.copy()
    perturbed[:, 0, :] += np.float32(0.25)
    perturbed[:, 1, :] -= np.float32(0.5)

    result = bloch_frobenius_error(ids, ideal, ids, perturbed)

    assert result["rmse_per_qubit"] == pytest.approx([0.25, 0.5])
    assert result["rmse_global"] == pytest.approx(np.sqrt((0.25**2 + 0.5**2) / 2))
    assert result["frobenius_global"] == pytest.approx(
        np.sqrt(4 * 3 * (0.25**2 + 0.5**2))
    )


def test_noise_on_one_qubit_is_attributed_to_that_qubit():
    ids = ["a", "b", "c"]
    ideal = _bloch()
    noisy = ideal.copy()
    noisy[:, 1, :] = np.float32(0.4)

    result = bloch_frobenius_error(ids, ideal, ids, noisy)

    assert result["rmse_per_qubit"] == pytest.approx([0.0, 0.4])
    assert result["rmse_global"] == pytest.approx(0.4 / np.sqrt(2))


def test_bloch_comparison_reorders_by_id_explicitly():
    ids_a = ["x", "y", "z"]
    ids_b = ["z", "x", "y"]
    B_a = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 30
    B_b = B_a[[2, 0, 1]]

    result = bloch_frobenius_error(ids_a, B_a, ids_b, B_b)

    assert result["sample_ids"] == ids_a
    assert result["rmse_global"] == 0.0


def test_identical_kernel_alignment_is_one():
    K = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]])
    ids = ["a", "b", "c"]

    assert kernel_alignment(K, K.copy(), ids, ids) == pytest.approx(1.0)


def test_different_kernel_alignment_matches_centered_expected_value():
    K_a, K_b = _centered_test_kernels()
    ids = ["a", "b", "c"]

    assert kernel_alignment(K_a, K_b, ids, ids) == pytest.approx(0.0, abs=1e-14)


def test_kernel_alignment_reorders_both_kernel_axes_by_id():
    K = np.array([[1.0, 0.7, 0.2], [0.7, 1.0, 0.3], [0.2, 0.3, 1.0]])
    ids = ["a", "b", "c"]
    permutation = [2, 0, 1]

    assert kernel_alignment(
        K,
        K[np.ix_(permutation, permutation)],
        ids,
        [ids[i] for i in permutation],
    ) == pytest.approx(1.0)


def test_centered_zero_kernel_has_explicit_denominator_error():
    K = np.ones((3, 3))

    with pytest.raises(ValueError, match="zero norm"):
        kernel_alignment(K, K, ["a", "b", "c"], ["a", "b", "c"])


def test_zero_drift_report_uses_metadata_run_ids():
    ids = ["ref-1", "ref-2", "ref-3"]
    B = _bloch()
    start = (ids, B, {"run_id": "start-001", "backend": "fake"})
    end = (ids, B.copy(), {"run_id": "end-001", "backend": "fake"})

    report = drift_report(start, end)

    assert report["comparison_type"] == "drift"
    assert report["run_start_id"] == "start-001"
    assert report["run_end_id"] == "end-001"
    assert report["reference_ids"] == ids
    assert report["bloch_drift"]["rmse_global"] == 0.0
    assert report["kernel_alignment"] is None


def test_drift_report_matches_known_perturbation_and_kernel_alignment():
    ids = ["r1", "r2", "r3"]
    start_B = _bloch()
    end_B = start_B.copy()
    end_B[:, 0, :] += np.float32(0.2)
    K_a, K_b = _centered_test_kernels()
    start = (ids, start_B, {"run_id": "before"})
    end = (ids, end_B, {"run_id": "after", "calibration": {"snapshot": 4}})

    report = drift_report(
        start,
        end,
        kernel_start=(ids, K_a),
        kernel_end=(ids, K_b),
    )

    assert report["run_start_id"] == "before"
    assert report["run_end_id"] == "after"
    assert report["reference_ids"] == ids
    assert report["bloch_drift"]["rmse_per_qubit"] == pytest.approx([0.2, 0.0])
    assert report["bloch_drift"]["rmse_global"] == pytest.approx(0.2 / np.sqrt(2))
    assert report["bloch_drift"]["frobenius_global"] == pytest.approx(0.2 * 3)
    assert report["kernel_alignment"][
        "centered_normalized_frobenius_alignment"
    ] == pytest.approx(0.0, abs=1e-14)
    assert report["meta_end"]["calibration"] == {"snapshot": 4}


def test_ids_must_match_and_cannot_be_missing():
    B = _bloch()
    with pytest.raises(ValueError, match="same sample IDs"):
        bloch_frobenius_error(["a", "b", "c"], B, ["a", "b", "missing"], B)
    with pytest.raises(ValueError, match="one ID per sample"):
        bloch_frobenius_error(["a", "b"], B, ["a", "b", "c"], B)
    with pytest.raises(ValueError, match="missing sample ID"):
        bloch_frobenius_error(["a", None, "c"], B, ["a", "b", "c"], B)


def test_duplicate_ids_are_rejected():
    B = _bloch()
    with pytest.raises(ValueError, match="duplicate"):
        bloch_frobenius_error(["a", "a", "c"], B, ["a", "b", "c"], B)


def test_incompatible_bloch_shapes_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        bloch_frobenius_error(
            ["a", "b", "c"],
            np.zeros((3, 2, 2), dtype=np.float32),
            ["a", "b", "c"],
            _bloch(),
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_bloch_values_are_rejected(bad_value):
    B = _bloch()
    B[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        bloch_frobenius_error(["a", "b", "c"], B, ["a", "b", "c"], _bloch())


def test_incompatible_qubit_counts_are_rejected():
    with pytest.raises(ValueError, match="same number of qubits"):
        bloch_frobenius_error(
            ["a", "b", "c"], _bloch(3, 2), ["a", "b", "c"], _bloch(3, 3)
        )


def test_kernel_invalid_shapes_values_and_ids_are_rejected():
    ids = ["a", "b", "c"]
    K = np.eye(3)
    with pytest.raises(ValueError, match="square"):
        kernel_alignment(np.ones((2, 3)), K, ids, ids)
    with pytest.raises(ValueError, match="finite"):
        kernel_alignment(
            np.array(
                [
                    [1.0, np.nan, 0.0],
                    [np.nan, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            K,
            ids,
            ids,
        )
    with pytest.raises(ValueError, match="finite"):
        infinite = K.copy()
        infinite[0, 1] = np.inf
        infinite[1, 0] = np.inf
        kernel_alignment(infinite, K, ids, ids)
    with pytest.raises(ValueError, match="symmetric"):
        asymmetric = K.copy()
        asymmetric[0, 1] = 0.25
        kernel_alignment(asymmetric, K, ids, ids)
    with pytest.raises(ValueError, match="same sample IDs"):
        kernel_alignment(K, K, ids, ["a", "b", "x"])


def test_metadata_may_be_empty_or_incomplete_but_must_be_a_dictionary():
    ids = ["a", "b", "c"]
    B = _bloch()
    report = compare_runs(
        (ids, B, {}),
        (ids, B, {"backend": "emulator"}),
        tier_a="L0",
        tier_b="L2",
    )

    assert report["meta_a"] == {}
    assert report["meta_b"] == {"backend": "emulator"}
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="metadata dictionary"):
        compare_runs((ids, B, None), (ids, B, {}), tier_a="L0", tier_b="L2")


def test_drift_requires_identifiable_run_ids():
    ids = ["a", "b", "c"]
    B = _bloch()
    with pytest.raises(ValueError, match=r"meta\['run_id'\]"):
        drift_report((ids, B, {}), (ids, B, {}))
    assert drift_report(
        (ids, B, {}), (ids, B, {}), run_start_id="before", run_end_id="after"
    )["run_start_id"] == "before"


def test_synthetic_acceptance_reports_are_quantitative_and_serializable():
    """Use contract-shaped synthetic results with known injected noise."""
    ids = ["sample-0", "sample-1", "sample-2"]
    ideal_B = _bloch()
    hw_B = ideal_B.copy()
    hw_B[:, 1, :] += np.float32(0.3)
    K_a, K_b = _centered_test_kernels()
    ideal_run = (ids, ideal_B, {"run_id": "ideal-1", "tier": "L0"})
    hw_run = (
        ids,
        hw_B,
        {"run_id": "hardware-1", "tier": "HW", "noise_level": np.float32(0.3)},
    )

    tier_report = compare_runs(
        ideal_run,
        hw_run,
        tier_a="L0",
        tier_b="HW",
        kernel_a=(ids, K_a),
        kernel_b=(ids, K_b),
    )
    assert tier_report["tier_a"] == "L0"
    assert tier_report["tier_b"] == "HW"
    assert tier_report["bloch_vector_error"]["rmse_per_qubit"] == pytest.approx(
        [0.0, 0.3]
    )
    assert tier_report["bloch_vector_error"]["rmse_global"] == pytest.approx(
        0.3 / np.sqrt(2)
    )
    assert tier_report["bloch_vector_error"]["frobenius_global"] == pytest.approx(
        0.3 * 3
    )
    assert tier_report["kernel_comparison"][
        "centered_normalized_frobenius_alignment"
    ] == pytest.approx(0.0, abs=1e-14)
    assert tier_report["meta_b"]["noise_level"] == pytest.approx(0.3)

    reference_start = (ids, ideal_B, {"run_id": "campaign-start"})
    reference_end_B = ideal_B.copy()
    reference_end_B[:, 0, :] -= np.float32(0.2)
    reference_end = (ids, reference_end_B, {"run_id": "campaign-end"})
    drift = drift_report(
        reference_start,
        reference_end,
        kernel_start=(ids, K_a),
        kernel_end=(ids, K_a),
    )
    assert drift["reference_ids"] == ids
    assert drift["bloch_drift"]["rmse_per_qubit"] == pytest.approx([0.2, 0.0])
    assert drift["bloch_drift"]["rmse_global"] == pytest.approx(0.2 / np.sqrt(2))
    assert drift["kernel_alignment"][
        "centered_normalized_frobenius_alignment"
    ] == pytest.approx(1.0)
    json.dumps(tier_report, allow_nan=False)
    json.dumps(drift, allow_nan=False)
