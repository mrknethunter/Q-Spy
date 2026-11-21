"""Acceptance tests for the D-03 feature transformation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from features.transforms import FEATURE_COLUMNS, make_features


def make_frame(rows: int, *, offset: float = 0.0) -> pd.DataFrame:
    values = np.arange(1, rows + 1, dtype=np.float64)[:, None] + offset
    multipliers = np.arange(1, len(FEATURE_COLUMNS) + 1, dtype=np.float64)[None, :]
    frame = pd.DataFrame(values * multipliers, columns=FEATURE_COLUMNS)
    frame["id"] = [f"trigger-{index}" for index in range(rows)]
    frame["label"] = "synthetic"
    return frame


def make_log_frame(log_values: list[float]) -> pd.DataFrame:
    raw = np.power(10.0, np.asarray(log_values, dtype=np.float64))[:, None]
    values = np.repeat(raw, len(FEATURE_COLUMNS), axis=1)
    return pd.DataFrame(values, columns=FEATURE_COLUMNS)


def test_output_shape_dtype_bounds_and_row_order() -> None:
    frame = make_frame(12)
    transformed, _ = make_features(frame)

    assert transformed.shape == (12, 7)
    assert transformed.dtype == np.float64
    assert np.isfinite(transformed).all()
    assert np.all((-1.0 <= transformed) & (transformed <= 1.0))
    # Row order is retained; the matrix follows FEATURE_COLUMNS, not metadata.
    assert transformed[0, 0] < transformed[-1, 0]


def test_preserves_feature_and_observation_order() -> None:
    frame = make_frame(8)
    expected, scaler = make_features(frame)
    row_order = [5, 1, 7, 0, 3, 2, 6, 4]
    reordered = frame.iloc[row_order].loc[
        :, [*reversed(FEATURE_COLUMNS), "id", "label"]
    ]

    actual, _ = make_features(reordered, scaler=scaler)
    np.testing.assert_array_equal(actual, expected[row_order])


def test_log_robust_scaling_and_clipping_formula() -> None:
    frame = make_frame(5)
    output, scaler = make_features(frame)
    logged = np.log10(frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    expected_scaled = (logged - scaler.center_) / scaler.scale_

    np.testing.assert_allclose(output, np.clip(expected_scaled, -1.0, 1.0))
    assert np.any(np.abs(expected_scaled) > 1.0)


def _scaler_with_known_unit_iqr() -> RobustScaler:
    _, scaler = make_features(make_log_frame([-2, -1, 0, 1, 2]))
    np.testing.assert_allclose(scaler.center_, np.zeros(7), atol=1e-15)
    np.testing.assert_allclose(scaler.scale_, np.full(7, 2.0), atol=1e-15)
    return scaler


def test_clipping_saturates_below_lower_bound() -> None:
    output, _ = make_features(make_log_frame([-6]), scaler=_scaler_with_known_unit_iqr())
    np.testing.assert_array_equal(output, np.full((1, 7), -1.0))


def test_clipping_saturates_above_upper_bound() -> None:
    output, _ = make_features(make_log_frame([6]), scaler=_scaler_with_known_unit_iqr())
    np.testing.assert_array_equal(output, np.full((1, 7), 1.0))


def test_clipping_preserves_values_inside_interval() -> None:
    output, _ = make_features(make_log_frame([-0.5, 0, 0.5]), scaler=_scaler_with_known_unit_iqr())
    expected = np.repeat(np.array([[-0.25], [0.0], [0.25]]), len(FEATURE_COLUMNS), axis=1)
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-15)


def test_clipping_preserves_values_exactly_on_limits() -> None:
    output, _ = make_features(make_log_frame([-2, 2]), scaler=_scaler_with_known_unit_iqr())
    np.testing.assert_allclose(output[0], np.full(7, -1.0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(output[1], np.full(7, 1.0), rtol=0, atol=1e-15)


def test_training_statistics_are_independent_of_validation_and_transform() -> None:
    training = make_frame(8)
    _, scaler = make_features(training)
    train_center, train_scale = scaler.center_.copy(), scaler.scale_.copy()

    validation_a = make_frame(2, offset=10_000.0)
    validation_b = make_frame(2, offset=1_000_000.0)
    make_features(validation_a, scaler=scaler)
    make_features(validation_b, scaler=scaler)

    np.testing.assert_array_equal(scaler.center_, train_center)
    np.testing.assert_array_equal(scaler.scale_, train_scale)
    _, repeated_scaler = make_features(training.copy())
    np.testing.assert_array_equal(repeated_scaler.center_, train_center)
    np.testing.assert_array_equal(repeated_scaler.scale_, train_scale)

    _, contaminated_scaler = make_features(pd.concat([training, validation_a], ignore_index=True))
    assert not np.allclose(contaminated_scaler.center_, train_center)


def test_constant_and_zero_iqr_features_remain_finite() -> None:
    frame = make_frame(4)
    frame[FEATURE_COLUMNS[0]] = 5.0
    output, scaler = make_features(frame)

    assert scaler.scale_[0] == 1.0
    np.testing.assert_array_equal(output[:, 0], np.zeros(4, dtype=np.float64))
    assert np.isfinite(output).all()


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_rejects_values_invalid_for_logarithm(bad_value: float) -> None:
    frame = make_frame(2)
    frame.loc[0, FEATURE_COLUMNS[0]] = bad_value
    with pytest.raises(ValueError):
        make_features(frame)


def test_rejects_non_numeric_features() -> None:
    frame = make_frame(2)
    frame[FEATURE_COLUMNS[0]] = frame[FEATURE_COLUMNS[0]].astype(object)
    frame.loc[0, FEATURE_COLUMNS[0]] = "not numeric"
    with pytest.raises(ValueError, match="numeric"):
        make_features(frame)


def test_rejects_missing_feature_columns_but_allows_additional_columns() -> None:
    frame = make_frame(2)
    make_features(frame)  # id and label are intentionally ignored.
    with pytest.raises(ValueError, match="missing required feature columns"):
        make_features(frame.drop(columns=[FEATURE_COLUMNS[0]]))


def test_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        make_features(make_frame(0))


def test_rejects_duplicate_required_columns() -> None:
    frame = make_frame(2)
    duplicated = pd.concat([frame, frame[[FEATURE_COLUMNS[0]]]], axis=1)
    with pytest.raises(ValueError, match="duplicate required feature columns"):
        make_features(duplicated)


def test_rejects_incompatible_or_unfitted_scaler() -> None:
    frame = make_frame(3)
    incompatible = RobustScaler().fit(pd.DataFrame(np.ones((3, 7)), columns=reversed(FEATURE_COLUMNS)))
    with pytest.raises(ValueError, match="incompatible"):
        make_features(frame, scaler=incompatible)

    with pytest.raises(ValueError, match="fitted"):
        make_features(frame, scaler=RobustScaler())


def test_rejects_scaler_with_wrong_feature_count() -> None:
    frame = make_frame(3)
    incompatible = RobustScaler().fit(np.ones((3, 6)))
    with pytest.raises(ValueError, match="exactly seven"):
        make_features(frame, scaler=incompatible)


@pytest.mark.parametrize(
    "config",
    [
        {"quantile_range": (10.0, 90.0)},
        {"with_centering": False},
        {"with_scaling": False},
        {"unit_variance": True},
    ],
    ids=["quantile-range", "centering", "scaling", "unit-variance"],
)
def test_rejects_scalers_with_incompatible_configuration(config: dict[str, object]) -> None:
    frame = make_frame(3)
    scaler = RobustScaler(**config).fit(frame.loc[:, FEATURE_COLUMNS])

    with pytest.raises(ValueError, match="incompatible configuration"):
        make_features(frame, scaler=scaler)
