"""Acceptance-focused tests for the D-03 feature transformation."""

import numpy as np
import pandas as pd
import pytest

from features.transforms import FEATURE_COLUMNS, make_features


def make_frame(rows: int, *, offset: float = 0.0) -> pd.DataFrame:
    """Build deterministic, positive synthetic Omicron-like measurements."""
    values = np.arange(1, rows + 1, dtype=np.float64)[:, None] + offset
    # Different scales per column ensure that the transform handles all seven
    # independently rather than relying on one shared numeric range.
    multipliers = np.arange(1, len(FEATURE_COLUMNS) + 1, dtype=np.float64)[None, :]
    frame = pd.DataFrame(values * multipliers, columns=FEATURE_COLUMNS)
    frame["id"] = [f"trigger-{index}" for index in range(rows)]
    frame["label"] = "synthetic"
    return frame


def test_output_has_contract_shape_dtype_and_bounds() -> None:
    """The output matches L-02's (N, 7), float64, [-1, 1] contract."""
    transformed, _ = make_features(make_frame(12))

    assert transformed.shape == (12, 7)
    assert transformed.dtype == np.float64
    assert np.isfinite(transformed).all()
    assert (transformed >= -1.0).all()
    assert (transformed <= 1.0).all()


def test_scaler_is_fitted_on_training_rows_and_reused_for_holdout() -> None:
    """Extreme holdout rows do not change the train-fitted scaler statistics."""
    training = make_frame(8)
    holdout = make_frame(3, offset=10_000.0)

    _, train_scaler = make_features(training)
    train_center = train_scaler.center_.copy()
    train_scale = train_scaler.scale_.copy()

    # Transforming holdout rows with the returned scaler must not refit it.
    heldout_features, reused_scaler = make_features(holdout, scaler=train_scaler)

    np.testing.assert_array_equal(reused_scaler.center_, train_center)
    np.testing.assert_array_equal(reused_scaler.scale_, train_scale)
    assert heldout_features.shape == (3, 7)

    # Independently fitting only the training frame gives the same statistics;
    # the very different holdout distribution has played no role in the fit.
    _, second_train_scaler = make_features(training.copy())
    np.testing.assert_allclose(train_scaler.center_, second_train_scaler.center_)
    np.testing.assert_allclose(train_scaler.scale_, second_train_scaler.scale_)

    # Fitting on the combined partitions would produce different statistics;
    # this guards against accidentally including holdout rows in the fit input.
    _, combined_scaler = make_features(pd.concat([training, holdout], ignore_index=True))
    assert not np.allclose(train_scaler.center_, combined_scaler.center_)
    assert not np.allclose(train_scaler.scale_, combined_scaler.scale_)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_rejects_values_that_cannot_be_log_transformed(bad_value: float) -> None:
    """Invalid measurements fail early with a useful error."""
    frame = make_frame(2)
    frame.loc[0, FEATURE_COLUMNS[0]] = bad_value

    with pytest.raises(ValueError):
        make_features(frame)


def test_rejects_missing_feature_columns() -> None:
    """Schema mismatches from upstream curation are reported explicitly."""
    frame = make_frame(2).drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing required feature columns"):
        make_features(frame)
