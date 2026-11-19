"""Leakage-safe transforms for the seven Omicron input features.

The feature names below are deliberately kept in one editable constant because
the final column names depend on the D-02 curated-data implementation. The
processing order follows the project proposal: log transform, robust scaling
and a bounded mapping suitable for the quantum feature map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# These canonical snake_case names describe the seven Omicron quantities in
# the proposal. If D-02 uses different names, update this tuple in one place.
FEATURE_COLUMNS: tuple[str, ...] = (
    "snr",
    "peak_frequency",
    "central_frequency",
    "bandwidth",
    "duration",
    "q_value",
    "amplitude",
)


def make_features(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
) -> tuple[np.ndarray, RobustScaler]:
    """Log-transform and scale the seven Omicron features without data leakage.

    Parameters
    ----------
    df:
        DataFrame containing one row per trigger and the columns listed in
        :data:`FEATURE_COLUMNS`. When ``scaler`` is ``None``, this DataFrame
        is used to fit the scaler and therefore must contain training rows
        only. For validation or test data, pass the scaler fitted on training
        rows.
    scaler:
        A previously fitted :class:`~sklearn.preprocessing.RobustScaler`.
        If omitted, a new scaler is fitted on the log-transformed values in
        ``df``. Reusing the returned scaler prevents validation/test data from
        affecting the training statistics.

    Returns
    -------
    X_tilde, scaler:
        ``X_tilde`` is a ``float64`` NumPy array with shape ``(N, 7)`` and
        values in ``[-1, 1]``. The second value is the fitted or reused scaler.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas DataFrame or ``scaler`` is not a fitted
        ``RobustScaler``.
    ValueError
        If required feature columns are missing, the frame has no rows, or
        feature values are non-finite or non-positive (``log10`` is undefined
        for those values).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Reject missing columns explicitly so a renamed upstream schema produces
    # an actionable error instead of silently selecting the wrong variables.
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required feature columns: {missing}")

    if df.empty:
        raise ValueError("df must contain at least one row")

    # Select only model inputs; metadata such as id, label, detector and GPS
    # must never enter the feature matrix.
    raw_features = df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    # All seven quantities are positive physical measurements in this task.
    # Validate before taking log10 to avoid propagating inf/nan into training.
    if not np.isfinite(raw_features).all():
        raise ValueError("feature values must all be finite")
    if (raw_features <= 0).any():
        raise ValueError("feature values must be positive for the log10 transform")

    logged_features = np.log10(raw_features)

    # Fitting on df occurs only when no scaler is supplied. The caller must
    # pass training rows for this branch; inference rows reuse train statistics.
    if scaler is None:
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        scaled_features = scaler.fit_transform(logged_features)
    else:
        if not isinstance(scaler, RobustScaler):
            raise TypeError("scaler must be a fitted sklearn RobustScaler")
        if not hasattr(scaler, "center_") or not hasattr(scaler, "scale_"):
            raise TypeError("scaler must be fitted before it is passed to make_features")
        scaled_features = scaler.transform(logged_features)

    # Robust scaling is unbounded. tanh is the chosen smooth squashing function:
    # it bounds every finite value and limits the influence of extreme outliers.
    X_tilde = np.tanh(scaled_features).astype(np.float64, copy=False)

    return X_tilde, scaler
