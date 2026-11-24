"""Leakage-safe transforms for the seven Omicron input features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Proposal section 4.2 order; metadata columns are intentionally excluded.
FEATURE_COLUMNS: tuple[str, ...] = (
    "snr",
    "peak_frequency",
    "central_freq",
    "bandwidth",
    "duration",
    "q_value",
    "amplitude",
)


def make_features(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
) -> tuple[np.ndarray, RobustScaler]:
    """Apply log10, robust scaling and clipping to the seven Omicron features.

    If ``scaler`` is ``None``, statistics are fitted on ``df``; callers must
    therefore pass training rows only. A fitted scaler can be reused on
    validation and test frames without updating its statistics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"df is missing required feature columns: {missing}")
    duplicated = [column for column in FEATURE_COLUMNS if list(df.columns).count(column) > 1]
    if duplicated:
        raise ValueError(f"df has duplicate required feature columns: {duplicated}")
    if df.empty:
        raise ValueError("df must contain at least one row")

    try:
        raw_features = df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature values must be numeric and convertible to float64") from exc

    if not np.isfinite(raw_features).all():
        raise ValueError("feature values must all be finite")
    if (raw_features <= 0).any():
        raise ValueError("feature values must be strictly positive for log10")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logged_features = np.log10(raw_features)
    if not np.isfinite(logged_features).all():
        raise ValueError("log10 produced non-finite feature values")
    logged_frame = pd.DataFrame(logged_features, columns=FEATURE_COLUMNS, index=df.index)

    if scaler is None:
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        scaled_features = scaler.fit_transform(logged_frame)
    else:
        if not isinstance(scaler, RobustScaler):
            raise TypeError("scaler must be a fitted sklearn RobustScaler")
        expected_config = {
            "quantile_range": (25.0, 75.0),
            "with_centering": True,
            "with_scaling": True,
            "unit_variance": False,
        }
        incompatible_config = {
            name: getattr(scaler, name, None)
            for name, expected in expected_config.items()
            if getattr(scaler, name, None) != expected
        }
        if incompatible_config:
            raise ValueError(
                "scaler has incompatible configuration; expected "
                "quantile_range=(25.0, 75.0), with_centering=True, "
                "with_scaling=True, unit_variance=False"
            )
        if not hasattr(scaler, "center_") or not hasattr(scaler, "scale_"):
            raise ValueError("scaler must be fitted before it is passed to make_features")
        if getattr(scaler, "n_features_in_", None) != len(FEATURE_COLUMNS):
            raise ValueError("scaler is incompatible: it must be fitted on exactly seven features")
        scaler_columns = getattr(scaler, "feature_names_in_", None)
        if scaler_columns is not None and tuple(scaler_columns) != FEATURE_COLUMNS:
            raise ValueError("scaler is incompatible with the required feature names and order")
        scaled_features = scaler.transform(logged_frame)

    scaled_features = np.asarray(scaled_features, dtype=np.float64)
    if not np.isfinite(scaled_features).all():
        raise ValueError("robust scaling produced non-finite feature values")

    # RobustScaler uses scale_=1 for a zero-IQR feature, leaving its centered
    # values finite and well-defined. Clip is the specified bounded mapping.
    X_tilde = np.clip(scaled_features, -1.0, 1.0).astype(np.float64, copy=False)
    return X_tilde, scaler
