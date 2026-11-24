# D-03 Feature Transforms

`make_features(df, scaler=None)` selects the seven Omicron columns in proposal
section 4.2 order (`snr`, `peak_frequency`, `central_frequency`, `bandwidth`,
`duration`, `q_value`, `amplitude`) and returns `(X_tilde, scaler)`. Metadata and
additional columns are ignored. The canonical column names remain subject to
confirmation against the D-02 curated-data schema.

The processing order is `log10` → `RobustScaler` → clipping. For feature `j`,

```text
z_j = (log10(x_j) - median_j) / IQR_j
X_tilde_j = clip(z_j, -1, 1)
```

## Output contract

For every accepted non-empty input, the function returns a NumPy matrix with
shape `(N, 7)`, dtype `float64`, finite values, and every value in the closed
interval `[-1, 1]`. After scaling, each value is clipped with
`np.clip(z, -1.0, 1.0)`: values below/above the limits saturate at `-1`/`1`,
while interior values and values exactly on either limit are preserved. The
tests assert these numeric results. `RobustScaler` uses unit scale for a
zero-IQR feature, so a constant training feature maps to zero.

Fit on training rows only, then reuse the returned scaler for validation/test:

```python
X_train, scaler = make_features(train_df)
X_validation, _ = make_features(validation_df, scaler=scaler)
X_test, _ = make_features(test_df, scaler=scaler)
```

Passing no scaler fits on every row in the supplied frame; the caller owns the
split boundary. Missing/non-numeric, non-finite, zero or negative features,
empty input, and incompatible or unfitted scalers raise informative errors.
A supplied scaler must use `quantile_range=(25.0, 75.0)`,
`with_centering=True`, `with_scaling=True`, and `unit_variance=False`, in
addition to matching the feature count and names/order. Rows are not dropped.
The adaptive `tanh` compression is an experimental alternative that is not
implemented; it may be evaluated separately by ablation.

Run D-03 tests with `python -m pytest -q tests/test_transforms.py`; run the full
available suite with `python -m pytest -q`.
