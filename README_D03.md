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

The returned matrix has shape `(N, 7)`, dtype `float64`, and values in
`[-1, 1]`. `RobustScaler` uses its documented unit scale for a zero-IQR feature,
so a constant training feature maps to zero and remains well-defined.

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

Focused tests: `python3 -m pytest -q tests/test_transforms.py`. The available
system Python does not include `pytest` or `scikit-learn`; the full suite was
run successfully with those packages installed into a temporary directory
under `/tmp` (19 passed). No dependency manifest is present in the repository.
