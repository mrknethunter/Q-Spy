# D-03 Feature Transforms

This component prepares the seven Omicron measurements used by Q-Spy. It
implements the interface defined for D-03:

```python
make_features(df, scaler=None) -> (X_tilde, scaler)
```

## What it does

`make_features` selects the seven columns listed in `FEATURE_COLUMNS`, applies
`log10`, fits or reuses a scikit-learn `RobustScaler`, and applies `tanh` to
bound the result to `[-1, 1]`. It returns a `float64` NumPy array with shape
`(N, 7)` and the scaler used.

The current feature column names are canonical placeholders in snake_case.
Update `FEATURE_COLUMNS` in `features/transforms.py` if the D-02 curated-data
schema adopts different names.

## Avoiding data leakage

Fit the scaler only on training rows. Reuse that fitted scaler for validation
and test data:

```python
from features.transforms import make_features

X_train, scaler = make_features(train_df)
X_validation, _ = make_features(validation_df, scaler=scaler)
X_test, _ = make_features(test_df, scaler=scaler)
```

When called without a scaler, the function fits on every row in the supplied
DataFrame. The caller must therefore pass only training rows in that call.

## Requirements and tests

The implementation uses NumPy, pandas, and scikit-learn. Pytest is needed to
run its tests. From the repository root, install those packages in your Python
environment and run:

```bash
python -m pytest -q tests/test_transforms.py
```

## Implementation note

The task requires values in `[-1, 1]` but does not prescribe the squashing
formula. This implementation uses `tanh` after robust scaling; this is a
documented design choice that can be changed if the team agrees on another
mapping. All seven measurements must currently be finite and positive because
the proposal specifies a base-10 logarithm for each one.
