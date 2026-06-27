# Validation Splits

Financial time series must be split chronologically. Random splits are not allowed because they can put future market regimes, volatility states, liquidity conditions, or post-event information into the training set while evaluating on earlier observations.

## Implemented Objects

`src/quant_platform/validation/splits.py` implements:

- `TimeSplit`: boundary labels for train, validation, and test.
- `single_time_split(index, train_size, validation_size, test_size)`.
- `walk_forward_splits(index, train_size, validation_size, test_size, step_size)`.

## Rules

- Index values must be strictly chronological in increasing order.
- Duplicate timestamps are rejected.
- Train, validation, and test windows do not overlap.
- Each split respects:

```text
train_end < validation_start <= validation_end < test_start <= test_end
```

- Insufficient observations fail with a clear error.

## Leakage Control

Walk-forward validation approximates the operational sequence of research decisions: fit or calibrate on past data, choose on validation data, and evaluate once on later test data. This reduces leakage and discourages strategy selection by the best historical curve.
