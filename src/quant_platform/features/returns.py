"""Return calculations and temporal alignment utilities."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

PandasObject: TypeAlias = pd.Series | pd.DataFrame


class ReturnCalculationError(ValueError):
    """Raised when return inputs violate mathematical conventions."""


def _as_float_pandas(data: PandasObject, name: str) -> PandasObject:
    if not isinstance(data, pd.Series | pd.DataFrame):
        raise TypeError(f"{name} must be a pandas Series or DataFrame.")
    if data.empty:
        raise ReturnCalculationError(f"{name} must not be empty.")
    try:
        values = data.astype(float)
    except (TypeError, ValueError) as exc:
        raise ReturnCalculationError(f"{name} must be numeric.") from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ReturnCalculationError(f"{name} must not contain NaN or infinite values.")
    return values


def _validate_prices_positive(prices: PandasObject) -> PandasObject:
    values = _as_float_pandas(prices, "prices")
    if (values <= 0).to_numpy().any():
        raise ReturnCalculationError("prices must be strictly positive.")
    return values


def simple_returns(prices: PandasObject) -> PandasObject:
    """Compute simple returns `R_t = P_t / P_{t-1} - 1`.

    The first observation is `NaN` because no previous price exists. Prices must be
    finite and strictly positive under the platform price convention.
    """

    clean_prices = _validate_prices_positive(prices)
    return clean_prices.pct_change(fill_method=None)


def log_returns(prices: PandasObject) -> PandasObject:
    """Compute log returns `r_t = log(P_t) - log(P_{t-1})`.

    The first observation is `NaN`. Zero or negative prices raise an error because the
    logarithm is undefined for the price convention used here.
    """

    clean_prices = _validate_prices_positive(prices)
    return np.log(clean_prices).diff()


def cumulative_returns(returns: PandasObject, initial_value: float = 1.0) -> PandasObject:
    """Build an equity curve `V_t = V_0 * prod_{s<=t}(1 + R_s)`.

    Inputs are periodic simple returns. The function preserves index and columns.
    """

    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ReturnCalculationError("initial_value must be finite and > 0.")
    clean_returns = _as_float_pandas(returns, "returns")
    if (clean_returns < -1).to_numpy().any():
        raise ReturnCalculationError("simple returns below -100% are invalid.")
    return initial_value * (1.0 + clean_returns).cumprod()


def excess_returns(
    returns: PandasObject,
    risk_free_rate: float | pd.Series,
) -> PandasObject:
    """Compute excess returns `r^e_t = r_t - r_{f,t}`.

    A scalar `risk_free_rate` is treated as a periodic rate already expressed at the
    same frequency as `returns`. A Series risk-free rate is aligned by timestamp.
    """

    clean_returns = _as_float_pandas(returns, "returns")
    if isinstance(risk_free_rate, int | float):
        if not np.isfinite(risk_free_rate):
            raise ReturnCalculationError("risk_free_rate must be finite.")
        return clean_returns - float(risk_free_rate)

    clean_rf = _as_float_pandas(risk_free_rate, "risk_free_rate")
    if not isinstance(clean_rf, pd.Series):
        raise TypeError("risk_free_rate must be a scalar or pandas Series.")
    aligned_rf = clean_rf.reindex(clean_returns.index)
    if aligned_rf.isna().any():
        raise ReturnCalculationError("risk_free_rate is missing observations after alignment.")
    if isinstance(clean_returns, pd.DataFrame):
        return clean_returns.sub(aligned_rf, axis=0)
    return clean_returns - aligned_rf


def align_features_and_forward_returns(
    features: PandasObject,
    returns: PandasObject,
    horizon: int = 1,
) -> tuple[PandasObject, PandasObject]:
    """Align `feature_t` with forward return `return_{t+horizon}`.

    The function shifts returns backward with `returns.shift(-horizon)` so a row labeled
    `t` contains only features observed at `t` and the target realized after `t`. Rows
    without a future target are dropped.
    """

    if horizon < 1:
        raise ReturnCalculationError("horizon must be >= 1.")
    clean_features = _as_float_pandas(features, "features")
    clean_returns = _as_float_pandas(returns, "returns")
    forward_returns = clean_returns.shift(-horizon)
    aligned_features, aligned_returns = clean_features.align(forward_returns, join="inner", axis=0)

    feature_valid = ~aligned_features.isna()
    return_valid = ~aligned_returns.isna()
    if isinstance(feature_valid, pd.DataFrame):
        feature_valid = feature_valid.all(axis=1)
    if isinstance(return_valid, pd.DataFrame):
        return_valid = return_valid.all(axis=1)
    valid = feature_valid & return_valid
    return aligned_features.loc[valid], aligned_returns.loc[valid]
