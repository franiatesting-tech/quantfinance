"""Benchmark target-weight generators for vectorized backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


class BenchmarkError(ValueError):
    """Raised when benchmark inputs are invalid."""


def _as_float_frame(frame: pd.DataFrame, name: str, require_positive: bool = False) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if frame.empty or frame.shape[1] == 0:
        raise BenchmarkError(f"{name} must not be empty.")
    if frame.columns.has_duplicates:
        raise BenchmarkError(f"{name} columns must be unique asset ids.")
    if frame.index.has_duplicates:
        raise BenchmarkError(f"{name} index must not contain duplicates.")
    if not frame.index.is_monotonic_increasing:
        raise BenchmarkError(f"{name} index must be sorted in increasing order.")
    try:
        clean_frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{name} must be numeric.") from exc
    if not np.isfinite(clean_frame.to_numpy(dtype=float)).all():
        raise BenchmarkError(f"{name} must not contain NaN or infinite values.")
    if require_positive and (clean_frame <= 0).to_numpy().any():
        raise BenchmarkError(f"{name} must contain strictly positive prices.")
    return clean_frame


def _empty_weights(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=frame.index, columns=frame.columns)


def cash_target_weights(prices_or_returns: pd.DataFrame) -> pd.DataFrame:
    """Return an all-cash target represented by zero asset weights."""

    clean_frame = _as_float_frame(prices_or_returns, "prices_or_returns")
    return _empty_weights(clean_frame)


def equal_weight_target_weights(prices_or_returns: pd.DataFrame) -> pd.DataFrame:
    """Return constant 1/N target weights dated at each input row."""

    clean_frame = _as_float_frame(prices_or_returns, "prices_or_returns")
    weights = _empty_weights(clean_frame)
    weights.loc[:, :] = 1.0 / clean_frame.shape[1]
    return weights


def buy_and_hold_target_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Return target weights implied by equal-weight fixed shares through time."""

    clean_prices = _as_float_frame(prices, "prices", require_positive=True)
    initial_weights = pd.Series(1.0 / clean_prices.shape[1], index=clean_prices.columns)
    shares = initial_weights / clean_prices.iloc[0]
    asset_values = clean_prices.mul(shares, axis=1)
    portfolio_value = asset_values.sum(axis=1)
    return asset_values.div(portfolio_value, axis=0)


def inverse_volatility_target_weights(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Return rolling inverse-volatility weights using data through each row `t`."""

    if lookback < 2:
        raise BenchmarkError("lookback must be >= 2 for inverse volatility.")
    clean_returns = _as_float_frame(returns, "returns")
    rolling_volatility = clean_returns.rolling(window=lookback, min_periods=lookback).std(ddof=1)
    weights = _empty_weights(clean_returns)
    for timestamp, row in rolling_volatility.iterrows():
        if row.isna().any() or (row <= 0).any():
            continue
        inverse = 1.0 / row
        weights.loc[timestamp, :] = inverse / inverse.sum()
    return weights


def momentum_target_weights(
    prices: pd.DataFrame,
    lookback: int,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return equal-weight top trailing-return targets dated at each row `t`."""

    if lookback < 1:
        raise BenchmarkError("lookback must be >= 1.")
    clean_prices = _as_float_frame(prices, "prices", require_positive=True)
    asset_count = clean_prices.shape[1]
    selected_count = asset_count if top_n is None else int(top_n)
    if selected_count < 1 or selected_count > asset_count:
        raise BenchmarkError("top_n must be between 1 and the number of assets.")

    trailing_returns = clean_prices / clean_prices.shift(lookback) - 1.0
    weights = _empty_weights(clean_prices)
    for timestamp, row in trailing_returns.iterrows():
        if row.isna().any():
            continue
        winners = row.sort_values(ascending=False).index[:selected_count]
        weights.loc[timestamp, winners] = 1.0 / selected_count
    return weights
