"""Simple long-only portfolio baselines.

These baselines are intentionally non-optimized. They provide auditable reference
portfolios that future models and optimizers must beat out of sample.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class PortfolioBaselineError(ValueError):
    """Raised when baseline portfolio inputs are invalid."""


def _asset_index(asset_ids: Sequence[str]) -> pd.Index:
    index = pd.Index(asset_ids, dtype="object")
    if index.empty:
        raise PortfolioBaselineError("asset_ids must not be empty.")
    if index.has_duplicates:
        raise PortfolioBaselineError("asset_ids must be unique.")
    return index


def _as_float_series(values: pd.Series | Mapping[str, float], name: str) -> pd.Series:
    try:
        series = (
            values.astype(float)
            if isinstance(values, pd.Series)
            else pd.Series(values, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise PortfolioBaselineError(f"{name} must be numeric.") from exc
    if series.empty:
        raise PortfolioBaselineError(f"{name} must not be empty.")
    if series.index.has_duplicates:
        raise PortfolioBaselineError(f"{name} index must be unique.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise PortfolioBaselineError(f"{name} must not contain NaN or infinite values.")
    return series


def _as_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty or prices.shape[1] == 0:
        raise PortfolioBaselineError("prices must not be empty.")
    if prices.columns.has_duplicates:
        raise PortfolioBaselineError("price columns must be unique asset ids.")
    try:
        clean_prices = prices.astype(float)
    except (TypeError, ValueError) as exc:
        raise PortfolioBaselineError("prices must be numeric.") from exc
    if not np.isfinite(clean_prices.to_numpy(dtype=float)).all():
        raise PortfolioBaselineError("prices must not contain NaN or infinite values.")
    if (clean_prices <= 0).to_numpy().any():
        raise PortfolioBaselineError("prices must be strictly positive.")
    return clean_prices


def _normalized_long_weights(
    weights: pd.Series | Mapping[str, float] | None,
    columns: pd.Index,
) -> pd.Series:
    if weights is None:
        return equal_weight_weights(tuple(str(column) for column in columns))
    clean_weights = _as_float_series(weights, "weights").reindex(columns)
    if clean_weights.isna().any():
        raise PortfolioBaselineError("weights are missing one or more price columns.")
    if (clean_weights < 0).any():
        raise PortfolioBaselineError("weights must be long-only and non-negative.")
    total = float(clean_weights.sum())
    if total <= 0:
        raise PortfolioBaselineError("weights must sum to a positive value.")
    return clean_weights / total


def equal_weight_weights(asset_ids: Sequence[str]) -> pd.Series:
    """Return a 1/N long-only weight vector."""

    index = _asset_index(asset_ids)
    return pd.Series(1.0 / len(index), index=index, dtype=float)


def inverse_volatility_weights(volatility: pd.Series | Mapping[str, float]) -> pd.Series:
    """Return weights proportional to inverse asset volatility."""

    clean_volatility = _as_float_series(volatility, "volatility")
    if (clean_volatility <= 0).any():
        raise PortfolioBaselineError("volatility values must be strictly positive.")
    inverse = 1.0 / clean_volatility
    return inverse / inverse.sum()


def buy_and_hold_returns(
    prices: pd.DataFrame,
    weights: pd.Series | Mapping[str, float] | None = None,
) -> pd.Series:
    """Compute simple returns of a fixed-share buy-and-hold portfolio."""

    clean_prices = _as_price_frame(prices)
    clean_weights = _normalized_long_weights(weights, clean_prices.columns)
    shares = clean_weights / clean_prices.iloc[0]
    portfolio_value = clean_prices.mul(shares, axis=1).sum(axis=1)
    return portfolio_value.pct_change(fill_method=None)


def momentum_weights(prices: pd.DataFrame, lookback: int, top_n: int | None = None) -> pd.DataFrame:
    """Build equal-weight long-only momentum weights from trailing returns.

    Weights labeled at `t` use only prices at `t` and `t-lookback`. A backtest
    must still shift these weights before applying returns for `t+1`.
    """

    if lookback < 1:
        raise PortfolioBaselineError("lookback must be >= 1.")
    clean_prices = _as_price_frame(prices)
    asset_count = clean_prices.shape[1]
    selected_count = asset_count if top_n is None else int(top_n)
    if selected_count < 1 or selected_count > asset_count:
        raise PortfolioBaselineError("top_n must be between 1 and the number of assets.")

    trailing_returns = clean_prices / clean_prices.shift(lookback) - 1.0
    weights = pd.DataFrame(0.0, index=clean_prices.index, columns=clean_prices.columns)
    for timestamp, row in trailing_returns.iterrows():
        if row.isna().any():
            continue
        winners = row.sort_values(ascending=False).index[:selected_count]
        weights.loc[timestamp, winners] = 1.0 / selected_count
    return weights
