"""Portfolio analytics for the professional quant terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
)
from quant_platform.features.returns import cumulative_returns
from quant_platform.portfolio.baselines import equal_weight_weights, inverse_volatility_weights
from quant_platform.research.returns import annual_rate_to_periodic, daily_simple_returns
from quant_platform.risk.drawdown import max_drawdown


class PortfolioAnalyticsError(ValueError):
    """Raised when portfolio analytics inputs are invalid."""


def compute_portfolio_analytics(
    prices: pd.DataFrame,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, object]:
    """Compute covariance, correlation, and baseline portfolio summaries."""

    clean_prices = _as_price_frame(prices)
    returns = daily_simple_returns(clean_prices)
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    equal_weights = equal_weight_weights(tuple(str(column) for column in clean_prices.columns))
    inverse_vol_weights = inverse_volatility_weights(volatility)
    periodic_rf = annual_rate_to_periodic(risk_free_rate_annual, periods_per_year)
    return {
        "asset_ids": [str(column) for column in clean_prices.columns],
        "observations": int(len(returns)),
        "annualized_mean_returns": _series_to_dict(returns.mean() * periods_per_year),
        "annualized_volatility": _series_to_dict(volatility),
        "correlation_matrix": _matrix_rows(returns.corr()),
        "annualized_covariance_matrix": _matrix_rows(returns.cov() * periods_per_year),
        "equal_weight": summarize_weighted_portfolio(
            returns,
            equal_weights,
            periodic_rf,
            periods_per_year,
        ),
        "inverse_volatility": summarize_weighted_portfolio(
            returns,
            inverse_vol_weights,
            periodic_rf,
            periods_per_year,
        ),
    }


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | dict[str, float]) -> pd.Series:
    """Compute simple portfolio returns from aligned asset returns and weights."""

    clean_returns = _as_return_frame(returns)
    clean_weights = _as_weight_series(weights, clean_returns.columns)
    return clean_returns.mul(clean_weights, axis=1).sum(axis=1)


def summarize_weighted_portfolio(
    returns: pd.DataFrame,
    weights: pd.Series | dict[str, float],
    periodic_rf: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, object]:
    """Summarize one long-only portfolio weight vector."""

    clean_returns = _as_return_frame(returns)
    clean_weights = _as_weight_series(weights, clean_returns.columns)
    port_returns = portfolio_returns(clean_returns, clean_weights)
    equity_curve = cumulative_returns(port_returns, initial_value=1.0)
    return {
        "weights": _series_to_dict(clean_weights),
        "annualized_return": annualized_return(port_returns, periods_per_year),
        "annualized_volatility": annualized_volatility(port_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(port_returns, periodic_rf, periods_per_year),
        "sortino_ratio": sortino_ratio(port_returns, periodic_rf, periods_per_year),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "final_equity": float(equity_curve.iloc[-1]),
        "return_series": _return_rows(port_returns),
    }


def _as_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise PortfolioAnalyticsError("prices must be a non-empty DataFrame.")
    clean = prices.astype(float)
    if clean.columns.has_duplicates:
        raise PortfolioAnalyticsError("prices columns must be unique.")
    if not clean.index.is_monotonic_increasing:
        raise PortfolioAnalyticsError("prices index must be sorted.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean <= 0).to_numpy().any():
        raise PortfolioAnalyticsError("prices must be finite and strictly positive.")
    return clean


def _as_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise PortfolioAnalyticsError("returns must be a non-empty DataFrame.")
    clean = returns.astype(float)
    if clean.columns.has_duplicates:
        raise PortfolioAnalyticsError("returns columns must be unique.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean < -1).to_numpy().any():
        raise PortfolioAnalyticsError("returns must be finite and above -100%.")
    return clean


def _as_weight_series(weights: pd.Series | dict[str, float], columns: pd.Index) -> pd.Series:
    clean = (
        weights.astype(float) if isinstance(weights, pd.Series) else pd.Series(weights, dtype=float)
    )
    clean = clean.reindex(columns)
    if clean.isna().any():
        raise PortfolioAnalyticsError("weights must align to return columns.")
    if (clean < 0).any():
        raise PortfolioAnalyticsError("weights must be long-only.")
    total = float(clean.sum())
    if total <= 0 or not np.isfinite(total):
        raise PortfolioAnalyticsError("weights must sum to a positive finite value.")
    return clean / total


def _series_to_dict(series: pd.Series) -> dict[str, float]:
    return {str(key): float(value) for key, value in series.items()}


def _matrix_rows(matrix: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for row_label, values in matrix.iterrows():
        row: dict[str, float | str] = {"asset_id": str(row_label)}
        row.update({str(column): float(value) for column, value in values.items()})
        rows.append(row)
    return rows


def _return_rows(returns: pd.Series) -> list[dict[str, float | str]]:
    return [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "return": float(value)}
        for timestamp, value in returns.items()
    ]
