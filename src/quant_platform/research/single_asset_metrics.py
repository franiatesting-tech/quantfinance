"""Single-stock analytics used by the professional quant terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)
from quant_platform.features.returns import cumulative_returns
from quant_platform.research.returns import annual_rate_to_periodic, daily_simple_returns
from quant_platform.risk.drawdown import drawdown, max_drawdown


class SingleAssetMetricError(ValueError):
    """Raised when single-asset analytics inputs are invalid."""


def compute_single_asset_metrics(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, dict[str, object]]:
    """Compute professional single-stock metrics for each price column."""

    clean_prices = _as_price_frame(prices)
    clean_benchmark = _as_price_series(benchmark_prices, "benchmark_prices")
    returns = daily_simple_returns(clean_prices)
    benchmark_returns = daily_simple_returns(clean_benchmark)
    periodic_rf = annual_rate_to_periodic(risk_free_rate_annual, periods_per_year)
    rows: dict[str, dict[str, object]] = {}
    for symbol in clean_prices.columns:
        asset_returns, market_returns = returns[str(symbol)].align(benchmark_returns, join="inner")
        asset_returns = asset_returns.dropna()
        market_returns = market_returns.reindex(asset_returns.index).dropna()
        asset_returns = asset_returns.reindex(market_returns.index)
        asset_prices = clean_prices[str(symbol)].reindex(asset_returns.index)
        equity_curve = cumulative_returns(asset_returns, initial_value=1.0)
        total_return = float(equity_curve.iloc[-1] - 1.0)
        end_price = float(asset_prices.iloc[-1])
        implied_start_price = end_price / (1.0 + total_return)
        asset_beta = beta(asset_returns, market_returns)
        asset_ann_return = annualized_return(asset_returns, periods_per_year)
        benchmark_ann_return = annualized_return(market_returns, periods_per_year)
        rows[str(symbol)] = {
            "symbol": str(symbol),
            "observations": int(len(asset_returns)),
            "start_timestamp": pd.Timestamp(asset_returns.index[0]).isoformat(),
            "end_timestamp": pd.Timestamp(asset_returns.index[-1]).isoformat(),
            "start_price": float(implied_start_price),
            "end_price": end_price,
            "total_return": total_return,
            "annualized_return": float(asset_ann_return),
            "annualized_volatility": annualized_volatility(asset_returns, periods_per_year),
            "sharpe_ratio": sharpe_ratio(asset_returns, periodic_rf, periods_per_year),
            "sortino_ratio": sortino_ratio(asset_returns, periodic_rf, periods_per_year),
            "hit_rate": hit_rate(asset_returns),
            "max_drawdown": float(max_drawdown(equity_curve)),
            "beta_to_benchmark": asset_beta,
            "treynor_ratio": treynor_ratio(asset_ann_return, risk_free_rate_annual, asset_beta),
            "jensen_alpha": jensen_alpha(
                asset_ann_return,
                benchmark_ann_return,
                risk_free_rate_annual,
                asset_beta,
            ),
            "series": _series_rows(asset_prices, asset_returns, equity_curve),
        }
    return rows


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute CAPM beta as covariance with benchmark divided by benchmark variance."""

    aligned_asset, aligned_benchmark = asset_returns.align(benchmark_returns, join="inner")
    if len(aligned_asset) < 2:
        return float("nan")
    variance = float(aligned_benchmark.var(ddof=1))
    if variance == 0 or not np.isfinite(variance):
        return float("nan")
    covariance = float(aligned_asset.cov(aligned_benchmark))
    return covariance / variance


def treynor_ratio(annual_return: float, annual_rf: float, beta_value: float) -> float:
    """Compute Treynor ratio using annual return and CAPM beta."""

    if beta_value == 0 or not np.isfinite(beta_value):
        return float("nan")
    return float((annual_return - annual_rf) / beta_value)


def jensen_alpha(
    annual_return: float,
    benchmark_annual_return: float,
    annual_rf: float,
    beta_value: float,
) -> float:
    """Compute annualized Jensen alpha against a benchmark proxy."""

    if not np.isfinite(beta_value):
        return float("nan")
    capm_return = annual_rf + beta_value * (benchmark_annual_return - annual_rf)
    return float(annual_return - capm_return)


def _series_rows(
    prices: pd.Series,
    returns: pd.Series,
    equity_curve: pd.Series,
) -> list[dict[str, float | str]]:
    drawdown_curve = drawdown(equity_curve)
    rows = []
    for timestamp in returns.index:
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "price": float(prices.loc[timestamp]),
                "return": float(returns.loc[timestamp]),
                "equity": float(equity_curve.loc[timestamp]),
                "drawdown": float(drawdown_curve.loc[timestamp]),
            }
        )
    return rows


def _as_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise SingleAssetMetricError("prices must be a non-empty DataFrame.")
    clean = prices.astype(float)
    if clean.columns.has_duplicates:
        raise SingleAssetMetricError("prices columns must be unique.")
    if not clean.index.is_monotonic_increasing:
        raise SingleAssetMetricError("prices index must be sorted.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean <= 0).to_numpy().any():
        raise SingleAssetMetricError("prices must be finite and strictly positive.")
    return clean


def _as_price_series(prices: pd.Series, name: str) -> pd.Series:
    if not isinstance(prices, pd.Series) or prices.empty:
        raise SingleAssetMetricError(f"{name} must be a non-empty Series.")
    clean = prices.astype(float)
    if not clean.index.is_monotonic_increasing:
        raise SingleAssetMetricError(f"{name} index must be sorted.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean <= 0).any():
        raise SingleAssetMetricError(f"{name} must be finite and strictly positive.")
    return clean
