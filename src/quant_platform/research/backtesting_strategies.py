"""Research-only strategy backtests for the professional terminal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtesting.benchmarks import (
    buy_and_hold_target_weights,
    equal_weight_target_weights,
)
from quant_platform.backtesting.costs import TransactionCostConfig
from quant_platform.backtesting.engine import (
    BacktestConfig,
    BacktestResult,
    run_vectorized_backtest,
)
from quant_platform.research.returns import daily_simple_returns


class TerminalBacktestError(ValueError):
    """Raised when terminal backtest inputs are invalid."""


def moving_average_crossover_weights(
    prices: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """Build long-only weights where short moving average is above long average."""

    clean_prices = _as_price_frame(prices)
    if short_window < 1 or long_window <= short_window:
        raise TerminalBacktestError("long_window must be greater than short_window.")
    short_ma = clean_prices.rolling(short_window, min_periods=short_window).mean()
    long_ma = clean_prices.rolling(long_window, min_periods=long_window).mean()
    signal = short_ma > long_ma
    weights = pd.DataFrame(0.0, index=clean_prices.index, columns=clean_prices.columns)
    for timestamp, row in signal.iterrows():
        selected = row[row].index
        if len(selected) > 0:
            weights.loc[timestamp, selected] = 1.0 / len(selected)
    return weights


def momentum_skip_weights(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
    top_n: int = 1,
) -> pd.DataFrame:
    """Build long-only momentum weights using lookback returns with a recent skip window."""

    clean_prices = _as_price_frame(prices)
    if lookback_days <= skip_days or skip_days < 0:
        raise TerminalBacktestError("lookback_days must be greater than skip_days >= 0.")
    if top_n < 1 or top_n > clean_prices.shape[1]:
        raise TerminalBacktestError("top_n must be between 1 and the number of assets.")
    trailing = clean_prices.shift(skip_days) / clean_prices.shift(lookback_days) - 1.0
    weights = pd.DataFrame(0.0, index=clean_prices.index, columns=clean_prices.columns)
    for timestamp, row in trailing.iterrows():
        if row.isna().any():
            continue
        winners = row.sort_values(ascending=False).index[:top_n]
        weights.loc[timestamp, winners] = 1.0 / top_n
    return weights


def run_terminal_backtests(
    prices: pd.DataFrame,
    initial_capital: float = 10_000.0,
    short_window: int = 50,
    long_window: int = 200,
    momentum_lookback_days: int = 252,
    momentum_skip_days: int = 21,
    periods_per_year: int = 252,
) -> dict[str, BacktestResult]:
    """Run buy-hold, equal-weight, MA crossover, and momentum backtests."""

    clean_prices = _as_price_frame(prices)
    returns = daily_simple_returns(clean_prices)
    aligned_prices = clean_prices.reindex(returns.index)
    config = BacktestConfig(
        periods_per_year=periods_per_year,
        execution_lag=1,
        cost_config=TransactionCostConfig.from_bps(commission_bps=1.0, spread_bps=2.0),
        initial_capital=initial_capital,
        allow_short=False,
        max_leverage=1.0,
    )
    target_weights = {
        "buy_and_hold": buy_and_hold_target_weights(aligned_prices),
        "equal_weight_rebalanced": equal_weight_target_weights(returns),
        "moving_average_crossover": moving_average_crossover_weights(
            aligned_prices,
            short_window=short_window,
            long_window=long_window,
        ),
        "momentum_12_1": momentum_skip_weights(
            aligned_prices,
            lookback_days=momentum_lookback_days,
            skip_days=momentum_skip_days,
            top_n=1,
        ),
    }
    return {
        name: run_vectorized_backtest(returns, weights, config)
        for name, weights in target_weights.items()
    }


def serialize_backtest_result(result: BacktestResult) -> dict[str, Any]:
    """Serialize a BacktestResult for JSON reports and UI charts."""

    return {
        "metrics": dict(result.metrics),
        "metadata": dict(result.metadata),
        "equity_curve": _series_rows(result.equity_curve, "equity"),
        "drawdown_curve": _drawdown_rows(result.equity_curve),
        "net_returns": _series_rows(result.net_returns, "return"),
        "turnover": _series_rows(result.turnover, "turnover"),
    }


def _as_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise TerminalBacktestError("prices must be a non-empty DataFrame.")
    clean = prices.astype(float)
    if clean.columns.has_duplicates:
        raise TerminalBacktestError("price columns must be unique.")
    if not clean.index.is_monotonic_increasing:
        raise TerminalBacktestError("prices index must be sorted.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean <= 0).to_numpy().any():
        raise TerminalBacktestError("prices must be finite and strictly positive.")
    return clean


def _series_rows(series: pd.Series, value_name: str) -> list[dict[str, float | str]]:
    return [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), value_name: float(value)}
        for timestamp, value in series.items()
    ]


def _drawdown_rows(equity_curve: pd.Series) -> list[dict[str, float | str]]:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return _series_rows(drawdown, "drawdown")
