"""Minimum vectorized backtesting engine with explicit execution lag."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtesting.costs import (
    TransactionCostConfig,
    transaction_costs_from_turnover,
)
from quant_platform.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)
from quant_platform.features.returns import cumulative_returns
from quant_platform.risk.drawdown import max_drawdown


class BacktestEngineError(ValueError):
    """Raised when a vectorized backtest input is invalid."""


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for vectorized portfolio accounting."""

    periods_per_year: int | float
    execution_lag: int = 1
    cost_config: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    initial_capital: float = 1.0
    allow_short: bool = False
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.periods_per_year) or self.periods_per_year <= 0:
            raise BacktestEngineError("periods_per_year must be finite and > 0.")
        if self.execution_lag < 1:
            raise BacktestEngineError("execution_lag must be >= 1 to avoid look-ahead bias.")
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise BacktestEngineError("initial_capital must be finite and > 0.")
        if not np.isfinite(self.max_leverage) or self.max_leverage <= 0:
            raise BacktestEngineError("max_leverage must be finite and > 0.")


@dataclass(frozen=True)
class BacktestResult:
    """Vectorized backtest outputs and audit metadata."""

    gross_returns: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    applied_weights: pd.DataFrame
    target_weights: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any]


def _as_float_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if frame.empty or frame.shape[1] == 0:
        raise BacktestEngineError(f"{name} must not be empty.")
    if frame.columns.has_duplicates:
        raise BacktestEngineError(f"{name} columns must be unique asset ids.")
    if frame.index.has_duplicates:
        raise BacktestEngineError(f"{name} index must not contain duplicates.")
    if not frame.index.is_monotonic_increasing:
        raise BacktestEngineError(f"{name} index must be sorted in increasing order.")
    try:
        clean_frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise BacktestEngineError(f"{name} must be numeric.") from exc
    if not np.isfinite(clean_frame.to_numpy(dtype=float)).all():
        raise BacktestEngineError(f"{name} must not contain NaN or infinite values.")
    return clean_frame


def _validate_weights(weights: pd.DataFrame, config: BacktestConfig) -> None:
    if not config.allow_short and (weights < 0).to_numpy().any():
        raise BacktestEngineError("target_weights contain short weights but allow_short=False.")
    leverage = weights.abs().sum(axis=1)
    if (leverage > config.max_leverage + 1e-12).any():
        raise BacktestEngineError("target_weights exceed max_leverage.")


def _build_metrics(
    net_returns: pd.Series,
    equity_curve: pd.Series,
    turnover: pd.Series,
    transaction_costs: pd.Series,
    periods_per_year: int | float,
) -> dict[str, float]:
    return {
        "annualized_return": annualized_return(net_returns, periods_per_year),
        "annualized_volatility": annualized_volatility(net_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(net_returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(net_returns, periods_per_year=periods_per_year),
        "calmar_ratio": calmar_ratio(net_returns, periods_per_year=periods_per_year),
        "hit_rate": hit_rate(net_returns),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "total_turnover": float(turnover.sum()),
        "average_turnover": float(turnover.mean()),
        "total_transaction_cost": float(transaction_costs.sum()),
        "final_equity": float(equity_curve.iloc[-1]),
    }


def run_vectorized_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Run vectorized portfolio accounting from target weights and asset returns.

    `target_weights` are decisions made with information available through `t`.
    Applied weights are shifted by `execution_lag`, so weights decided at `t` are
    first eligible for returns at `t+1` when `execution_lag=1`.
    """

    clean_returns = _as_float_frame(returns, "returns")
    clean_targets = _as_float_frame(target_weights, "target_weights")
    if list(clean_returns.columns) != list(clean_targets.columns):
        raise BacktestEngineError("returns and target_weights must have identical columns.")
    missing_target_dates = clean_returns.index.difference(clean_targets.index)
    if len(missing_target_dates) > 0:
        raise BacktestEngineError("target_weights must contain every return timestamp.")
    if (clean_returns < -1.0).to_numpy().any():
        raise BacktestEngineError("returns cannot contain simple returns below -100%.")
    _validate_weights(clean_targets, config)

    applied_full = clean_targets.shift(config.execution_lag).fillna(0.0)
    applied_weights = applied_full.reindex(clean_returns.index)
    target_aligned = clean_targets.reindex(clean_returns.index)
    if applied_weights.isna().to_numpy().any() or target_aligned.isna().to_numpy().any():
        raise BacktestEngineError("weights are missing observations after alignment.")

    gross_returns = clean_returns.mul(applied_weights, axis=0).sum(axis=1)
    previous_weights = pd.Series(0.0, index=applied_weights.columns)
    previous_applied_weights = applied_weights.shift(1).fillna(previous_weights)
    turnover = (applied_weights - previous_applied_weights).abs().sum(axis=1)
    transaction_costs = transaction_costs_from_turnover(turnover, config.cost_config)
    net_returns = gross_returns - transaction_costs - config.cost_config.funding_rate
    if (net_returns < -1.0).any():
        raise BacktestEngineError("net_returns contain a simple return below -100% after costs.")

    equity_curve = cumulative_returns(net_returns, initial_value=config.initial_capital)
    metrics = _build_metrics(
        net_returns=net_returns,
        equity_curve=equity_curve,
        turnover=turnover,
        transaction_costs=transaction_costs,
        periods_per_year=config.periods_per_year,
    )
    metadata: dict[str, Any] = {
        "periods_per_year": float(config.periods_per_year),
        "execution_lag": int(config.execution_lag),
        "initial_capital": float(config.initial_capital),
        "allow_short": bool(config.allow_short),
        "max_leverage": float(config.max_leverage),
        "commission_rate": config.cost_config.commission_rate,
        "spread_rate": config.cost_config.spread_rate,
        "slippage_rate": config.cost_config.slippage_rate,
        "funding_rate": config.cost_config.funding_rate,
    }
    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        equity_curve=equity_curve,
        turnover=turnover,
        transaction_costs=transaction_costs,
        applied_weights=applied_weights,
        target_weights=target_aligned,
        metrics=metrics,
        metadata=metadata,
    )
