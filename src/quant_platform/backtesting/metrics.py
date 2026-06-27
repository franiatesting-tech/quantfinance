"""Basic backtest performance metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_platform.features.returns import cumulative_returns
from quant_platform.risk.drawdown import max_drawdown

ArrayLike = Sequence[float] | np.ndarray | pd.Series


class PerformanceMetricError(ValueError):
    """Raised when performance metric inputs are invalid."""


def _validate_periods_per_year(periods_per_year: int | float) -> float:
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise PerformanceMetricError("periods_per_year must be finite and > 0.")
    return float(periods_per_year)


def _as_clean_return_series(returns: ArrayLike, name: str = "returns") -> pd.Series:
    try:
        series = (
            returns.astype(float)
            if isinstance(returns, pd.Series)
            else pd.Series(returns, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise PerformanceMetricError(f"{name} must be numeric.") from exc
    if series.empty:
        raise PerformanceMetricError(f"{name} must not be empty.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise PerformanceMetricError(f"{name} must not contain NaN or infinite values.")
    if (series < -1).any():
        raise PerformanceMetricError(f"{name} contains a simple return below -100%.")
    return series


def _excess_return_series(returns: pd.Series, risk_free_rate: float | pd.Series) -> pd.Series:
    if isinstance(risk_free_rate, int | float):
        if not np.isfinite(risk_free_rate):
            raise PerformanceMetricError("risk_free_rate must be finite.")
        return returns - float(risk_free_rate)
    risk_free = _as_clean_return_series(risk_free_rate, "risk_free_rate")
    aligned_risk_free = risk_free.reindex(returns.index)
    if aligned_risk_free.isna().any():
        raise PerformanceMetricError("risk_free_rate is missing observations after alignment.")
    return returns - aligned_risk_free


def annualized_return(returns: ArrayLike, periods_per_year: int | float) -> float:
    """Compute CAGR-style annualized return from periodic simple returns.

    Formula: `prod(1 + R_t) ** (periods_per_year / n) - 1`.
    """

    annualization = _validate_periods_per_year(periods_per_year)
    clean_returns = _as_clean_return_series(returns)
    total_growth = float((1.0 + clean_returns).prod())
    return float(total_growth ** (annualization / len(clean_returns)) - 1.0)


def annualized_volatility(returns: ArrayLike, periods_per_year: int | float) -> float:
    """Compute annualized volatility `std(R_t) * sqrt(A)`.

    The sample standard deviation uses `ddof=1`. With fewer than two observations,
    the result is `np.nan`.
    """

    annualization = _validate_periods_per_year(periods_per_year)
    clean_returns = _as_clean_return_series(returns)
    periodic_volatility = clean_returns.std(ddof=1)
    return float(periodic_volatility * np.sqrt(annualization))


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float | pd.Series = 0.0,
    periods_per_year: int | float = 252,
) -> float:
    """Compute annualized Sharpe ratio from periodic excess returns.

    Formula: `mean(R_t - R_f) / std(R_t - R_f) * sqrt(A)`. If volatility is zero or
    undefined, returns `np.nan` instead of dividing by zero. Benhamou (2019) documents
    that empirical Sharpe ratios have statistical estimation error, so this scalar should
    not be interpreted without sample-size context.
    """

    annualization = _validate_periods_per_year(periods_per_year)
    clean_returns = _as_clean_return_series(returns)
    excess = _excess_return_series(clean_returns, risk_free_rate)
    volatility = excess.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return float("nan")
    return float(excess.mean() / volatility * np.sqrt(annualization))


def sortino_ratio(
    returns: ArrayLike,
    risk_free_rate: float | pd.Series = 0.0,
    periods_per_year: int | float = 252,
) -> float:
    """Compute annualized Sortino ratio using downside deviation.

    Formula: `mean(excess) / sqrt(mean(min(excess, 0)^2)) * sqrt(A)`. If downside
    deviation is zero, returns `np.nan`.
    """

    annualization = _validate_periods_per_year(periods_per_year)
    clean_returns = _as_clean_return_series(returns)
    excess = _excess_return_series(clean_returns, risk_free_rate)
    downside = excess.clip(upper=0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return float("nan")
    return float(excess.mean() / downside_deviation * np.sqrt(annualization))


def calmar_ratio(returns: ArrayLike, periods_per_year: int | float = 252) -> float:
    """Compute Calmar ratio `annualized_return / abs(max_drawdown)`.

    Max drawdown is computed on the equity curve generated from the supplied periodic
    simple returns. If max drawdown is zero, returns `np.nan`.
    """

    annualization = _validate_periods_per_year(periods_per_year)
    clean_returns = _as_clean_return_series(returns)
    equity_curve = cumulative_returns(clean_returns, initial_value=1.0)
    mdd = float(max_drawdown(equity_curve))
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(annualized_return(clean_returns, annualization) / abs(mdd))


def hit_rate(returns: ArrayLike) -> float:
    """Compute the fraction of periods with strictly positive returns."""

    clean_returns = _as_clean_return_series(returns)
    return float((clean_returns > 0).mean())
