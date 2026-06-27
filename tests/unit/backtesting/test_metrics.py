from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtesting.metrics import (
    PerformanceMetricError,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)


def test_annualized_return_for_manual_returns() -> None:
    returns = pd.Series([0.01, 0.01])

    assert annualized_return(returns, periods_per_year=2) == pytest.approx(0.0201)


def test_annualized_volatility_constant_returns_is_zero() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])

    assert annualized_volatility(returns, periods_per_year=252) == pytest.approx(0.0)


def test_sharpe_ratio_returns_nan_for_zero_volatility() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])

    assert np.isnan(sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252))


def test_sortino_ratio_with_downside_returns() -> None:
    returns = pd.Series([0.10, -0.05, 0.02, -0.03])

    result = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=1)

    assert result == pytest.approx(0.3429971702850177)


def test_calmar_ratio_uses_drawdown_on_equity_curve() -> None:
    returns = pd.Series([0.20, -0.25, 2.0 / 3.0])

    assert calmar_ratio(returns, periods_per_year=3) == pytest.approx(2.0)


def test_hit_rate_counts_strictly_positive_returns() -> None:
    returns = pd.Series([0.10, -0.20, 0.0, 0.30])

    assert hit_rate(returns) == pytest.approx(0.5)


def test_metrics_reject_nan_returns() -> None:
    with pytest.raises(PerformanceMetricError, match="NaN"):
        annualized_return([0.01, float("nan")], periods_per_year=252)


def test_metrics_reject_infinite_returns() -> None:
    with pytest.raises(PerformanceMetricError, match="infinite"):
        annualized_volatility([0.01, float("inf")], periods_per_year=252)


def test_metrics_reject_invalid_periods_per_year() -> None:
    with pytest.raises(PerformanceMetricError, match="periods_per_year"):
        sharpe_ratio([0.01, 0.02], periods_per_year=0)
