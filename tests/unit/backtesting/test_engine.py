from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtesting.costs import TransactionCostConfig
from quant_platform.backtesting.engine import (
    BacktestConfig,
    BacktestEngineError,
    run_vectorized_backtest,
)


def returns_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.03], "B": [0.00, 0.01, 0.02, -0.01]},
        index=index,
    )


def constant_weights(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({"A": 0.5, "B": 0.5}, index=index)


def test_backtest_without_costs_with_constant_weights() -> None:
    returns = returns_frame()
    target = constant_weights(returns.index)
    config = BacktestConfig(periods_per_year=252, initial_capital=100.0)

    result = run_vectorized_backtest(returns, target, config)

    expected_gross = pd.Series([0.0, 0.015, 0.005, 0.01], index=returns.index)
    pd.testing.assert_series_equal(result.gross_returns, expected_gross)
    pd.testing.assert_series_equal(result.net_returns, expected_gross)
    assert result.equity_curve.iloc[-1] == pytest.approx(100.0 * 1.015 * 1.005 * 1.01)


def test_execution_lag_applies_previous_day_weights() -> None:
    returns = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0]},
        index=pd.date_range("2024-01-02", periods=2, freq="D", tz="UTC"),
    )
    target = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=returns.index)

    result = run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))

    assert result.gross_returns.iloc[0] == pytest.approx(0.0)
    assert result.gross_returns.iloc[1] == pytest.approx(0.0)
    assert result.applied_weights.iloc[1]["A"] == pytest.approx(1.0)


def test_engine_does_not_use_same_day_weights_for_same_day_returns() -> None:
    returns = pd.DataFrame(
        {"A": [0.50], "B": [0.0]},
        index=pd.date_range("2024-01-02", periods=1, freq="D", tz="UTC"),
    )
    target = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=returns.index)

    result = run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))

    assert result.gross_returns.iloc[0] == pytest.approx(0.0)


def test_transaction_costs_reduce_net_returns() -> None:
    returns = returns_frame()
    target = constant_weights(returns.index)
    config = BacktestConfig(
        periods_per_year=252,
        cost_config=TransactionCostConfig.from_bps(commission_bps=10.0),
    )

    result = run_vectorized_backtest(returns, target, config)

    assert (result.transaction_costs >= 0).all()
    assert result.net_returns.sum() < result.gross_returns.sum()


def test_engine_rejects_misaligned_columns() -> None:
    returns = returns_frame()
    target = pd.DataFrame({"A": 0.5, "C": 0.5}, index=returns.index)

    with pytest.raises(BacktestEngineError, match="columns"):
        run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))


def test_engine_rejects_short_weights_when_not_allowed() -> None:
    returns = returns_frame()
    target = pd.DataFrame({"A": 1.2, "B": -0.2}, index=returns.index)

    with pytest.raises(BacktestEngineError, match="short"):
        run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))


def test_engine_rejects_excessive_leverage() -> None:
    returns = returns_frame()
    target = pd.DataFrame({"A": 0.8, "B": 0.8}, index=returns.index)

    with pytest.raises(BacktestEngineError, match="max_leverage"):
        run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))


def test_engine_rejects_nan_and_infinite_inputs() -> None:
    returns = returns_frame()
    returns.iloc[0, 0] = np.nan
    target = constant_weights(returns.index)

    with pytest.raises(BacktestEngineError, match="NaN or infinite"):
        run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))


def test_engine_rejects_execution_lag_less_than_one() -> None:
    with pytest.raises(BacktestEngineError, match="execution_lag"):
        BacktestConfig(periods_per_year=252, execution_lag=0)
