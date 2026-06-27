from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.backtesting.benchmarks import equal_weight_target_weights
from quant_platform.backtesting.engine import BacktestConfig, run_vectorized_backtest
from quant_platform.reporting.backtest_report import build_backtest_report


def sample_result():  # noqa: ANN201
    index = pd.date_range("2024-01-02", periods=5, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, 0.00, 0.02], "B": [0.00, 0.01, -0.01, 0.02, 0.01]},
        index=index,
    )
    target = equal_weight_target_weights(returns)
    return run_vectorized_backtest(returns, target, BacktestConfig(periods_per_year=252))


def test_backtest_report_contains_expected_keys() -> None:
    report = build_backtest_report(sample_result())

    expected_keys = {
        "number_of_observations",
        "start_date",
        "end_date",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "historical_var_95",
        "historical_expected_shortfall_95",
        "total_turnover",
        "average_turnover",
        "total_transaction_cost",
        "final_equity",
        "metadata",
    }
    assert expected_keys.issubset(report.keys())


def test_backtest_report_final_equity_matches_result() -> None:
    result = sample_result()
    report = build_backtest_report(result)

    assert report["final_equity"] == pytest.approx(result.equity_curve.iloc[-1])


def test_backtest_report_adds_benchmark_comparison() -> None:
    result = sample_result()
    benchmark = sample_result()

    report = build_backtest_report(result, {"equal_weight": benchmark})

    assert "benchmarks" in report
    assert "equal_weight" in report["benchmarks"]
    assert "excess_final_equity" in report["benchmarks"]["equal_weight"]


def test_backtest_report_has_no_nan_core_metrics_for_valid_inputs() -> None:
    report = build_backtest_report(sample_result())
    core_keys = [
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "historical_var_95",
        "historical_expected_shortfall_95",
        "final_equity",
    ]

    for key in core_keys:
        assert report[key] is not None
        assert not math.isnan(report[key])
