"""Backtest report assembly from vectorized backtest results."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_platform.backtesting.engine import BacktestResult
from quant_platform.risk.expected_shortfall import historical_expected_shortfall
from quant_platform.risk.var import historical_var


def _clean_metric(value: float) -> float | None:
    clean_value = float(value)
    if not np.isfinite(clean_value):
        return None
    return clean_value


def _result_summary(result: BacktestResult) -> dict[str, float | int | str | None]:
    losses = (-result.net_returns).clip(lower=0.0)
    return {
        "number_of_observations": int(len(result.net_returns)),
        "start_date": str(result.net_returns.index[0]),
        "end_date": str(result.net_returns.index[-1]),
        "annualized_return": _clean_metric(result.metrics["annualized_return"]),
        "annualized_volatility": _clean_metric(result.metrics["annualized_volatility"]),
        "sharpe_ratio": _clean_metric(result.metrics["sharpe_ratio"]),
        "sortino_ratio": _clean_metric(result.metrics["sortino_ratio"]),
        "calmar_ratio": _clean_metric(result.metrics["calmar_ratio"]),
        "max_drawdown": _clean_metric(result.metrics["max_drawdown"]),
        "historical_var_95": _clean_metric(historical_var(losses, alpha=0.95)),
        "historical_expected_shortfall_95": _clean_metric(
            historical_expected_shortfall(losses, alpha=0.95)
        ),
        "total_turnover": _clean_metric(result.metrics["total_turnover"]),
        "average_turnover": _clean_metric(result.metrics["average_turnover"]),
        "total_transaction_cost": _clean_metric(result.metrics["total_transaction_cost"]),
        "final_equity": _clean_metric(result.metrics["final_equity"]),
    }


def build_backtest_report(
    result: BacktestResult,
    benchmark_results: dict[str, BacktestResult] | None = None,
) -> dict[str, Any]:
    """Build a dictionary report for one strategy and optional benchmarks."""

    report: dict[str, Any] = _result_summary(result)
    report["metadata"] = dict(result.metadata)
    if benchmark_results:
        benchmarks: dict[str, Any] = {}
        strategy_final_equity = float(result.equity_curve.iloc[-1])
        for name, benchmark_result in benchmark_results.items():
            summary = _result_summary(benchmark_result)
            summary["excess_final_equity"] = _clean_metric(
                float(benchmark_result.equity_curve.iloc[-1]) - strategy_final_equity
            )
            benchmarks[name] = summary
        report["benchmarks"] = benchmarks
    return report
