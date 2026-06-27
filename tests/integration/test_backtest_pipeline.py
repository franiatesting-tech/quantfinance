from __future__ import annotations

import pandas as pd

from quant_platform.backtesting.benchmarks import (
    buy_and_hold_target_weights,
    equal_weight_target_weights,
)
from quant_platform.backtesting.costs import TransactionCostConfig
from quant_platform.backtesting.engine import BacktestConfig, run_vectorized_backtest
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import AssetMetadata, Frequency, MarketType
from quant_platform.data.synthetic import generate_synthetic_market_bars
from quant_platform.features.returns import simple_returns
from quant_platform.reporting.backtest_report import build_backtest_report


def test_full_vectorized_backtest_pipeline_is_reproducible() -> None:
    assets = [
        AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
        AssetMetadata("SYN_EQ_002", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
        AssetMetadata("SYN_EQ_003", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
    ]
    bars = generate_synthetic_market_bars(
        assets,
        start="2024-01-01",
        periods=60,
        frequency=Frequency.DAILY,
        seed=123,
    )
    run_market_data_quality_checks(bars)
    close = bars.pivot(index="timestamp", columns="asset_id", values="close").sort_index()
    close = close.astype(float)
    returns = simple_returns(close).dropna(axis=0, how="any")

    target_weights = equal_weight_target_weights(returns)
    config = BacktestConfig(
        periods_per_year=252,
        execution_lag=1,
        cost_config=TransactionCostConfig.from_bps(
            commission_bps=1.0,
            spread_bps=2.0,
            slippage_bps=1.0,
        ),
        initial_capital=1_000_000.0,
    )
    result = run_vectorized_backtest(returns, target_weights, config)

    benchmark_weights = buy_and_hold_target_weights(close)
    benchmark = run_vectorized_backtest(returns, benchmark_weights, config)
    report = build_backtest_report(result, {"buy_and_hold": benchmark})

    assert (result.equity_curve > 0).all()
    assert not result.net_returns.empty
    assert (result.transaction_costs >= 0).all()
    assert report["number_of_observations"] == len(result.net_returns)
    assert "annualized_return" in report
    assert "buy_and_hold" in report["benchmarks"]
    assert result.applied_weights.iloc[0].sum() == 0.0
    pd.testing.assert_series_equal(
        result.applied_weights.iloc[1],
        target_weights.iloc[0],
        check_names=False,
    )

    repeated_bars = generate_synthetic_market_bars(
        assets,
        start="2024-01-01",
        periods=60,
        frequency=Frequency.DAILY,
        seed=123,
    )
    repeated_close = repeated_bars.pivot(index="timestamp", columns="asset_id", values="close")
    repeated_returns = simple_returns(repeated_close.sort_index().astype(float)).dropna(axis=0)
    repeated_target = equal_weight_target_weights(repeated_returns)
    repeated = run_vectorized_backtest(repeated_returns, repeated_target, config)

    pd.testing.assert_series_equal(result.net_returns, repeated.net_returns)
