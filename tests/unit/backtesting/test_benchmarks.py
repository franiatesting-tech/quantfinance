from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.backtesting.benchmarks import (
    buy_and_hold_target_weights,
    equal_weight_target_weights,
    inverse_volatility_target_weights,
    momentum_target_weights,
)


def price_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [100.0, 99.0, 98.0, 97.0]},
        index=index,
    )


def test_equal_weight_target_weights_are_constant() -> None:
    prices = price_frame()

    weights = equal_weight_target_weights(prices)

    assert weights.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert weights.iloc[0]["A"] == pytest.approx(0.5)
    assert weights.iloc[-1]["B"] == pytest.approx(0.5)


def test_buy_and_hold_target_weights_drift_with_prices() -> None:
    prices = price_frame()

    weights = buy_and_hold_target_weights(prices)

    assert weights.iloc[0]["A"] == pytest.approx(0.5)
    assert weights.iloc[-1]["A"] > 0.5
    assert weights.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_inverse_volatility_target_weights_use_only_trailing_window() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {"A": [0.01, 0.03, 0.02, 0.01, 0.99], "B": [0.01, 0.01, 0.01, 0.04, -0.99]},
        index=index,
    )

    baseline = inverse_volatility_target_weights(returns, lookback=3)
    changed_future = returns.copy()
    changed_future.iloc[-1] = [10.0, -10.0]
    changed = inverse_volatility_target_weights(changed_future, lookback=3)

    pd.testing.assert_series_equal(baseline.iloc[3], changed.iloc[3])


def test_momentum_target_weights_select_top_trailing_asset() -> None:
    prices = price_frame()

    weights = momentum_target_weights(prices, lookback=1, top_n=1)

    assert weights.iloc[0].sum() == pytest.approx(0.0)
    assert weights.iloc[1]["A"] == pytest.approx(1.0)
    assert weights.iloc[2]["B"] == pytest.approx(0.0)


def test_insufficient_lookback_produces_initial_zero_weights() -> None:
    prices = price_frame()
    returns = prices.pct_change(fill_method=None).dropna()

    inv_vol = inverse_volatility_target_weights(returns, lookback=3)
    momentum = momentum_target_weights(prices, lookback=2, top_n=1)

    assert inv_vol.iloc[0].sum() == pytest.approx(0.0)
    assert inv_vol.iloc[1].sum() == pytest.approx(0.0)
    assert momentum.iloc[0].sum() == pytest.approx(0.0)
    assert momentum.iloc[1].sum() == pytest.approx(0.0)


def test_benchmark_weights_sum_to_one_when_signal_is_valid() -> None:
    prices = price_frame()
    returns = prices.pct_change(fill_method=None).dropna()

    inv_vol = inverse_volatility_target_weights(returns, lookback=3)
    momentum = momentum_target_weights(prices, lookback=2, top_n=1)

    assert inv_vol.iloc[-1].sum() == pytest.approx(1.0)
    assert momentum.iloc[-1].sum() == pytest.approx(1.0)
