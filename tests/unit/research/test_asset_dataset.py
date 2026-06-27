from __future__ import annotations

import pandas as pd

from quant_platform.research.asset_dataset import load_quant_terminal_config, make_synthetic_ohlcv
from quant_platform.research.returns import close_prices_from_ohlcv, daily_simple_returns


def test_load_quant_terminal_config_and_make_synthetic_ohlcv() -> None:
    config = load_quant_terminal_config("configs/quant_terminal_3_stocks.yaml")

    ohlcv = make_synthetic_ohlcv(config, end=pd.Timestamp("2025-12-31", tz="UTC"), seed=7)
    prices = close_prices_from_ohlcv(
        ohlcv,
        [*config.selected_stocks, config.benchmark_symbol],
    )
    returns = daily_simple_returns(prices)

    assert config.selected_stocks == ("AAPL", "MSFT", "NVDA")
    assert set(prices.columns) == {"AAPL", "MSFT", "NVDA", "SPY"}
    assert len(prices) == config.lookback_years * 252
    assert returns.shape[0] == prices.shape[0] - 1
    assert (prices > 0).all().all()
