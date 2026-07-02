from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.state_of_art.ornstein_uhlenbeck import (
    estimate_ou_from_spread,
    pairs_zscore_backtest,
    screen_ou_pairs,
)


def test_estimate_ou_from_spread_identifies_mean_reversion() -> None:
    rng = np.random.default_rng(7)
    values = [0.0]
    for _ in range(220):
        values.append(0.05 + 0.82 * values[-1] + rng.normal(0.0, 0.1))
    spread = pd.Series(values)

    result = estimate_ou_from_spread(spread)

    assert result["mean_reverting"] is True
    assert 0.0 < result["phi"] < 1.0
    assert result["half_life_days"] > 0


def test_screen_pairs_and_backtest_returns_research_metrics() -> None:
    rng = np.random.default_rng(11)
    index = pd.date_range("2020-01-01", periods=320, freq="B")
    base = np.cumsum(rng.normal(0.0002, 0.01, size=len(index)))
    y = 100.0 * np.exp(base + rng.normal(0.0, 0.01, size=len(index)))
    x = 80.0 * np.exp(base + rng.normal(0.0, 0.01, size=len(index)))
    z = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=len(index))))
    prices = pd.DataFrame({"Y": y, "X": x, "Z": z}, index=index)

    rows = screen_ou_pairs(prices, max_pairs=3)
    backtest = pairs_zscore_backtest(prices, rows[0])

    assert rows
    assert rows[0]["ou_mean_reverting"] is True
    assert backtest["observations"] > 100
    assert backtest["model_status"] == "RESEARCH_ONLY_BACKTEST_WITH_ONE_DAY_LAG"
