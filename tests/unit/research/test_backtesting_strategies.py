from __future__ import annotations

import pandas as pd

from quant_platform.research.backtesting_strategies import moving_average_crossover_weights


def test_moving_average_signal_uses_current_and_past_prices_only() -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0, 2.0], "B": [4.0, 3.0, 2.0, 1.0]})

    weights = moving_average_crossover_weights(prices, short_window=1, long_window=2)

    assert weights.iloc[1].to_dict() == {"A": 1.0, "B": 0.0}
    assert weights.iloc[3].to_dict() == {"A": 0.0, "B": 0.0}
