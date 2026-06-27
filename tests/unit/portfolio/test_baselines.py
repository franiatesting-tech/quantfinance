from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.portfolio.baselines import (
    PortfolioBaselineError,
    buy_and_hold_returns,
    equal_weight_weights,
    inverse_volatility_weights,
    momentum_weights,
)


def test_equal_weight_weights_sum_to_one() -> None:
    weights = equal_weight_weights(["A", "B", "C"])

    assert weights.sum() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_inverse_volatility_weights_match_manual_values() -> None:
    weights = inverse_volatility_weights(pd.Series({"A": 0.20, "B": 0.10}))

    assert weights["A"] == pytest.approx(1.0 / 3.0)
    assert weights["B"] == pytest.approx(2.0 / 3.0)
    assert weights.sum() == pytest.approx(1.0)


def test_inverse_volatility_weights_reject_zero_volatility() -> None:
    with pytest.raises(PortfolioBaselineError, match="strictly positive"):
        inverse_volatility_weights({"A": 0.0, "B": 0.10})


def test_buy_and_hold_returns_use_fixed_initial_shares() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 50.0]})

    result = buy_and_hold_returns(prices, weights={"A": 0.5, "B": 0.5})

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.05)


def test_momentum_weights_select_top_trailing_return_without_future_rows() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [100.0, 99.0, 98.0]})

    weights = momentum_weights(prices, lookback=1, top_n=1)

    assert weights.iloc[0].sum() == pytest.approx(0.0)
    assert weights.loc[1, "A"] == pytest.approx(1.0)
    assert weights.loc[1, "B"] == pytest.approx(0.0)
    assert weights.loc[2, "A"] == pytest.approx(1.0)
