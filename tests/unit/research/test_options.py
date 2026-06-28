from __future__ import annotations

import numpy as np

from quant_platform.research.options import (
    black_scholes_price,
    option_scenario_table,
    protective_put_payoff,
    put_call_parity_gap,
)


def test_put_call_parity_includes_dividend_yield() -> None:
    spot = 100.0
    strike = 105.0
    rate = 0.04
    dividend_yield = 0.01
    maturity = 1.0
    volatility = 0.2
    call = black_scholes_price(spot, strike, rate, volatility, maturity, "call", dividend_yield)
    put = black_scholes_price(spot, strike, rate, volatility, maturity, "put", dividend_yield)

    gap = put_call_parity_gap(call, put, spot, strike, rate, maturity, dividend_yield)

    assert abs(gap) < 1e-10


def test_protective_put_floor_and_scenarios() -> None:
    payoff = protective_put_payoff(spot=100.0, strike=95.0, put_premium=2.0, points=3)
    scenarios = option_scenario_table(100.0, 0.03, 0.2)

    assert np.isclose(payoff[0]["net_payoff"], -7.0)
    assert [row["scenario"] for row in scenarios] == ["90%", "ATM", "110%"]
