from __future__ import annotations

import numpy as np

from quant_platform.research.rates_derivatives import (
    plain_vanilla_swap_summary,
    sofr_futures_implied_rate,
)


def test_simplified_swap_is_zero_when_fixed_equals_forward() -> None:
    result = plain_vanilla_swap_summary(1_000_000, 0.04, 0.04, 0.03, 2.0, 2)

    assert np.isclose(result["receive_floating_pay_fixed_npv"], 0.0)
    assert result["real_market_status"] == "DATA_REQUIRED_FOR_REAL_MARKET_VALUATION"


def test_sofr_futures_implied_rate() -> None:
    assert sofr_futures_implied_rate(95.25)["implied_rate"] == 0.0475
