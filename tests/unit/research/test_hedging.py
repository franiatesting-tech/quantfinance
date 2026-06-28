from __future__ import annotations

import pandas as pd

from quant_platform.research.hedging import hedge_contract_count, minimum_variance_hedge_ratio


def test_minimum_variance_hedge_ratio_and_signed_contracts() -> None:
    hedge = pd.Series([0.01, -0.02, 0.03, -0.01])
    spot = 2.0 * hedge

    ratio = minimum_variance_hedge_ratio(spot, hedge)
    contracts = hedge_contract_count(1_000_000, 200_000, ratio["hedge_ratio"])

    assert ratio["hedge_ratio"] == 2.0
    assert contracts["rounded_contracts"] == -10
    assert contracts["hedge_direction"] == "short_hedge"
