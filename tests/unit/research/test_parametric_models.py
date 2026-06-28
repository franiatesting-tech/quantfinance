from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.exposure import exposure_profile, simulate_exposure_paths
from quant_platform.research.fixed_income import bond_summary
from quant_platform.research.hedging import hedge_contract_count, minimum_variance_hedge_ratio
from quant_platform.research.options import (
    binomial_crr_price,
    black_scholes_price,
    put_call_parity_gap,
)
from quant_platform.research.rates_derivatives import (
    plain_vanilla_swap_summary,
    sofr_futures_implied_rate,
)
from quant_platform.research.var_models import compute_var_summary


def test_var_summary_uses_positive_loss_convention() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.04, 0.0])

    summary = compute_var_summary(returns, alpha=0.8)

    assert summary["loss_sign_convention"] == "L_t = -r_t (standard, no clip)"
    assert summary["parametric_normal"]["model_status"] == "PARAMETRIC_EDUCATIONAL_MODEL"


def test_options_put_call_parity_and_binomial_price() -> None:
    call = black_scholes_price(100, 100, 0.05, 0.2, 1.0, "call")
    put = black_scholes_price(100, 100, 0.05, 0.2, 1.0, "put")
    binomial_call = binomial_crr_price(100, 100, 0.05, 0.2, 1.0, steps=50, option_type="call")

    assert put_call_parity_gap(call, put, 100, 100, 0.05, 1.0) == pytest.approx(0.0)
    assert binomial_call == pytest.approx(call, rel=0.03)


def test_fixed_income_rates_hedging_and_exposure_models_are_labeled() -> None:
    bond = bond_summary(1000, 0.04, 0.045, 5, 2)
    swap = plain_vanilla_swap_summary(1_000_000, 0.04, 0.038, 0.035, 5, 2)
    sofr = sofr_futures_implied_rate(95.25)
    hedge = minimum_variance_hedge_ratio(
        pd.Series([0.01, -0.01, 0.02, -0.02]),
        pd.Series([0.008, -0.008, 0.015, -0.015]),
    )
    contracts = hedge_contract_count(1_000_000, 200_000, float(hedge["hedge_ratio"]))
    paths = simulate_exposure_paths(10, 4, volatility=1000, seed=1)
    exposure = exposure_profile(paths)

    assert bond["model_status"] == "PARAMETRIC_EDUCATIONAL_MODEL"
    assert swap["real_market_status"] == "DATA_REQUIRED_FOR_REAL_MARKET_VALUATION"
    assert sofr["implied_rate"] == pytest.approx(0.0475)
    assert contracts["real_market_status"] == "DATA_REQUIRED_FOR_REAL_HEDGE"
    assert exposure["model_status"] == "PARAMETRIC_EDUCATIONAL_MODEL"
    assert np.asarray(paths).shape == (10, 4)
