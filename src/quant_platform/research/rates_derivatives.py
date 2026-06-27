"""Parametric educational rates-derivatives analytics."""

from __future__ import annotations

import numpy as np


class RatesDerivativeError(ValueError):
    """Raised when rates-derivative inputs are invalid."""


def plain_vanilla_swap_summary(
    notional: float,
    fixed_rate: float,
    floating_forward_rate: float,
    discount_rate: float,
    maturity_years: float,
    payments_per_year: int = 2,
) -> dict[str, float | str]:
    """Compute simplified receive-floating/pay-fixed swap NPV and DV01."""

    _validate_swap_inputs(
        notional,
        fixed_rate,
        floating_forward_rate,
        discount_rate,
        maturity_years,
        payments_per_year,
    )
    periods = int(round(maturity_years * payments_per_year))
    times = np.arange(1, periods + 1) / payments_per_year
    discount_factors = 1.0 / (1.0 + discount_rate / payments_per_year) ** np.arange(1, periods + 1)
    fixed_leg = float(np.sum(notional * fixed_rate / payments_per_year * discount_factors))
    floating_leg = float(
        np.sum(notional * floating_forward_rate / payments_per_year * discount_factors)
    )
    bumped_fixed = float(
        np.sum(notional * (fixed_rate + 0.0001) / payments_per_year * discount_factors)
    )
    return {
        "receive_floating_pay_fixed_npv": floating_leg - fixed_leg,
        "fixed_leg_pv": fixed_leg,
        "floating_leg_pv": floating_leg,
        "fixed_leg_dv01": bumped_fixed - fixed_leg,
        "weighted_average_maturity_years": float(np.average(times, weights=discount_factors)),
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
        "real_market_status": "DATA_REQUIRED_FOR_REAL_MARKET_VALUATION",
    }


def sofr_futures_implied_rate(futures_price: float) -> dict[str, float | str]:
    """Convert a SOFR futures price into an educational implied annual rate."""

    clean_price = float(futures_price)
    if not np.isfinite(clean_price) or clean_price <= 0:
        raise RatesDerivativeError("futures_price must be finite and > 0.")
    return {
        "futures_price": clean_price,
        "implied_rate": (100.0 - clean_price) / 100.0,
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
        "real_market_status": "DATA_REQUIRED_FOR_REAL_MARKET_VALUATION",
    }


def _validate_swap_inputs(
    notional: float,
    fixed_rate: float,
    floating_forward_rate: float,
    discount_rate: float,
    maturity_years: float,
    payments_per_year: int,
) -> None:
    values = (notional, fixed_rate, floating_forward_rate, discount_rate, maturity_years)
    if any(not np.isfinite(value) for value in values):
        raise RatesDerivativeError("swap inputs must be finite.")
    if notional <= 0 or maturity_years <= 0 or payments_per_year < 1:
        raise RatesDerivativeError("notional, maturity, and payment frequency must be positive.")
    if discount_rate <= -1:
        raise RatesDerivativeError("discount_rate must be greater than -100%.")
