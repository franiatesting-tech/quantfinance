"""Parametric educational fixed-income analytics."""

from __future__ import annotations

import numpy as np


class FixedIncomeError(ValueError):
    """Raised when fixed-income inputs are invalid."""


def bond_summary(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    maturity_years: float,
    frequency: int = 2,
    yield_bump: float = 0.0001,
) -> dict[str, float | str]:
    """Compute bond price, duration, convexity, and bumped-price sensitivity."""

    _validate_bond_inputs(face_value, coupon_rate, yield_to_maturity, maturity_years, frequency)
    price = bond_price(face_value, coupon_rate, yield_to_maturity, maturity_years, frequency)
    macaulay = macaulay_duration(
        face_value, coupon_rate, yield_to_maturity, maturity_years, frequency
    )
    modified = macaulay / (1.0 + yield_to_maturity / frequency)
    convex = convexity(face_value, coupon_rate, yield_to_maturity, maturity_years, frequency)
    bumped_price = bond_price(
        face_value,
        coupon_rate,
        yield_to_maturity + yield_bump,
        maturity_years,
        frequency,
    )
    return {
        "price": price,
        "macaulay_duration_years": macaulay,
        "modified_duration_years": modified,
        "convexity": convex,
        "dv01": float((price - bumped_price) / (yield_bump / 0.0001)),
        "yield_bump": float(yield_bump),
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
    }


def bond_price(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """Price a fixed-coupon bond from discounted cash flows."""

    cash_flows = _cash_flows(face_value, coupon_rate, maturity_years, frequency)
    periods = np.arange(1, len(cash_flows) + 1)
    discount = (1.0 + yield_to_maturity / frequency) ** periods
    return float(np.sum(cash_flows / discount))


def macaulay_duration(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """Compute Macaulay duration in years."""

    cash_flows = _cash_flows(face_value, coupon_rate, maturity_years, frequency)
    periods = np.arange(1, len(cash_flows) + 1)
    discount = (1.0 + yield_to_maturity / frequency) ** periods
    pv = cash_flows / discount
    price = float(np.sum(pv))
    return float(np.sum((periods / frequency) * pv) / price)


def convexity(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    maturity_years: float,
    frequency: int = 2,
) -> float:
    """Compute discrete convexity for a fixed-coupon bond."""

    cash_flows = _cash_flows(face_value, coupon_rate, maturity_years, frequency)
    periods = np.arange(1, len(cash_flows) + 1)
    y = yield_to_maturity / frequency
    pv = cash_flows / (1.0 + y) ** periods
    price = float(np.sum(pv))
    convex = np.sum(periods * (periods + 1) * pv / (1.0 + y) ** 2)
    return float(convex / (price * frequency**2))


def _cash_flows(
    face_value: float,
    coupon_rate: float,
    maturity_years: float,
    frequency: int,
) -> np.ndarray:
    periods = int(round(maturity_years * frequency))
    if periods < 1:
        raise FixedIncomeError("bond must have at least one cash-flow period.")
    coupon = face_value * coupon_rate / frequency
    cash_flows = np.full(periods, coupon, dtype=float)
    cash_flows[-1] += face_value
    return cash_flows


def _validate_bond_inputs(
    face_value: float,
    coupon_rate: float,
    yield_to_maturity: float,
    maturity_years: float,
    frequency: int,
) -> None:
    values = (face_value, coupon_rate, yield_to_maturity, maturity_years)
    if any(not np.isfinite(value) for value in values):
        raise FixedIncomeError("bond inputs must be finite.")
    if face_value <= 0 or coupon_rate < 0 or maturity_years <= 0:
        raise FixedIncomeError("face value and maturity must be > 0; coupon must be >= 0.")
    if yield_to_maturity <= -1:
        raise FixedIncomeError("yield_to_maturity must be greater than -100%.")
    if frequency < 1:
        raise FixedIncomeError("frequency must be >= 1.")
