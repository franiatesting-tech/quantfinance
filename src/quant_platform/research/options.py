"""Parametric educational options analytics."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


class OptionModelError(ValueError):
    """Raised when option model inputs are invalid."""


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity_years: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """Compute Black-Scholes-Merton European option price."""

    inputs = _option_inputs(spot, strike, volatility, maturity_years)
    clean_type = _option_type(option_type)
    d1, d2 = _d1_d2(*inputs, rate, dividend_yield)
    normal = NormalDist()
    discount = math.exp(-rate * maturity_years)
    dividend_discount = math.exp(-dividend_yield * maturity_years)
    if clean_type == "call":
        return float(spot * dividend_discount * normal.cdf(d1) - strike * discount * normal.cdf(d2))
    return float(strike * discount * normal.cdf(-d2) - spot * dividend_discount * normal.cdf(-d1))


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity_years: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float | str]:
    """Compute core Black-Scholes Greeks for a European option."""

    inputs = _option_inputs(spot, strike, volatility, maturity_years)
    clean_type = _option_type(option_type)
    d1, d2 = _d1_d2(*inputs, rate, dividend_yield)
    normal = NormalDist()
    pdf_d1 = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    dividend_discount = math.exp(-dividend_yield * maturity_years)
    discount = math.exp(-rate * maturity_years)
    if clean_type == "call":
        delta = dividend_discount * normal.cdf(d1)
        theta = (
            -spot * dividend_discount * pdf_d1 * volatility / (2.0 * math.sqrt(maturity_years))
            - rate * strike * discount * normal.cdf(d2)
            + dividend_yield * spot * dividend_discount * normal.cdf(d1)
        )
        rho = strike * maturity_years * discount * normal.cdf(d2)
    else:
        delta = dividend_discount * (normal.cdf(d1) - 1.0)
        theta = (
            -spot * dividend_discount * pdf_d1 * volatility / (2.0 * math.sqrt(maturity_years))
            + rate * strike * discount * normal.cdf(-d2)
            - dividend_yield * spot * dividend_discount * normal.cdf(-d1)
        )
        rho = -strike * maturity_years * discount * normal.cdf(-d2)
    gamma = dividend_discount * pdf_d1 / (spot * volatility * math.sqrt(maturity_years))
    vega = spot * dividend_discount * pdf_d1 * math.sqrt(maturity_years)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta_annual": float(theta),
        "rho": float(rho),
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
    }


def binomial_crr_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity_years: float,
    steps: int = 100,
    option_type: str = "call",
) -> float:
    """Compute a Cox-Ross-Rubinstein European option price."""

    _option_inputs(spot, strike, volatility, maturity_years)
    clean_type = _option_type(option_type)
    if steps < 1:
        raise OptionModelError("steps must be >= 1.")
    dt = maturity_years / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp(rate * dt)
    probability = (growth - down) / (up - down)
    if probability < 0 or probability > 1:
        raise OptionModelError("CRR risk-neutral probability is outside [0, 1].")
    terminal = np.array([spot * up**j * down ** (steps - j) for j in range(steps + 1)])
    if clean_type == "call":
        values = np.maximum(terminal - strike, 0.0)
    else:
        values = np.maximum(strike - terminal, 0.0)
    discount = math.exp(-rate * dt)
    for _ in range(steps):
        values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
    return float(values[0])


def put_call_parity_gap(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity_years: float,
) -> float:
    """Return put-call parity gap: `C - P - (S - K exp(-rT))`."""

    return float(call_price - put_price - (spot - strike * math.exp(-rate * maturity_years)))


def payoff_profile(
    spot: float,
    strike: float,
    option_type: str = "call",
    points: int = 41,
) -> list[dict[str, float]]:
    """Build a simple option payoff profile around current spot."""

    if points < 3:
        raise OptionModelError("points must be >= 3.")
    _option_inputs(spot, strike, 0.2, 1.0)
    clean_type = _option_type(option_type)
    grid = np.linspace(0.5 * spot, 1.5 * spot, points)
    rows = []
    for terminal_spot in grid:
        payoff = max(float(terminal_spot - strike), 0.0)
        if clean_type == "put":
            payoff = max(float(strike - terminal_spot), 0.0)
        rows.append({"underlying_price": float(terminal_spot), "payoff": payoff})
    return rows


def _d1_d2(
    spot: float,
    strike: float,
    volatility: float,
    maturity_years: float,
    rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity_years
    ) / (volatility * math.sqrt(maturity_years))
    d2 = d1 - volatility * math.sqrt(maturity_years)
    return d1, d2


def _option_inputs(
    spot: float,
    strike: float,
    volatility: float,
    maturity_years: float,
) -> tuple[float, float, float, float]:
    values = (float(spot), float(strike), float(volatility), float(maturity_years))
    if any(not np.isfinite(value) for value in values):
        raise OptionModelError("option inputs must be finite.")
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity_years <= 0:
        raise OptionModelError("spot, strike, volatility, and maturity must be > 0.")
    return values


def _option_type(option_type: str) -> str:
    clean = option_type.lower().strip()
    if clean not in {"call", "put"}:
        raise OptionModelError("option_type must be 'call' or 'put'.")
    return clean
