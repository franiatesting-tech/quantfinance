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


def black_scholes_diagnostics(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity_years: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, float | str]:
    """Return explanatory Black-Scholes-Merton diagnostics for one option.

    The diagnostics are intentionally model-based and educational. They expose the
    mathematical inputs used by the paper: discount factors, d1/d2, moneyness,
    intrinsic value, time value, forward price, and breakeven.
    """

    inputs = _option_inputs(spot, strike, volatility, maturity_years)
    clean_type = _option_type(option_type)
    d1, d2 = _d1_d2(*inputs, rate, dividend_yield)
    price = black_scholes_price(
        spot, strike, rate, volatility, maturity_years, clean_type, dividend_yield
    )
    forward = spot * math.exp((rate - dividend_yield) * maturity_years)
    spot_dividend_discount = spot * math.exp(-dividend_yield * maturity_years)
    strike_discount = strike * math.exp(-rate * maturity_years)
    if clean_type == "call":
        intrinsic = max(spot - strike, 0.0)
        breakeven = strike + price
    else:
        intrinsic = max(strike - spot, 0.0)
        breakeven = strike - price
    time_value = max(price - intrinsic, 0.0)
    normal = NormalDist()
    return {
        "option_type": clean_type,
        "spot": float(spot),
        "strike": float(strike),
        "maturity_years": float(maturity_years),
        "rate": float(rate),
        "dividend_yield": float(dividend_yield),
        "volatility": float(volatility),
        "d1": float(d1),
        "d2": float(d2),
        "price": float(price),
        "moneyness_spot_over_strike": float(spot / strike),
        "forward_price": float(forward),
        "spot_dividend_discounted": float(spot_dividend_discount),
        "strike_discounted": float(strike_discount),
        "intrinsic_value": float(intrinsic),
        "time_value": float(time_value),
        "breakeven_at_maturity": float(breakeven),
        "risk_neutral_exercise_probability": float(
            normal.cdf(d2) if clean_type == "call" else normal.cdf(-d2)
        ),
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
    dividend_yield: float = 0.0,
) -> float:
    """Compute a Cox-Ross-Rubinstein European option price."""

    _option_inputs(spot, strike, volatility, maturity_years)
    _rate_inputs(rate, dividend_yield)
    clean_type = _option_type(option_type)
    if steps < 1:
        raise OptionModelError("steps must be >= 1.")
    dt = maturity_years / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend_yield) * dt)
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
    dividend_yield: float = 0.0,
) -> float:
    """Return parity gap: `C - P - (S exp(-qT) - K exp(-rT))`."""

    _rate_inputs(rate, dividend_yield)
    forward_parity = spot * math.exp(-dividend_yield * maturity_years)
    forward_parity -= strike * math.exp(-rate * maturity_years)
    return float(call_price - put_price - forward_parity)


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


def protective_put_payoff(
    spot: float,
    strike: float,
    put_premium: float = 0.0,
    points: int = 41,
) -> list[dict[str, float]]:
    """Build protective-put payoff rows for one stock plus one put option."""

    if points < 3:
        raise OptionModelError("points must be >= 3.")
    _option_inputs(spot, strike, 0.2, 1.0)
    if not np.isfinite(put_premium) or put_premium < 0:
        raise OptionModelError("put_premium must be finite and >= 0.")
    grid = np.linspace(0.5 * spot, 1.5 * spot, points)
    rows = []
    for terminal_spot in grid:
        stock_pnl = float(terminal_spot - spot)
        put_payoff = max(float(strike - terminal_spot), 0.0)
        rows.append(
            {
                "underlying_price": float(terminal_spot),
                "stock_pnl": stock_pnl,
                "put_payoff": put_payoff,
                "net_payoff": stock_pnl + put_payoff - put_premium,
            }
        )
    return rows


def option_scenario_table(
    spot: float,
    rate: float,
    volatility: float,
    maturity_years: float = 1.0,
    dividend_yield: float = 0.0,
    steps: int = 100,
) -> list[dict[str, float | str]]:
    """Return option analytics for 90%, ATM, and 110% strike scenarios."""

    rows: list[dict[str, float | str]] = []
    for label, multiplier in (("90%", 0.9), ("ATM", 1.0), ("110%", 1.1)):
        strike = spot * multiplier
        call = black_scholes_price(
            spot, strike, rate, volatility, maturity_years, "call", dividend_yield
        )
        put = black_scholes_price(
            spot, strike, rate, volatility, maturity_years, "put", dividend_yield
        )
        greeks = black_scholes_greeks(
            spot, strike, rate, volatility, maturity_years, "call", dividend_yield
        )
        rows.append(
            {
                "scenario": label,
                "strike": float(strike),
                "moneyness_spot_over_strike": float(spot / strike),
                "black_scholes_call": call,
                "black_scholes_put": put,
                "binomial_call": binomial_crr_price(
                    spot, strike, rate, volatility, maturity_years, steps, "call", dividend_yield
                ),
                "binomial_put": binomial_crr_price(
                    spot, strike, rate, volatility, maturity_years, steps, "put", dividend_yield
                ),
                "call_delta": float(greeks["delta"]),
                "call_gamma": float(greeks["gamma"]),
                "call_vega": float(greeks["vega"]),
                "call_theta_annual": float(greeks["theta_annual"]),
                "call_rho": float(greeks["rho"]),
                "call_intrinsic_value": max(float(spot - strike), 0.0),
                "call_time_value": max(float(call - max(spot - strike, 0.0)), 0.0),
                "put_intrinsic_value": max(float(strike - spot), 0.0),
                "put_time_value": max(float(put - max(strike - spot, 0.0)), 0.0),
                "put_call_parity_gap": put_call_parity_gap(
                    call, put, spot, strike, rate, maturity_years, dividend_yield
                ),
                "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
            }
        )
    return rows


def _d1_d2(
    spot: float,
    strike: float,
    volatility: float,
    maturity_years: float,
    rate: float,
    dividend_yield: float,
) -> tuple[float, float]:
    _rate_inputs(rate, dividend_yield)
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


def _rate_inputs(rate: float, dividend_yield: float = 0.0) -> None:
    if not np.isfinite(rate) or not np.isfinite(dividend_yield):
        raise OptionModelError("rate and dividend_yield must be finite.")


def _option_type(option_type: str) -> str:
    clean = option_type.lower().strip()
    if clean not in {"call", "put"}:
        raise OptionModelError("option_type must be 'call' or 'put'.")
    return clean
