"""Parametric hedge analytics for exposure management."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class HedgingError(ValueError):
    """Raised when hedge inputs are invalid."""


def minimum_variance_hedge_ratio(
    spot_returns: pd.Series, hedge_returns: pd.Series
) -> dict[str, float | str]:
    """Compute minimum-variance hedge ratio from aligned spot and hedge returns."""

    spot, hedge = _aligned_returns(spot_returns, hedge_returns)
    hedge_variance = float(hedge.var(ddof=1))
    if hedge_variance == 0 or not np.isfinite(hedge_variance):
        raise HedgingError("hedge return variance must be positive and finite.")
    covariance = float(spot.cov(hedge))
    ratio = covariance / hedge_variance
    correlation = float(spot.corr(hedge))
    return {
        "hedge_ratio": ratio,
        "correlation": correlation,
        "spot_volatility": float(spot.std(ddof=1)),
        "hedge_volatility": float(hedge.std(ddof=1)),
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
        "real_market_status": "DATA_REQUIRED_FOR_REAL_HEDGE",
    }


def hedge_contract_count(
    exposure_value: float,
    futures_contract_value: float,
    hedge_ratio: float,
) -> dict[str, float | int | str]:
    """Estimate contracts needed for a parametric futures hedge."""

    values = (exposure_value, futures_contract_value, hedge_ratio)
    if any(not np.isfinite(value) for value in values):
        raise HedgingError("hedge contract inputs must be finite.")
    if exposure_value <= 0 or futures_contract_value <= 0:
        raise HedgingError("exposure and contract value must be > 0.")
    raw_contracts = hedge_ratio * exposure_value / futures_contract_value
    return {
        "raw_contracts": float(raw_contracts),
        "rounded_contracts": int(math.ceil(abs(raw_contracts))) * (1 if raw_contracts >= 0 else -1),
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
        "real_market_status": "DATA_REQUIRED_FOR_REAL_HEDGE",
    }


def _aligned_returns(
    spot_returns: pd.Series, hedge_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    if not isinstance(spot_returns, pd.Series) or not isinstance(hedge_returns, pd.Series):
        raise HedgingError("returns must be pandas Series.")
    spot, hedge = spot_returns.astype(float).align(hedge_returns.astype(float), join="inner")
    spot = spot.dropna()
    hedge = hedge.reindex(spot.index).dropna()
    spot = spot.reindex(hedge.index)
    if len(spot) < 2:
        raise HedgingError("at least two aligned return observations are required.")
    if (
        not np.isfinite(spot.to_numpy(dtype=float)).all()
        or not np.isfinite(hedge.to_numpy(dtype=float)).all()
    ):
        raise HedgingError("returns must be finite.")
    return spot, hedge
