"""Synthetic OHLCV market data generation for local tests and demos.

The generator is intentionally simple and auditable. It creates deterministic
synthetic bars from a random seed, never downloads real data, and validates the
output against the canonical market bar quality checks before returning it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import AssetMetadata, Frequency, MarketType


class SyntheticDataError(ValueError):
    """Raised when synthetic market data parameters are invalid."""


def _positive_float(value: float, name: str, allow_zero: bool = False) -> float:
    clean_value = float(value)
    if not np.isfinite(clean_value):
        raise SyntheticDataError(f"{name} must be finite.")
    if allow_zero:
        if clean_value < 0:
            raise SyntheticDataError(f"{name} must be >= 0.")
    elif clean_value <= 0:
        raise SyntheticDataError(f"{name} must be > 0.")
    return clean_value


def _utc_timestamp(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _pandas_frequency(frequency: Frequency, market_type: MarketType) -> str:
    if frequency == Frequency.DAILY:
        return "B" if market_type == MarketType.EQUITY else "D"
    if frequency == Frequency.HOURLY:
        return "h"
    if frequency == Frequency.MINUTE:
        return "min"
    raise SyntheticDataError(f"Unsupported frequency: {frequency}")


def generate_synthetic_ohlcv(
    asset: AssetMetadata,
    start: str | pd.Timestamp,
    periods: int,
    frequency: Frequency = Frequency.DAILY,
    seed: int | None = 0,
    start_price: float = 100.0,
    drift: float = 0.0002,
    volatility: float = 0.02,
    volume_base: float = 1_000.0,
    available_delay: str | pd.Timedelta = "0min",
) -> pd.DataFrame:
    """Generate canonical OHLCV bars for one synthetic asset.

    Prices follow a clipped Gaussian simple-return process. This is not a market
    model and must not be used for inference; it exists to exercise data quality,
    feature, risk, portfolio, and reporting code without external providers.
    """

    if periods <= 0:
        raise SyntheticDataError("periods must be > 0.")
    clean_start_price = _positive_float(start_price, "start_price")
    clean_volatility = _positive_float(volatility, "volatility", allow_zero=True)
    clean_volume_base = _positive_float(volume_base, "volume_base", allow_zero=True)
    if not np.isfinite(drift):
        raise SyntheticDataError("drift must be finite.")

    market_type = MarketType(asset.market_type)
    clean_frequency = Frequency(frequency)
    start_timestamp = _utc_timestamp(start, "start")
    delay = pd.Timedelta(available_delay)
    if delay < pd.Timedelta(0):
        raise SyntheticDataError("available_delay must be >= 0.")

    timestamps = pd.date_range(
        start=start_timestamp,
        periods=periods,
        freq=_pandas_frequency(clean_frequency, market_type),
    )
    rng = np.random.default_rng(seed)

    returns = rng.normal(loc=float(drift), scale=clean_volatility, size=periods)
    returns = np.clip(returns, -0.95, None)
    close = clean_start_price * np.cumprod(1.0 + returns)

    open_prices = np.empty(periods, dtype=float)
    open_prices[0] = clean_start_price
    if periods > 1:
        open_prices[1:] = close[:-1]
    open_noise = rng.normal(loc=0.0, scale=clean_volatility * 0.15, size=periods)
    open_prices = np.maximum(open_prices * (1.0 + open_noise), np.finfo(float).eps)

    range_floor = max(clean_volatility * 0.25, 1e-4)
    intraperiod_range = np.maximum(
        np.abs(rng.normal(loc=clean_volatility, scale=clean_volatility * 0.25, size=periods)),
        range_floor,
    )
    high = np.maximum(open_prices, close) * (1.0 + intraperiod_range)
    low = np.minimum(open_prices, close) * np.maximum(1.0 - intraperiod_range, 0.01)

    volume_scale = clean_volume_base * 0.10
    volume = np.maximum(rng.normal(loc=clean_volume_base, scale=volume_scale, size=periods), 0.0)

    bars = pd.DataFrame(
        {
            "asset_id": asset.asset_id,
            "timestamp": timestamps,
            "available_at": timestamps + delay,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "market_type": market_type.value,
            "currency": asset.currency,
            "source": asset.source,
            "venue": asset.venue,
        }
    )
    return run_market_data_quality_checks(bars)


def generate_synthetic_market_bars(
    assets: Sequence[AssetMetadata],
    start: str | pd.Timestamp,
    periods: int,
    frequency: Frequency = Frequency.DAILY,
    seed: int | None = 0,
    start_price: float = 100.0,
    drift: float = 0.0002,
    volatility: float = 0.02,
    volume_base: float = 1_000.0,
    available_delay: str | pd.Timedelta = "0min",
) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for a non-empty asset universe."""

    if not assets:
        raise SyntheticDataError("assets must not be empty.")
    frames = []
    for position, asset in enumerate(assets):
        asset_seed = None if seed is None else int(seed) + position
        frames.append(
            generate_synthetic_ohlcv(
                asset=asset,
                start=start,
                periods=periods,
                frequency=frequency,
                seed=asset_seed,
                start_price=start_price,
                drift=drift,
                volatility=volatility,
                volume_base=volume_base,
                available_delay=available_delay,
            )
        )
    bars = pd.concat(frames, ignore_index=True)
    return run_market_data_quality_checks(bars)
