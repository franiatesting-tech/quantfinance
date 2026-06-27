"""Return and price-shape helpers for research analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.features.returns import log_returns, simple_returns


class ResearchReturnError(ValueError):
    """Raised when research return inputs are invalid."""


def close_prices_from_ohlcv(
    ohlcv: pd.DataFrame,
    symbols: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Pivot normalized long OHLCV data into a wide close-price frame."""

    required = {"asset_id", "timestamp", "close"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ResearchReturnError(f"OHLCV frame is missing columns: {sorted(missing)}")
    frame = ohlcv.loc[:, ["asset_id", "timestamp", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["asset_id", "timestamp", "close"])
    if frame.empty:
        raise ResearchReturnError("OHLCV frame has no valid close prices.")
    wide = frame.pivot_table(
        index="timestamp",
        columns="asset_id",
        values="close",
        aggfunc="last",
    ).sort_index()
    if symbols is not None:
        missing_symbols = [symbol for symbol in symbols if symbol not in wide.columns]
        if missing_symbols:
            raise ResearchReturnError(f"Missing close prices for symbols: {missing_symbols}")
        wide = wide.loc[:, list(symbols)]
    wide = wide.dropna(how="any")
    if wide.empty:
        raise ResearchReturnError("Close-price frame is empty after alignment.")
    return wide.astype(float)


def daily_simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily simple returns and drop the initial undefined observation."""

    return simple_returns(prices).dropna(how="any")


def daily_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily log returns and drop the initial undefined observation."""

    return log_returns(prices).dropna(how="any")


def annual_rate_to_periodic(annual_rate: float, periods_per_year: int | float = 252) -> float:
    """Convert an annual simple rate to a same-frequency periodic rate."""

    if not np.isfinite(annual_rate):
        raise ResearchReturnError("annual_rate must be finite.")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ResearchReturnError("periods_per_year must be finite and > 0.")
    return float((1.0 + annual_rate) ** (1.0 / float(periods_per_year)) - 1.0)


def price_return_rows(
    prices: pd.Series, returns: pd.Series | None = None
) -> list[dict[str, float | str]]:
    """Serialize one price series and optional returns for charting."""

    clean_prices = prices.astype(float).dropna()
    clean_returns = returns.astype(float).dropna() if returns is not None else None
    rows: list[dict[str, float | str]] = []
    for timestamp, price in clean_prices.items():
        row: dict[str, float | str] = {"timestamp": pd.Timestamp(timestamp).isoformat()}
        row["price"] = float(price)
        if clean_returns is not None and timestamp in clean_returns.index:
            row["return"] = float(clean_returns.loc[timestamp])
        rows.append(row)
    return rows
