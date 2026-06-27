"""Market data quality checks for canonical OHLCV bars."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_platform.data.schemas import MarketBarSchema


class MarketDataQualityError(ValueError):
    """Raised when market data violates the canonical schema or quality rules."""


def _ensure_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")


def _bad_index_sample(df: pd.DataFrame, mask: pd.Series) -> list[object]:
    return list(df.index[mask].to_list()[:5])


def _numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected_columns = list(columns)
    try:
        return df[selected_columns].apply(pd.to_numeric, errors="coerce")
    except KeyError as exc:
        raise MarketDataQualityError(f"Missing required columns for numeric check: {exc}") from exc


def _finite_mask(values: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.isfinite(values.to_numpy(dtype=float)),
        index=values.index,
        columns=values.columns,
    )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """Validate that all required columns are present."""

    _ensure_dataframe(df)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise MarketDataQualityError(f"Missing required columns: {missing}")
    return df


def validate_no_duplicate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Validate no duplicate bars by asset, timestamp, source, and venue if present."""

    validate_required_columns(df, MarketBarSchema.duplicate_key_base_columns)
    key_columns = MarketBarSchema.duplicate_key_columns(tuple(df.columns))
    duplicated = df.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        raise MarketDataQualityError(
            "Duplicate market bars detected for key columns "
            f"{key_columns}; sample indices={_bad_index_sample(df, duplicated)}"
        )
    return df


def validate_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC inequalities: high >= max(open, close), low <= min(open, close)."""

    validate_required_columns(df, MarketBarSchema.price_columns)
    prices = _numeric_frame(df, MarketBarSchema.price_columns)
    finite_rows = _finite_mask(prices).all(axis=1)
    invalid = (
        ~finite_rows
        | (prices["high"] < prices[["open", "close"]].max(axis=1))
        | (prices["low"] > prices[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        raise MarketDataQualityError(
            "Invalid OHLC bars: require high >= max(open, close) and "
            f"low <= min(open, close); sample indices={_bad_index_sample(df, invalid)}"
        )
    return df


def validate_non_negative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Validate volume is finite and non-negative."""

    validate_required_columns(df, ("volume",))
    volume = _numeric_frame(df, ("volume",))
    invalid = (~_finite_mask(volume).all(axis=1)) | (volume["volume"] < 0)
    if invalid.any():
        raise MarketDataQualityError(
            "Invalid volume: require finite volume >= 0; "
            f"sample indices={_bad_index_sample(df, invalid)}"
        )
    return df


def validate_available_at_not_before_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Validate availability time is not earlier than the observation timestamp."""

    validate_required_columns(df, ("timestamp", "available_at"))
    timestamp = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    available_at = pd.to_datetime(df["available_at"], utc=True, errors="coerce")
    invalid = timestamp.isna() | available_at.isna() | (available_at < timestamp)
    if invalid.any():
        raise MarketDataQualityError(
            "Invalid temporal availability: require available_at >= timestamp; "
            f"sample indices={_bad_index_sample(df, invalid)}"
        )
    return df


def validate_prices_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC prices are finite and strictly positive."""

    validate_required_columns(df, MarketBarSchema.price_columns)
    prices = _numeric_frame(df, MarketBarSchema.price_columns)
    invalid = (~_finite_mask(prices)) | (prices <= 0)
    invalid_rows = invalid.any(axis=1)
    if invalid_rows.any():
        raise MarketDataQualityError(
            "Invalid prices: require finite open/high/low/close > 0; "
            f"sample indices={_bad_index_sample(df, invalid_rows)}"
        )
    return df


def run_market_data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run all canonical market bar checks and return the original DataFrame.

    The checks enforce the foundation conventions: complete schema, positive prices,
    OHLC consistency, non-negative volume, temporal availability, and no duplicate bars.
    """

    validate_required_columns(df, MarketBarSchema.required_columns)
    validate_prices_positive(df)
    validate_ohlc_consistency(df)
    validate_non_negative_volume(df)
    validate_available_at_not_before_timestamp(df)
    validate_no_duplicate_bars(df)
    return df
