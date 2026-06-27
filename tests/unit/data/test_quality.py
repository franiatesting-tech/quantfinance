from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.data.quality import (
    MarketDataQualityError,
    run_market_data_quality_checks,
    validate_available_at_not_before_timestamp,
    validate_no_duplicate_bars,
    validate_non_negative_volume,
    validate_ohlc_consistency,
    validate_prices_positive,
    validate_required_columns,
)
from quant_platform.data.schemas import MarketBarSchema


def valid_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["SYN_EQ_001", "SYN_EQ_001"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "available_at": pd.to_datetime(["2024-01-01 22:00", "2024-01-02 22:00"], utc=True),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
            "market_type": ["equity", "equity"],
            "currency": ["USD", "USD"],
            "source": ["synthetic", "synthetic"],
            "venue": ["SIM", "SIM"],
        }
    )


def test_run_market_data_quality_checks_accepts_valid_data() -> None:
    df = valid_market_data()

    result = run_market_data_quality_checks(df)

    assert result is df


def test_validate_required_columns_rejects_missing_column() -> None:
    df = valid_market_data().drop(columns=["close"])

    with pytest.raises(MarketDataQualityError, match="Missing required columns"):
        validate_required_columns(df, MarketBarSchema.required_columns)


def test_validate_no_duplicate_bars_rejects_duplicate_keys() -> None:
    df = pd.concat([valid_market_data(), valid_market_data().iloc[[0]]], ignore_index=True)

    with pytest.raises(MarketDataQualityError, match="Duplicate market bars"):
        validate_no_duplicate_bars(df)


def test_validate_ohlc_consistency_rejects_high_below_close() -> None:
    df = valid_market_data()
    df.loc[0, "high"] = 100.5

    with pytest.raises(MarketDataQualityError, match="Invalid OHLC bars"):
        validate_ohlc_consistency(df)


def test_validate_ohlc_consistency_rejects_low_above_open() -> None:
    df = valid_market_data()
    df.loc[0, "low"] = 100.5

    with pytest.raises(MarketDataQualityError, match="Invalid OHLC bars"):
        validate_ohlc_consistency(df)


def test_validate_non_negative_volume_rejects_negative_volume() -> None:
    df = valid_market_data()
    df.loc[0, "volume"] = -1.0

    with pytest.raises(MarketDataQualityError, match="Invalid volume"):
        validate_non_negative_volume(df)


def test_validate_available_at_rejects_look_ahead_violation() -> None:
    df = valid_market_data()
    df.loc[0, "available_at"] = pd.Timestamp("2023-12-31 23:59", tz="UTC")

    with pytest.raises(MarketDataQualityError, match="available_at >= timestamp"):
        validate_available_at_not_before_timestamp(df)


def test_validate_prices_positive_rejects_zero_price() -> None:
    df = valid_market_data()
    df.loc[0, "close"] = 0.0

    with pytest.raises(MarketDataQualityError, match="Invalid prices"):
        validate_prices_positive(df)


def test_validate_prices_positive_rejects_infinite_price() -> None:
    df = valid_market_data()
    df.loc[0, "open"] = float("inf")

    with pytest.raises(MarketDataQualityError, match="Invalid prices"):
        validate_prices_positive(df)
