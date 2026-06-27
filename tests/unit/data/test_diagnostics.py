from __future__ import annotations

import pandas as pd

from quant_platform.data.diagnostics import (
    build_data_quality_summary,
    detect_date_gaps,
    summarize_ohlcv_coverage,
)
from quant_platform.data.schemas import MarketType


def _bars(asset_id: str = "SPY", market_type: str = "equity") -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "asset_id": asset_id,
            "timestamp": timestamp,
            "available_at": timestamp + pd.Timedelta(days=1),
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000.0, 1100.0, 1200.0],
            "market_type": market_type,
            "currency": "USD" if market_type == MarketType.EQUITY.value else "USDT",
            "source": "unit_test",
            "venue": "TEST",
        }
    )


def test_summarize_ohlcv_coverage_valid_dataset() -> None:
    summary = summarize_ohlcv_coverage(_bars())

    assert summary.loc[0, "asset_id"] == "SPY"
    assert summary.loc[0, "row_count"] == 3
    assert summary.loc[0, "missing_required_values"] == 0
    assert summary.loc[0, "available_at_violations"] == 0


def test_summary_flags_duplicates_missing_and_available_at_violations() -> None:
    frame = pd.concat([_bars(), _bars().iloc[[0]]], ignore_index=True)
    frame.loc[1, "close"] = None
    frame.loc[2, "available_at"] = frame.loc[2, "timestamp"] - pd.Timedelta(days=1)

    quality = build_data_quality_summary(frame)

    assert "duplicate_rows" in quality["failed_checks"]
    assert "missing_required_values" in quality["failed_checks"]
    assert "available_at_violations" in quality["failed_checks"]


def test_detect_crypto_daily_gap_is_failed_check() -> None:
    frame = _bars("BTCUSDT", MarketType.CRYPTO.value).iloc[[0, 2]].reset_index(drop=True)

    gaps = detect_date_gaps(frame)

    assert len(gaps) == 1
    assert gaps.loc[0, "severity"] == "failed_check"
    assert gaps.loc[0, "missing_periods"] == 1


def test_detect_equity_daily_gap_is_warning() -> None:
    frame = _bars("SPY", MarketType.EQUITY.value).iloc[[0, 2]].reset_index(drop=True)

    gaps = detect_date_gaps(frame)

    assert len(gaps) == 1
    assert gaps.loc[0, "severity"] == "warning"


def test_zero_volume_is_warning_not_failed_check() -> None:
    frame = _bars()
    frame.loc[1, "volume"] = 0.0

    quality = build_data_quality_summary(frame)

    assert "zero_volume_rows" in quality["warnings"]
    assert "zero_volume_rows" not in quality["failed_checks"]
