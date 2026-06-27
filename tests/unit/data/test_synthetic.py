from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import AssetMetadata, Frequency, MarketType
from quant_platform.data.synthetic import (
    SyntheticDataError,
    generate_synthetic_market_bars,
    generate_synthetic_ohlcv,
)


def test_generate_synthetic_ohlcv_creates_valid_canonical_bars() -> None:
    asset = AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic", venue="SIM")

    bars = generate_synthetic_ohlcv(asset, start="2024-01-01", periods=5, seed=42)

    assert len(bars) == 5
    assert bars["asset_id"].unique().tolist() == ["SYN_EQ_001"]
    assert bars["timestamp"].dt.tz is not None
    assert run_market_data_quality_checks(bars) is bars


def test_generate_synthetic_ohlcv_is_reproducible_for_same_seed() -> None:
    asset = AssetMetadata("SYN_BTC", MarketType.CRYPTO, "USD", "synthetic", venue="SIM")

    first = generate_synthetic_ohlcv(asset, "2024-01-01", 4, Frequency.DAILY, seed=7)
    second = generate_synthetic_ohlcv(asset, "2024-01-01", 4, Frequency.DAILY, seed=7)

    pd.testing.assert_frame_equal(first, second)


def test_generate_synthetic_market_bars_supports_multiple_assets() -> None:
    assets = [
        AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
        AssetMetadata("SYN_EQ_002", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
    ]

    bars = generate_synthetic_market_bars(assets, start="2024-01-01", periods=3, seed=10)

    assert len(bars) == 6
    assert set(bars["asset_id"]) == {"SYN_EQ_001", "SYN_EQ_002"}
    assert run_market_data_quality_checks(bars) is bars


def test_generate_synthetic_ohlcv_rejects_invalid_start_price() -> None:
    asset = AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic")

    with pytest.raises(SyntheticDataError, match="start_price"):
        generate_synthetic_ohlcv(asset, "2024-01-01", 3, start_price=0.0)
