from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from quant_platform.data.registry import (
    DatasetRegistryError,
    list_dataset_versions,
    load_dataset,
    read_dataset_manifest,
    register_dataset,
)
from quant_platform.data.schemas import AssetMetadata, DatasetMetadata, Frequency, MarketType
from quant_platform.data.synthetic import generate_synthetic_ohlcv


def test_register_and_load_dataset_round_trips_market_bars(tmp_path) -> None:  # noqa: ANN001
    asset = AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic", venue="SIM")
    bars = generate_synthetic_ohlcv(asset, "2024-01-01", periods=4, seed=3)
    metadata = DatasetMetadata(
        dataset_id="synthetic_equity",
        version="v1",
        source="synthetic",
        market_type=MarketType.EQUITY,
        frequency=Frequency.DAILY,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    registered = register_dataset(bars, metadata, tmp_path)
    loaded, loaded_manifest = load_dataset(tmp_path, "synthetic_equity", "v1")

    assert registered.row_count == len(bars)
    assert loaded_manifest == registered
    assert list_dataset_versions(tmp_path, "synthetic_equity") == ["v1"]
    assert loaded["asset_id"].tolist() == bars["asset_id"].tolist()
    assert loaded["close"].tolist() == pytest.approx(bars["close"].tolist())


def test_register_dataset_prevents_overwriting_existing_version(tmp_path) -> None:  # noqa: ANN001
    asset = AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic", venue="SIM")
    bars = generate_synthetic_ohlcv(asset, "2024-01-01", periods=3, seed=4)
    metadata = DatasetMetadata(
        dataset_id="synthetic_equity",
        version="v1",
        source="synthetic",
        market_type=MarketType.EQUITY,
        frequency=Frequency.DAILY,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    register_dataset(bars, metadata, tmp_path)

    with pytest.raises(DatasetRegistryError, match="already exists"):
        register_dataset(bars, metadata, tmp_path)


def test_read_dataset_manifest_rejects_missing_dataset(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(DatasetRegistryError, match="manifest not found"):
        read_dataset_manifest(tmp_path, "missing", "v1")


def test_load_dataset_accepts_mixed_timestamp_precision(tmp_path) -> None:  # noqa: ANN001
    asset = AssetMetadata("BTCUSDT", MarketType.CRYPTO, "USDT", "binance_public", venue="BINANCE")
    bars = generate_synthetic_ohlcv(asset, "2024-01-01", periods=2, seed=5)
    bars["available_at"] = pd.to_datetime(bars["available_at"], utc=True) + pd.Timedelta(
        milliseconds=999
    )
    metadata = DatasetMetadata(
        dataset_id="mixed_precision",
        version="v1",
        source="synthetic",
        market_type=MarketType.CRYPTO,
        frequency=Frequency.DAILY,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    register_dataset(bars, metadata, tmp_path)
    loaded, _ = load_dataset(tmp_path, "mixed_precision", "v1")

    assert str(loaded["available_at"].dt.tz) == "UTC"
    assert (loaded["available_at"] >= loaded["timestamp"]).all()
