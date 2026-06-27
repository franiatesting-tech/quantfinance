from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_platform.data.ingestion import (
    IngestionError,
    download_combined_daily_universe,
    load_universe_config,
    register_real_dataset,
)
from quant_platform.data.providers.base import OHLCVRequest, OHLCVResponse
from quant_platform.data.schemas import MarketType


def bars(symbol: str, market_type: MarketType, source: str, venue: str) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "asset_id": symbol,
            "timestamp": timestamp,
            "available_at": timestamp + pd.Timedelta(days=1),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
            "market_type": market_type.value,
            "currency": "USD" if market_type == MarketType.EQUITY else "USDT",
            "source": source,
            "venue": venue,
        }
    )


class FakeProvider:
    def __init__(self, market_type: MarketType) -> None:
        self.market_type = market_type

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        symbol = request.symbols[0]
        source = "fake_equity" if self.market_type == MarketType.EQUITY else "fake_crypto"
        venue = "FAKE_EQ" if self.market_type == MarketType.EQUITY else "FAKE_CRYPTO"
        return OHLCVResponse(
            data=bars(symbol, self.market_type, source, venue),
            successful_symbols=(symbol,),
            failed_symbols={symbol + "_FAIL": "simulated"},
            metadata={"provider": source},
        )


class FailingProvider:
    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        raise ValueError("simulated provider failure")


def test_load_universe_config_reads_initial_universe() -> None:
    config = load_universe_config("configs/universe_etfs_crypto_daily.yaml")

    assert config["base_currency"] == "USD"
    assert "SPY" in config["equity_universe"]["us_core_etfs"]
    assert "BTCUSDT" in config["crypto_universe"]["binance_spot_usdt"]


def test_download_combined_daily_universe_records_success_and_failures() -> None:
    config = load_universe_config("configs/universe_etfs_crypto_daily.yaml")

    response = download_combined_daily_universe(
        config,
        start="2024-01-01",
        end="2024-01-03",
        equity_provider=FakeProvider(MarketType.EQUITY),
        crypto_provider=FakeProvider(MarketType.CRYPTO),
        limit_equity=1,
        limit_crypto=1,
    )

    assert len(response.successful_symbols) == 2
    assert response.failed_symbols
    assert response.metadata["successful_symbols"]
    assert response.metadata["failed_symbols"]
    assert set(response.data["market_type"]) == {"equity", "crypto"}


def test_download_combined_daily_universe_uses_explicit_provider_name(monkeypatch) -> None:  # noqa: ANN001
    config = load_universe_config("configs/universe_etfs_crypto_daily.yaml")

    def fake_factory(name, settings, dataset_code=None):  # noqa: ANN001
        assert name in {"polygon", "binance_public"}
        if name == "polygon":
            return FakeProvider(MarketType.EQUITY)
        return FakeProvider(MarketType.CRYPTO)

    monkeypatch.setattr("quant_platform.data.ingestion.create_market_data_provider", fake_factory)

    response = download_combined_daily_universe(
        config,
        start="2024-01-01",
        end="2024-01-03",
        equity_provider_name="polygon",
        crypto_provider_name="binance_public",
        limit_equity=1,
        limit_crypto=1,
    )

    assert response.metadata["equity"]["selected_provider"] == "polygon"


def test_download_combined_daily_universe_missing_key_fails_clearly() -> None:
    config = load_universe_config("configs/universe_etfs_crypto_daily.yaml")

    with pytest.raises(IngestionError, match="alpha_vantage"):
        download_combined_daily_universe(
            config,
            start="2024-01-01",
            end="2024-01-03",
            equity_provider_name="alpha_vantage",
            crypto_provider=FakeProvider(MarketType.CRYPTO),
            allow_fallback=False,
            limit_equity=1,
            limit_crypto=1,
        )


def test_download_combined_daily_universe_falls_back_to_yfinance(monkeypatch) -> None:  # noqa: ANN001
    config = load_universe_config("configs/universe_etfs_crypto_daily.yaml")

    def fake_factory(name, settings, dataset_code=None):  # noqa: ANN001
        if name == "alpha_vantage":
            return FailingProvider()
        if name == "yfinance":
            return FakeProvider(MarketType.EQUITY)
        return FakeProvider(MarketType.CRYPTO)

    monkeypatch.setattr("quant_platform.data.ingestion.create_market_data_provider", fake_factory)

    response = download_combined_daily_universe(
        config,
        start="2024-01-01",
        end="2024-01-03",
        equity_provider_name="alpha_vantage",
        crypto_provider=FakeProvider(MarketType.CRYPTO),
        fallback_provider_name="yfinance",
        limit_equity=1,
        limit_crypto=1,
    )

    assert response.metadata["equity"]["selected_provider"] == "yfinance"
    assert "fallback_reason" in response.metadata["equity"]


def test_download_combined_daily_universe_without_fallback_raises(monkeypatch) -> None:  # noqa: ANN001
    config = load_universe_config("configs/universe_etfs_crypto_daily.yaml")

    def fake_factory(name, settings, dataset_code=None):  # noqa: ANN001
        if name == "alpha_vantage":
            return FailingProvider()
        return FakeProvider(MarketType.CRYPTO)

    monkeypatch.setattr("quant_platform.data.ingestion.create_market_data_provider", fake_factory)

    with pytest.raises(IngestionError, match="alpha_vantage"):
        download_combined_daily_universe(
            config,
            start="2024-01-01",
            end="2024-01-03",
            equity_provider_name="alpha_vantage",
            crypto_provider=FakeProvider(MarketType.CRYPTO),
            allow_fallback=False,
            limit_equity=1,
            limit_crypto=1,
        )


def test_register_real_dataset_rejects_empty_response(tmp_path) -> None:  # noqa: ANN001
    response = OHLCVResponse(data=pd.DataFrame(), successful_symbols=())

    with pytest.raises(ValueError, match="empty"):
        register_real_dataset(response, tmp_path, "empty", "v1")


def test_register_real_dataset_reports_total_failure(tmp_path) -> None:  # noqa: ANN001
    response = OHLCVResponse(
        data=pd.DataFrame(),
        successful_symbols=(),
        failed_symbols={"SPY": "No rows returned."},
    )

    with pytest.raises(ValueError, match="all symbols failed"):
        register_real_dataset(response, tmp_path, "empty", "v1")


def test_register_real_dataset_writes_coverage_metadata_and_quality_report(tmp_path) -> None:  # noqa: ANN001
    response = OHLCVResponse(
        data=bars("SPY", MarketType.EQUITY, "fake_equity", "FAKE_EQ"),
        successful_symbols=("SPY",),
        failed_symbols={"BAD": "simulated"},
        metadata={"provider": "fake"},
    )

    registered = register_real_dataset(response, tmp_path, "real_daily_demo", "v1")

    manifest = json.loads(registered.manifest_path.read_text(encoding="utf-8"))
    report_path = registered.manifest_path.parent / manifest["quality_report_file"]
    assert report_path.exists()
    assert manifest["coverage_metadata"]["symbols_successful"] == ["SPY"]
    assert manifest["coverage_metadata"]["symbols_failed"] == ["BAD"]
