from __future__ import annotations

from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.providers.binance_public_provider import (
    BinancePublicSpotProvider,
    normalize_binance_klines,
)
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


def kline_payload() -> list[list[object]]:
    return [
        [
            1704067200000,
            "100.0",
            "110.0",
            "90.0",
            "105.0",
            "1234.0",
            1704153599999,
            "0",
            1,
            "0",
            "0",
            "0",
        ]
    ]


def request(symbols: tuple[str, ...]) -> OHLCVRequest:
    return OHLCVRequest(
        symbols=symbols,
        start="2024-01-01",
        end="2024-01-03",
        frequency="1d",
        market_type=MarketType.CRYPTO,
        source="binance_public",
        currency="USDT",
    )


class FakeResponse:
    def __init__(self, payload, fail: bool = False) -> None:  # noqa: ANN001
        self.payload = payload
        self.fail = fail

    def raise_for_status(self) -> None:
        if self.fail:
            raise RuntimeError("HTTP error")

    def json(self):  # noqa: ANN201
        return self.payload


def test_normalize_binance_klines_outputs_canonical_schema() -> None:
    bars = normalize_binance_klines(kline_payload(), "BTCUSDT")

    assert bars["asset_id"].tolist() == ["BTCUSDT"]
    assert bars["source"].tolist() == ["binance_public"]
    assert bars["venue"].tolist() == ["BINANCE_SPOT"]
    assert bars["available_at"].iloc[0] >= bars["timestamp"].iloc[0]
    assert run_market_data_quality_checks(bars) is bars


def test_binance_provider_downloads_without_api_key(monkeypatch) -> None:  # noqa: ANN001
    calls = []

    def fake_get(url, params, timeout):  # noqa: ANN001
        calls.append((url, params, timeout))
        return FakeResponse(kline_payload())

    monkeypatch.setattr(
        "quant_platform.data.providers.binance_public_provider.requests.get",
        fake_get,
    )

    response = BinancePublicSpotProvider().download_ohlcv(request(("BTCUSDT",)))

    assert response.successful_symbols == ("BTCUSDT",)
    assert not response.failed_symbols
    assert calls[0][1]["symbol"] == "BTCUSDT"


def test_binance_provider_records_failed_symbol(monkeypatch) -> None:  # noqa: ANN001
    def fake_get(url, params, timeout):  # noqa: ANN001, ARG001
        if params["symbol"] == "BADUSDT":
            return FakeResponse([], fail=True)
        return FakeResponse(kline_payload())

    monkeypatch.setattr(
        "quant_platform.data.providers.binance_public_provider.requests.get",
        fake_get,
    )

    response = BinancePublicSpotProvider().download_ohlcv(request(("BTCUSDT", "BADUSDT")))

    assert response.successful_symbols == ("BTCUSDT",)
    assert "BADUSDT" in response.failed_symbols
