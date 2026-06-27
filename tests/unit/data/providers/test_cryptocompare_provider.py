from __future__ import annotations

import pytest

from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.providers.cryptocompare_provider import CryptoCompareDailyProvider
from quant_platform.data.schemas import MarketType


class FakeHttpClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def get_json(self, url, params=None, headers=None):  # noqa: ANN001
        assert params["api_key"] == "secret"
        return self.payload


def _request(symbol: str = "BTC/USD") -> OHLCVRequest:
    return OHLCVRequest(
        (symbol,),
        "2024-01-01",
        "2024-01-03",
        "1d",
        MarketType.CRYPTO,
        "cryptocompare",
        "USD",
    )


def _payload() -> dict[str, object]:
    return {
        "Response": "Success",
        "Data": {
            "Data": [
                {
                    "time": 1704153600,
                    "open": 100,
                    "high": 102,
                    "low": 99,
                    "close": 101,
                    "volumefrom": 10,
                }
            ]
        },
    }


def test_cryptocompare_valid_response_schema() -> None:
    response = CryptoCompareDailyProvider("secret", FakeHttpClient(_payload())).download_ohlcv(
        _request()
    )

    assert response.successful_symbols == ("BTC/USD",)
    assert response.data.iloc[0]["source"] == "cryptocompare"
    assert response.data.iloc[0]["currency"] == "USD"


def test_cryptocompare_invalid_pair_records_failure() -> None:
    response = CryptoCompareDailyProvider("secret", FakeHttpClient(_payload())).download_ohlcv(
        _request("BAD")
    )

    assert response.data.empty
    assert "BAD" in response.failed_symbols


def test_cryptocompare_api_error_records_failure() -> None:
    response = CryptoCompareDailyProvider(
        "secret",
        FakeHttpClient({"Response": "Error"}),
    ).download_ohlcv(_request())

    assert response.data.empty
    assert "BTC/USD" in response.failed_symbols


def test_cryptocompare_missing_key_fails() -> None:
    with pytest.raises(ValueError, match="API key"):
        CryptoCompareDailyProvider("")
