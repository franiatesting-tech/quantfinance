from __future__ import annotations

import pytest

from quant_platform.data.providers.alpha_vantage_provider import AlphaVantageDailyProvider
from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.schemas import MarketType


class FakeHttpClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def get_json(self, url, params=None, headers=None):  # noqa: ANN001
        assert "secret" in params["apikey"]
        return self.payload


def _request() -> OHLCVRequest:
    return OHLCVRequest(
        symbols=("SPY",),
        start="2024-01-01",
        end="2024-01-03",
        frequency="1d",
        market_type=MarketType.EQUITY,
        source="alpha_vantage",
        currency="USD",
    )


def _payload() -> dict[str, object]:
    return {
        "Time Series (Daily)": {
            "2024-01-02": {
                "1. open": "100",
                "2. high": "102",
                "3. low": "99",
                "4. close": "101",
                "5. adjusted close": "101",
                "6. volume": "1000",
            }
        }
    }


def test_alpha_vantage_valid_response_schema() -> None:
    provider = AlphaVantageDailyProvider("secret", http_client=FakeHttpClient(_payload()))

    response = provider.download_ohlcv(_request())

    assert response.successful_symbols == ("SPY",)
    assert response.data.iloc[0]["source"] == "alpha_vantage"
    assert response.data.iloc[0]["venue"] == "ALPHA_VANTAGE"


def test_alpha_vantage_rate_limit_note_records_failure() -> None:
    provider = AlphaVantageDailyProvider("secret", http_client=FakeHttpClient({"Note": "limit"}))

    response = provider.download_ohlcv(_request())

    assert response.data.empty
    assert "SPY" in response.failed_symbols
    assert "secret" not in str(response.failed_symbols)


def test_alpha_vantage_error_message_records_failure() -> None:
    provider = AlphaVantageDailyProvider(
        "secret",
        http_client=FakeHttpClient({"Error Message": "bad symbol"}),
    )

    response = provider.download_ohlcv(_request())

    assert response.data.empty
    assert "SPY" in response.failed_symbols


def test_alpha_vantage_missing_api_key_fails() -> None:
    with pytest.raises(ValueError, match="API key"):
        AlphaVantageDailyProvider("")
