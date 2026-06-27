from __future__ import annotations

import pytest

from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.providers.polygon_provider import PolygonDailyProvider
from quant_platform.data.schemas import MarketType


class FakeHttpClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def get_json(self, url, params=None, headers=None):  # noqa: ANN001
        assert params["apiKey"] == "secret"
        return self.payload


def _request() -> OHLCVRequest:
    return OHLCVRequest(
        ("SPY",),
        "2024-01-01",
        "2024-01-03",
        "1d",
        MarketType.EQUITY,
        "polygon",
        "USD",
    )


def test_polygon_valid_response_schema() -> None:
    payload = {
        "status": "OK",
        "results": [
            {"t": 1704153600000, "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000}
        ],
    }
    response = PolygonDailyProvider("secret", FakeHttpClient(payload)).download_ohlcv(_request())

    assert response.successful_symbols == ("SPY",)
    assert response.data.iloc[0]["source"] == "polygon"


def test_polygon_no_results_records_failure() -> None:
    response = PolygonDailyProvider(
        "secret",
        FakeHttpClient({"status": "OK", "results": []}),
    ).download_ohlcv(_request())

    assert response.data.empty
    assert "SPY" in response.failed_symbols


def test_polygon_api_error_records_failure() -> None:
    response = PolygonDailyProvider(
        "secret",
        FakeHttpClient({"status": "ERROR"}),
    ).download_ohlcv(_request())

    assert response.data.empty
    assert "SPY" in response.failed_symbols


def test_polygon_missing_api_key_fails() -> None:
    with pytest.raises(ValueError, match="API key"):
        PolygonDailyProvider("")
