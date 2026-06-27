from __future__ import annotations

import pytest

from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.providers.nasdaq_data_link_provider import NasdaqDataLinkProvider
from quant_platform.data.schemas import MarketType


class FakeHttpClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def get_json(self, url, params=None, headers=None):  # noqa: ANN001
        assert params["api_key"] == "secret"
        return self.payload


def _request() -> OHLCVRequest:
    return OHLCVRequest(
        ("SPY",),
        "2024-01-01",
        "2024-01-03",
        "1d",
        MarketType.EQUITY,
        "nasdaq_data_link",
        "USD",
    )


def _payload() -> dict[str, object]:
    return {
        "dataset": {
            "column_names": ["Date", "Open", "High", "Low", "Close", "Volume"],
            "data": [["2024-01-02", 100, 102, 99, 101, 1000]],
        }
    }


def test_nasdaq_data_link_valid_response_schema() -> None:
    provider = NasdaqDataLinkProvider(
        "secret",
        dataset_code="TEST/{symbol}",
        http_client=FakeHttpClient(_payload()),
    )

    response = provider.download_ohlcv(_request())

    assert response.successful_symbols == ("SPY",)
    assert response.data.iloc[0]["source"] == "nasdaq_data_link"


def test_nasdaq_data_link_missing_key_fails() -> None:
    with pytest.raises(ValueError, match="API key"):
        NasdaqDataLinkProvider("")


def test_nasdaq_data_link_missing_dataset_code_fails() -> None:
    provider = NasdaqDataLinkProvider("secret", http_client=FakeHttpClient(_payload()))

    with pytest.raises(ValueError, match="dataset_code"):
        provider.download_ohlcv(_request())


def test_nasdaq_data_link_api_error_records_failure() -> None:
    provider = NasdaqDataLinkProvider(
        "secret",
        dataset_code="TEST/{symbol}",
        http_client=FakeHttpClient({"quandl_error": {"message": "bad"}}),
    )

    response = provider.download_ohlcv(_request())
    assert response.data.empty
    assert "SPY" in response.failed_symbols
    assert "secret" not in str(response.failed_symbols)
