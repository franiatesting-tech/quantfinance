from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.providers.yfinance_provider import (
    YFinanceDailyProvider,
    normalize_yfinance_frame,
)
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


def raw_yfinance_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 102.0],
            "High": [103.0, 104.0],
            "Low": [99.0, 101.0],
            "Close": [102.0, 103.0],
            "Adj Close": [101.0, 102.0],
            "Volume": [1000.0, 1100.0],
        },
        index=index,
    )


def request(symbols: tuple[str, ...]) -> OHLCVRequest:
    return OHLCVRequest(
        symbols=symbols,
        start="2024-01-01",
        end="2024-01-03",
        frequency="1d",
        market_type=MarketType.EQUITY,
        source="yfinance",
        currency="USD",
    )


def test_normalize_yfinance_frame_outputs_canonical_schema() -> None:
    bars = normalize_yfinance_frame(raw_yfinance_frame(), "SPY")

    assert bars["asset_id"].unique().tolist() == ["SPY"]
    assert bars["source"].unique().tolist() == ["yfinance"]
    assert bars["venue"].unique().tolist() == ["YAHOO_PUBLIC"]
    assert bars["close"].tolist() == [101.0, 102.0]
    assert run_market_data_quality_checks(bars) is bars


def test_yfinance_provider_records_failed_symbol(monkeypatch) -> None:  # noqa: ANN001
    def fake_download(symbol, **kwargs):  # noqa: ANN001, ARG001
        if symbol == "BAD":
            return pd.DataFrame()
        return raw_yfinance_frame()

    monkeypatch.setattr(
        "quant_platform.data.providers.yfinance_provider.yf.download",
        fake_download,
    )

    response = YFinanceDailyProvider().download_ohlcv(request(("SPY", "BAD")))

    assert response.successful_symbols == ("SPY",)
    assert "BAD" in response.failed_symbols
    assert not response.data.empty


def test_yfinance_provider_accepts_index_symbol(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "quant_platform.data.providers.yfinance_provider.yf.download",
        lambda symbol, **kwargs: raw_yfinance_frame(),  # noqa: ARG005
    )

    response = YFinanceDailyProvider().download_ohlcv(request(("^GSPC",)))

    assert response.successful_symbols == ("^GSPC",)
    assert response.data["asset_id"].unique().tolist() == ["^GSPC"]


def test_normalize_yfinance_multiindex_frame() -> None:
    raw = raw_yfinance_frame()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["SPY"]])

    bars = normalize_yfinance_frame(raw, "SPY")

    assert len(bars) == 2
    assert run_market_data_quality_checks(bars) is bars
