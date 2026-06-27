from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import OHLCVRequest, OHLCVResponse
from quant_platform.data.schemas import MarketType
from quant_platform.pipelines.real_data_backtest import run_real_data_backtest_pipeline


def bars(symbol: str, market_type: MarketType, source: str, venue: str) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    close = [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]
    return pd.DataFrame(
        {
            "asset_id": symbol,
            "timestamp": timestamp,
            "available_at": timestamp + pd.Timedelta(days=1),
            "open": close,
            "high": [value + 1.0 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000.0] * len(close),
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
            failed_symbols={},
            metadata={"provider": source},
        )


def test_real_data_backtest_pipeline_with_mocked_providers(tmp_path) -> None:  # noqa: ANN001
    summary = run_real_data_backtest_pipeline(
        dataset_id="mock_real_daily",
        version="v1",
        registry_dir=tmp_path / "registry",
        risk_profiles_path="configs/risk_profiles.yaml",
        universe_config_path="configs/universe_etfs_crypto_daily.yaml",
        start="2024-01-01",
        end="2024-01-10",
        equity_provider=FakeProvider(MarketType.EQUITY),
        crypto_provider=FakeProvider(MarketType.CRYPTO),
        report_dir=tmp_path / "reports",
        trial_registry_path=tmp_path / "trials.jsonl",
        limit_equity=1,
        limit_crypto=1,
        download_if_missing=True,
    )

    assert summary["dataset_id"] == "mock_real_daily"
    assert summary["number_of_assets"] == 2
    assert summary["number_of_observations"] == 5
    assert summary["cost_policy"] == "mixed_equity_crypto_conservative_max"
    assert (tmp_path / "trials.jsonl").exists()
    assert summary["report_path"].endswith("mock_real_daily_v1_conservative_report.json")
