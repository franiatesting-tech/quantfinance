from __future__ import annotations

import pandas as pd

from quant_platform.data.ingestion import register_real_dataset
from quant_platform.data.providers.base import OHLCVResponse
from quant_platform.data.schemas import MarketType
from quant_platform.pipelines.profile_comparison import run_profile_comparison_pipeline


def _bars(symbol: str, market_type: MarketType, start_price: float) -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    close = [start_price, start_price + 1, start_price + 2, start_price + 1, start_price + 3,
             start_price + 4, start_price + 3, start_price + 5]
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
            "source": "mock",
            "venue": "MOCK",
        }
    )


def test_profile_comparison_pipeline_with_registered_mock_dataset(tmp_path) -> None:  # noqa: ANN001
    data = pd.concat(
        [
            _bars("SPY", MarketType.EQUITY, 100.0),
            _bars("BTCUSDT", MarketType.CRYPTO, 200.0),
        ],
        ignore_index=True,
    )
    register_real_dataset(
        OHLCVResponse(data=data, successful_symbols=("SPY", "BTCUSDT")),
        tmp_path / "registry",
        "mock_real_daily",
        "v1",
    )

    result = run_profile_comparison_pipeline(
        dataset_id="mock_real_daily",
        version="v1",
        universe_config="configs/universe_etfs_crypto_daily.yaml",
        risk_profiles_config="configs/risk_profiles.yaml",
        registry_dir=tmp_path / "registry",
        output_dir=tmp_path / "reports",
        trial_registry_path=tmp_path / "trials.jsonl",
    )

    assert result["dataset_id"] == "mock_real_daily"
    assert result["profiles"]["conservative"]["number_of_assets"] == 2
    assert len(result["comparison_table"]) == 9
    assert (tmp_path / "reports" / "mock_real_daily_v1_profile_comparison.json").exists()


def test_profile_comparison_dry_run_does_not_require_dataset(tmp_path) -> None:  # noqa: ANN001
    result = run_profile_comparison_pipeline(
        dataset_id="missing",
        version="v1",
        universe_config="configs/universe_etfs_crypto_daily.yaml",
        risk_profiles_config="configs/risk_profiles.yaml",
        registry_dir=tmp_path / "registry",
        output_dir=tmp_path / "reports",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert not (tmp_path / "reports").exists()
