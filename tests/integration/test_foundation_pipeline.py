from __future__ import annotations

from datetime import UTC, datetime

from quant_platform.data.registry import load_dataset, register_dataset
from quant_platform.data.schemas import AssetMetadata, DatasetMetadata, Frequency, MarketType
from quant_platform.data.synthetic import generate_synthetic_market_bars
from quant_platform.reporting.foundation_report import (
    build_foundation_report,
    report_has_finite_core_metrics,
)


def test_synthetic_registry_to_foundation_report_pipeline(tmp_path) -> None:  # noqa: ANN001
    assets = [
        AssetMetadata("SYN_EQ_001", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
        AssetMetadata("SYN_EQ_002", MarketType.EQUITY, "USD", "synthetic", venue="SIM"),
    ]
    bars = generate_synthetic_market_bars(assets, "2024-01-01", periods=40, seed=11)
    metadata = DatasetMetadata(
        dataset_id="synthetic_foundation",
        version="v1",
        source="synthetic",
        market_type=MarketType.EQUITY,
        frequency=Frequency.DAILY,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    register_dataset(bars, metadata, tmp_path)
    loaded, manifest = load_dataset(tmp_path, "synthetic_foundation", "v1")

    report = build_foundation_report(
        loaded,
        periods_per_year=252,
        alpha=0.95,
        dataset_id=manifest.metadata.dataset_id,
        dataset_version=manifest.metadata.version,
    )

    assert report.dataset_id == "synthetic_foundation"
    assert report.dataset_version == "v1"
    assert report.row_count == len(bars)
    assert report.asset_count == 2
    assert report.observation_count == 39
    assert report.historical_expected_shortfall >= report.historical_var
    assert report_has_finite_core_metrics(report)
