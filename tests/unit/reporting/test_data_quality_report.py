from __future__ import annotations

import json

import pandas as pd

from quant_platform.reporting.data_quality_report import (
    build_data_quality_report,
    write_data_quality_report,
)


def _bars() -> pd.DataFrame:
    timestamp = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "asset_id": "SPY",
            "timestamp": timestamp,
            "available_at": timestamp + pd.Timedelta(days=1),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
            "market_type": "equity",
            "currency": "USD",
            "source": "unit_test",
            "venue": "TEST",
        }
    )


def test_build_data_quality_report_includes_provider_metadata() -> None:
    report = build_data_quality_report(
        _bars(),
        {"successful_symbols": ("SPY",), "failed_symbols": {"BAD": "No rows"}},
    )

    assert report["symbols_total"] == 2
    assert report["symbols_successful"] == ["SPY"]
    assert report["symbols_failed"] == ["BAD"]
    assert "partial_symbol_failures" in report["warnings"]
    assert report["suitable_for_backtest_demo"] is True


def test_write_data_quality_report_writes_json(tmp_path) -> None:  # noqa: ANN001
    report = build_data_quality_report(_bars())
    path = write_data_quality_report(report, tmp_path / "quality" / "report.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["symbols_successful"] == ["SPY"]
    assert payload["coverage_by_asset"][0]["asset_id"] == "SPY"
