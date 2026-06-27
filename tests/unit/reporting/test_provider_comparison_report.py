from __future__ import annotations

import json

from quant_platform.reporting.provider_comparison_report import (
    compare_provider_coverage,
    write_provider_comparison_report,
)


def _report(provider: str, symbol: str) -> dict[str, object]:
    return {
        "symbols_successful": [symbol],
        "symbols_failed": ["BAD"],
        "date_range": {"start_timestamp": "2024-01-01", "end_timestamp": "2024-01-02"},
        "coverage_by_asset": [
            {
                "asset_id": symbol,
                "row_count": 2,
                "missing_required_values": 0,
                "duplicate_rows": 0,
                "zero_volume_rows": 0,
                "available_at_violations": 0,
                "min_close": 100.0,
                "max_close": 101.0,
            }
        ],
        "date_gaps": [],
        "warnings": [],
        "failed_checks": [],
        "provider_metadata": {"provider": provider},
    }


def test_compare_provider_coverage_summarizes_reports() -> None:
    comparison = compare_provider_coverage([_report("polygon", "SPY"), _report("alpha", "QQQ")])

    assert comparison["providers_compared"] == ["polygon", "alpha"]
    assert comparison["symbols_successful_union"] == ["QQQ", "SPY"]
    assert comparison["provider_summaries"][0]["row_count"] == 2


def test_write_provider_comparison_report(tmp_path) -> None:  # noqa: ANN001
    path = write_provider_comparison_report(
        compare_provider_coverage([_report("polygon", "SPY")]),
        tmp_path / "providers.json",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["providers_compared"] == ["polygon"]
