"""Compare coverage and quality reports across read-only data providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_provider_coverage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare provider coverage summaries from data quality reports."""

    provider_rows = []
    all_successful: set[str] = set()
    all_failed: set[str] = set()
    for report in reports:
        provider_metadata = report.get("provider_metadata", {})
        provider_name = _provider_name(provider_metadata)
        successful = set(str(symbol) for symbol in report.get("symbols_successful", []))
        failed = set(str(symbol) for symbol in report.get("symbols_failed", []))
        all_successful.update(successful)
        all_failed.update(failed)
        coverage = report.get("coverage_by_asset", [])
        provider_rows.append(
            {
                "provider": provider_name,
                "symbols_successful": sorted(successful),
                "symbols_failed": sorted(failed),
                "row_count": int(sum(int(row.get("row_count", 0)) for row in coverage)),
                "date_range": report.get("date_range", {}),
                "missing_required_values": int(
                    sum(int(row.get("missing_required_values", 0)) for row in coverage)
                ),
                "duplicate_rows": int(sum(int(row.get("duplicate_rows", 0)) for row in coverage)),
                "zero_volume_rows": int(
                    sum(int(row.get("zero_volume_rows", 0)) for row in coverage)
                ),
                "available_at_violations": int(
                    sum(int(row.get("available_at_violations", 0)) for row in coverage)
                ),
                "min_close": _safe_min(row.get("min_close") for row in coverage),
                "max_close": _safe_max(row.get("max_close") for row in coverage),
                "warnings": sorted(str(item) for item in report.get("warnings", [])),
                "failed_checks": sorted(str(item) for item in report.get("failed_checks", [])),
                "date_gap_count": len(report.get("date_gaps", [])),
            }
        )
    return {
        "providers_compared": [row["provider"] for row in provider_rows],
        "symbols_successful_union": sorted(all_successful),
        "symbols_failed_union": sorted(all_failed),
        "provider_summaries": provider_rows,
        "assumptions": [
            "Comparison uses quality-report metadata, not tick-by-tick price reconciliation.",
            "Provider coverage can differ because of symbols, plan limits, calendars, "
            "and adjustments.",
        ],
    }


def write_provider_comparison_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write provider comparison JSON to a local ignored report path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _provider_name(provider_metadata: object) -> str:
    if not isinstance(provider_metadata, dict):
        return "unknown"
    provider = provider_metadata.get("provider") or provider_metadata.get("selected_provider")
    if isinstance(provider, str):
        return provider
    nested_names = []
    for value in provider_metadata.values():
        if isinstance(value, dict) and isinstance(value.get("provider"), str):
            nested_names.append(value["provider"])
    return "+".join(sorted(nested_names)) if nested_names else "unknown"


def _safe_min(values) -> float | None:  # noqa: ANN001
    clean = [float(value) for value in values if value is not None]
    return min(clean) if clean else None


def _safe_max(values) -> float | None:  # noqa: ANN001
    clean = [float(value) for value in values if value is not None]
    return max(clean) if clean else None
