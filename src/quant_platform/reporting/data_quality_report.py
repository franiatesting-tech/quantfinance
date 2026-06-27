"""JSON data quality reports for read-only research datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.data.diagnostics import build_data_quality_summary


def build_data_quality_report(
    df: pd.DataFrame,
    provider_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable data quality report for a normalized OHLCV dataset."""

    metadata = provider_metadata or {}
    summary = build_data_quality_summary(df)
    failed_symbols = _failed_symbols(metadata)
    successful_symbols = _successful_symbols(metadata, summary["assets"])
    date_range = _date_range(df)
    warnings = list(summary["warnings"])
    if failed_symbols:
        warnings.append("partial_symbol_failures")
    suitable = bool(
        summary["total_rows"] > 0
        and successful_symbols
        and not summary["failed_checks"]
        and date_range["start_timestamp"] is not None
        and date_range["end_timestamp"] is not None
    )
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "symbols_total": int(len(set(successful_symbols) | set(failed_symbols))),
        "symbols_successful": successful_symbols,
        "symbols_failed": failed_symbols,
        "date_range": date_range,
        "coverage_by_asset": summary["coverage_table"],
        "date_gaps": summary["date_gap_table"],
        "failed_checks": summary["failed_checks"],
        "warnings": sorted(set(warnings)),
        "assumptions": [
            "Equity daily gaps require calendar-aware review and are warnings in Iteration 006.",
            "Crypto daily spot data is expected to be continuous on a 24/7 calendar.",
            "Free public providers can revise data and may have licensing or availability limits.",
            "Risk profile drawdown targets are evaluation thresholds, not guarantees.",
        ],
        "provider_metadata": metadata,
        "suitable_for_backtest_demo": suitable,
    }


def write_data_quality_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON data quality report to a local ignored path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _failed_symbols(metadata: dict[str, Any]) -> list[str]:
    failed = metadata.get("failed_symbols", {})
    if isinstance(failed, dict):
        return sorted(str(symbol) for symbol in failed)
    if isinstance(failed, list | tuple | set):
        return sorted(str(symbol) for symbol in failed)
    return []


def _successful_symbols(metadata: dict[str, Any], fallback_assets: list[str]) -> list[str]:
    successful = metadata.get("successful_symbols")
    if isinstance(successful, list | tuple | set):
        return sorted(str(symbol) for symbol in successful)
    return sorted(str(asset) for asset in fallback_assets)


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty or "timestamp" not in df.columns:
        return {"start_timestamp": None, "end_timestamp": None}
    timestamp = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    if timestamp.empty:
        return {"start_timestamp": None, "end_timestamp": None}
    return {
        "start_timestamp": timestamp.min().isoformat(),
        "end_timestamp": timestamp.max().isoformat(),
    }
