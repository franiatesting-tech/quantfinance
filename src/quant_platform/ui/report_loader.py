"""Load local registry manifests and generated reports for the read-only UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def discover_datasets(registry_dir: str | Path = "data/registry") -> list[dict[str, Any]]:
    """Discover local registered datasets without loading market-bar CSV files."""

    root = Path(registry_dir)
    if not root.exists():
        return []
    rows = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = _read_json_object(manifest_path)
        if manifest is None:
            rows.append(_error_row(manifest_path, "invalid_manifest_json"))
            continue
        metadata = manifest.get("metadata", {})
        data_path = manifest_path.parent / str(manifest.get("data_file", "data.csv"))
        quality_file = manifest.get("quality_report_file")
        quality_path = manifest_path.parent / str(quality_file) if quality_file else None
        rows.append(
            {
                "dataset_id": str(metadata.get("dataset_id", manifest_path.parent.parent.name)),
                "version": str(metadata.get("version", manifest_path.parent.name)),
                "source": str(metadata.get("source", "unknown")),
                "market_type": str(metadata.get("market_type", "unknown")),
                "frequency": str(metadata.get("frequency", "unknown")),
                "row_count": int(manifest.get("row_count", 0)),
                "columns": [str(column) for column in manifest.get("columns", [])],
                "manifest_path": str(manifest_path),
                "data_path": str(data_path),
                "data_path_exists": data_path.exists(),
                "quality_report_path": str(quality_path) if quality_path else None,
                "quality_report_exists": bool(quality_path and quality_path.exists()),
                "coverage_metadata": manifest.get("coverage_metadata", {}),
                "registered_at": str(manifest.get("registered_at", "")),
            }
        )
    return rows


def discover_reports(report_dir: str | Path = "reports/generated") -> list[dict[str, Any]]:
    """Discover generated JSON reports without executing any pipeline."""

    root = Path(report_dir)
    if not root.exists():
        return []
    rows = []
    for report_path in sorted(root.rglob("*.json")):
        payload = _read_json_object(report_path)
        if payload is None:
            rows.append(_error_row(report_path, "invalid_report_json"))
            continue
        rows.append(
            {
                "path": str(report_path),
                "name": report_path.name,
                "report_type": classify_report(payload),
                "size_bytes": report_path.stat().st_size,
                "modified_at": _modified_at(report_path),
                "summary": summarize_report_payload(payload),
            }
        )
    return rows


def read_json_report(path: str | Path) -> dict[str, Any]:
    """Read one JSON report and return an object payload."""

    payload = _read_json_object(Path(path))
    if payload is None:
        raise ValueError(f"Invalid JSON report: {path}")
    return payload


def load_trial_registry(
    path: str | Path = "reports/generated/strategy_trials.jsonl",
) -> list[dict[str, Any]]:
    """Load the local JSONL strategy trial registry if it exists."""

    trial_path = Path(path)
    if not trial_path.exists():
        return []
    rows = []
    lines = trial_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"line_number": line_number, "error": "invalid_jsonl"})
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            rows.append({"line_number": line_number, "error": "non_object_jsonl"})
    return rows


def load_dataset_quality_report(dataset_row: dict[str, Any]) -> dict[str, Any] | None:
    """Load the data quality report referenced by a discovered dataset row."""

    quality_path = dataset_row.get("quality_report_path")
    if not quality_path:
        return None
    path = Path(str(quality_path))
    if not path.exists():
        return None
    return _read_json_object(path)


def classify_report(payload: dict[str, Any]) -> str:
    """Classify a generated report payload by stable top-level keys."""

    if "comparison_table" in payload and "profiles" in payload:
        return "profile_comparison"
    if "provider_summaries" in payload:
        return "provider_comparison"
    if payload.get("report_type") == "professional_quant_terminal":
        return "professional_quant_terminal"
    if payload.get("report_type") == "institutional_state_of_art_quant_study":
        return "institutional_study"
    if payload.get("report_type") == "institutional_quant_study_metadata":
        return "institutional_study_metadata"
    if payload.get("report_type") == "final_institutional_quant_finance_paper_metadata":
        return "final_academic_paper"
    if payload.get("report_type") == "final_institutional_quant_package_metadata":
        return "final_institutional_package"
    if payload.get("report_type") == "seven_quant_algorithms_research_study":
        return "seven_algorithm_study"
    if payload.get("report_type") == "seven_quant_algorithms_study_metadata":
        return "seven_algorithm_study_metadata"
    if payload.get("report_type") == "academic_stock_report_metadata":
        return "academic_stock_report"
    if "coverage_by_asset" in payload and "suitable_for_backtest_demo" in payload:
        return "data_quality"
    if "final_equity" in payload and "metadata" in payload:
        return "backtest"
    return "json_report"


def summarize_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact report summary for tables and status cards."""

    report_type = classify_report(payload)
    if report_type == "profile_comparison":
        return {
            "dataset_id": payload.get("dataset_id"),
            "version": payload.get("version"),
            "profiles": sorted(str(key) for key in payload.get("profiles", {})),
            "warnings": len(payload.get("warnings", [])),
        }
    if report_type == "data_quality":
        return {
            "symbols_total": payload.get("symbols_total"),
            "failed_checks": len(payload.get("failed_checks", [])),
            "warnings": len(payload.get("warnings", [])),
            "suitable_for_backtest_demo": payload.get("suitable_for_backtest_demo"),
        }
    if report_type == "backtest":
        metadata = payload.get("metadata", {})
        return {
            "strategy": metadata.get("strategy_name", "equal_weight_real_data_demo")
            if isinstance(metadata, dict)
            else "unknown",
            "final_equity": payload.get("final_equity"),
            "max_drawdown": payload.get("max_drawdown"),
            "sharpe_ratio": payload.get("sharpe_ratio"),
        }
    if report_type == "provider_comparison":
        return {
            "providers_compared": payload.get("providers_compared", []),
            "provider_count": len(payload.get("provider_summaries", [])),
        }
    if report_type == "professional_quant_terminal":
        data = payload.get("data", {})
        universe = payload.get("universe", {})
        return {
            "symbols": universe.get("selected_stocks", []),
            "benchmark": universe.get("benchmark"),
            "data_mode": data.get("mode"),
            "warnings": len(data.get("warnings", [])) if isinstance(data, dict) else 0,
        }
    if report_type == "institutional_study":
        source = payload.get("source_terminal_report", {})
        tables = payload.get("tables", {})
        findings = tables.get("critical_findings", []) if isinstance(tables, dict) else []
        return {
            "data_mode": source.get("data_mode") if isinstance(source, dict) else None,
            "warnings": len(source.get("warnings", [])) if isinstance(source, dict) else 0,
            "critical_findings": len(findings) if isinstance(findings, list) else 0,
            "research_only": payload.get("research_only", True),
        }
    if report_type == "institutional_study_metadata":
        outputs = payload.get("outputs", {})
        return {
            "study_path": outputs.get("json") if isinstance(outputs, dict) else None,
            "markdown_path": outputs.get("markdown") if isinstance(outputs, dict) else None,
            "critical_findings": payload.get("critical_findings_count"),
            "research_only": payload.get("research_only", True),
        }
    if report_type == "final_academic_paper":
        outputs = payload.get("outputs", {})
        return {
            "markdown_path": outputs.get("markdown") if isinstance(outputs, dict) else None,
            "html_path": outputs.get("html") if isinstance(outputs, dict) else None,
            "pdf_path": outputs.get("pdf") if isinstance(outputs, dict) else None,
            "research_only": payload.get("research_only", True),
        }
    if report_type == "final_institutional_package":
        outputs = payload.get("outputs", {})
        return {
            "data_mode": payload.get("data_mode"),
            "warnings": len(payload.get("warnings", [])),
            "research_only": payload.get("research_only", True),
            "has_final_paper": "final_paper" in outputs if isinstance(outputs, dict) else False,
        }
    if report_type == "seven_algorithm_study":
        source = payload.get("source_terminal_report", {})
        tables = payload.get("tables", {})
        decisions = tables.get("broker_readiness_decisions", []) if isinstance(tables, dict) else []
        return {
            "data_mode": source.get("data_mode") if isinstance(source, dict) else None,
            "warnings": len(source.get("warnings", [])) if isinstance(source, dict) else 0,
            "algorithms": len(decisions) if isinstance(decisions, list) else 0,
            "research_only": payload.get("research_only", True),
        }
    if report_type == "seven_algorithm_study_metadata":
        outputs = payload.get("outputs", {})
        return {
            "json_path": outputs.get("json") if isinstance(outputs, dict) else None,
            "markdown_path": outputs.get("markdown") if isinstance(outputs, dict) else None,
            "html_path": outputs.get("html") if isinstance(outputs, dict) else None,
            "pdf_path": outputs.get("pdf") if isinstance(outputs, dict) else None,
            "research_only": payload.get("research_only", True),
        }
    if report_type == "academic_stock_report":
        outputs = payload.get("outputs", {})
        figures = payload.get("figures", {})
        return {
            "asset_id": payload.get("asset_id"),
            "formats": payload.get("formats", []),
            "figure_count": len(figures) if isinstance(figures, dict) else 0,
            "markdown_path": outputs.get("md") if isinstance(outputs, dict) else None,
            "html_path": outputs.get("html") if isinstance(outputs, dict) else None,
            "pdf_path": outputs.get("pdf") if isinstance(outputs, dict) else None,
        }
    return {"top_level_keys": sorted(str(key) for key in payload)[:8]}


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _modified_at(path: Path) -> str:
    return str(path.stat().st_mtime_ns)


def _error_row(path: Path, error: str) -> dict[str, Any]:
    return {"path": str(path), "name": path.name, "error": error}
