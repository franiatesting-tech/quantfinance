from __future__ import annotations

import json

from quant_platform.ui.report_loader import (
    classify_report,
    discover_datasets,
    discover_reports,
    load_dataset_quality_report,
    load_trial_registry,
    read_json_report,
)


def test_discover_datasets_reads_manifest_without_loading_csv(tmp_path) -> None:  # noqa: ANN001
    dataset_dir = tmp_path / "registry" / "demo" / "v1"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset_id": "demo",
                    "version": "v1",
                    "source": "read_only_real_data",
                    "market_type": "equity",
                    "frequency": "1d",
                    "created_at": "2024-01-01T00:00:00+00:00",
                },
                "data_file": "data.csv",
                "row_count": 10,
                "columns": ["timestamp", "asset_id", "close"],
                "quality_report_file": "data_quality_report.json",
                "coverage_metadata": {
                    "symbols_successful": ["SPY"],
                    "symbols_failed": [],
                    "failed_checks": [],
                    "warnings": [],
                    "suitable_for_backtest_demo": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "data_quality_report.json").write_text(
        json.dumps({"coverage_by_asset": [], "suitable_for_backtest_demo": True}),
        encoding="utf-8",
    )

    rows = discover_datasets(tmp_path / "registry")

    assert rows[0]["dataset_id"] == "demo"
    assert rows[0]["data_path_exists"] is False
    assert rows[0]["quality_report_exists"] is True
    assert load_dataset_quality_report(rows[0]) == {
        "coverage_by_asset": [],
        "suitable_for_backtest_demo": True,
    }


def test_discover_reports_classifies_backtest_and_profile_reports(tmp_path) -> None:  # noqa: ANN001
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    backtest = {"final_equity": 10001.0, "max_drawdown": -0.1, "metadata": {}}
    comparison = {"profiles": {"conservative": {}, "aggressive": {}}, "comparison_table": []}
    academic = {
        "report_type": "academic_stock_report_metadata",
        "asset_id": "AAPL",
        "outputs": {"md": "AAPL.md", "html": "AAPL.html"},
        "figures": {"price_history": "price_history.html"},
    }
    final_package = {
        "report_type": "final_institutional_quant_package_metadata",
        "outputs": {"final_paper": {"pdf": "paper.pdf"}},
        "warnings": [],
    }
    final_paper = {
        "report_type": "final_institutional_quant_finance_paper_metadata",
        "outputs": {"markdown": "paper.md", "html": "paper.html", "pdf": "paper.pdf"},
    }
    study_metadata = {
        "report_type": "institutional_quant_study_metadata",
        "outputs": {"json": "institutional_quant_study.json"},
        "critical_findings_count": 2,
    }
    (report_dir / "backtest.json").write_text(json.dumps(backtest), encoding="utf-8")
    (report_dir / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    (report_dir / "academic.json").write_text(json.dumps(academic), encoding="utf-8")
    (report_dir / "final_package.json").write_text(json.dumps(final_package), encoding="utf-8")
    (report_dir / "final_paper.json").write_text(json.dumps(final_paper), encoding="utf-8")
    (report_dir / "study_metadata.json").write_text(json.dumps(study_metadata), encoding="utf-8")

    rows = discover_reports(report_dir)

    assert {row["report_type"] for row in rows} == {
        "academic_stock_report",
        "backtest",
        "final_academic_paper",
        "final_institutional_package",
        "institutional_study_metadata",
        "profile_comparison",
    }
    assert classify_report(read_json_report(report_dir / "backtest.json")) == "backtest"
    assert (
        classify_report(read_json_report(report_dir / "academic.json")) == "academic_stock_report"
    )
    assert (
        classify_report(read_json_report(report_dir / "final_package.json"))
        == "final_institutional_package"
    )


def test_load_trial_registry_handles_invalid_lines(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "strategy_trials.jsonl"
    path.write_text('{"trial_id": "a"}\nnot-json\n', encoding="utf-8")

    rows = load_trial_registry(path)

    assert rows[0]["trial_id"] == "a"
    assert rows[1]["error"] == "invalid_jsonl"
