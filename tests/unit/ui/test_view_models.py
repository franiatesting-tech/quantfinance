from __future__ import annotations

import json

from quant_platform.config.settings import load_settings_from_env
from quant_platform.ui.view_models import (
    backtest_metric_rows,
    build_platform_snapshot,
    build_provider_rows,
    conceptual_flow_steps,
    dataset_quality_rows,
    load_risk_profile_summary,
    load_universe_summary,
    report_series,
    summarize_providers,
    transaction_cost_components,
)


def test_build_provider_rows_reduces_credentials_to_state() -> None:
    settings = load_settings_from_env(
        {
            "POLYGON_API_KEY": "secret-value",
            "POLYGON_ENABLED": "false",
        }
    )

    rows = build_provider_rows(settings)
    payload = json.dumps(rows)
    polygon = next(row for row in rows if row["name"] == "polygon")

    assert polygon["credential_state"] == "configured"
    assert polygon["status"] == "configured_but_disabled"
    assert "secret-value" not in payload
    assert summarize_providers(rows)["configured_but_disabled"] == 1


def test_load_universe_and_risk_profile_summaries() -> None:
    universe = load_universe_summary("configs/universe_etfs_crypto_daily.yaml")
    risk = load_risk_profile_summary("configs/risk_profiles.yaml")

    assert universe["total_symbols"] == 44
    assert len(risk["profiles"]) == 2
    assert {profile["profile"] for profile in risk["profiles"]} == {"conservative", "aggressive"}


def test_build_platform_snapshot_is_local_and_safe(tmp_path) -> None:  # noqa: ANN001
    settings = load_settings_from_env({"ALPHA_VANTAGE_API_KEY": "secret-value"})
    snapshot = build_platform_snapshot(
        settings=settings,
        registry_dir=tmp_path / "registry",
        report_dir=tmp_path / "reports",
    )

    payload = json.dumps(snapshot)
    assert snapshot["datasets"] == []
    assert snapshot["reports"] == []
    assert snapshot["conceptual_flow"]
    assert "secret-value" not in payload
    assert "alpha_vantage_api_key" not in payload


def test_dataset_quality_rows_flattens_coverage_metadata() -> None:
    rows = dataset_quality_rows(
        [
            {
                "dataset_id": "demo",
                "version": "v1",
                "row_count": 5,
                "quality_report_exists": True,
                "coverage_metadata": {
                    "symbols_successful": ["SPY"],
                    "symbols_failed": ["QQQ"],
                    "failed_checks": ["duplicate_rows"],
                    "warnings": ["calendar_gap"],
                    "suitable_for_backtest_demo": False,
                },
            }
        ]
    )

    assert rows[0]["symbols_total"] == 2
    assert rows[0]["failed_checks"] == 1
    assert rows[0]["warnings"] == 1


def test_backtest_metric_rows_extracts_numeric_metrics() -> None:
    rows = backtest_metric_rows({"final_equity": "10001.5", "sharpe_ratio": "bad"})
    values = {row["metric"]: row["value"] for row in rows}

    assert values["final_equity"] == 10001.5
    assert values["sharpe_ratio"] is None


def test_conceptual_flow_and_report_helpers() -> None:
    flow = conceptual_flow_steps()
    report = {
        "equity_curve": [{"timestamp": "2024-01-01", "equity": 10000}],
        "total_transaction_cost": "12.5",
    }

    assert flow[0]["block"] == "Providers"
    assert report_series(report, "equity_curve") == report["equity_curve"]
    assert transaction_cost_components(report) == {"total_transaction_cost": 12.5}
