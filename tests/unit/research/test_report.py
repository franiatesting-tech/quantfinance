from __future__ import annotations

import json

from quant_platform.config.settings import load_settings_from_env
from quant_platform.research.asset_dataset import load_quant_terminal_config
from quant_platform.research.report import (
    build_and_write_quant_terminal_report,
    build_quant_terminal_report,
)
from quant_platform.ui.report_loader import classify_report, summarize_report_payload


def test_build_quant_terminal_report_offline_synthetic_is_research_only() -> None:
    config = load_quant_terminal_config("configs/quant_terminal_3_stocks.yaml")

    report = build_quant_terminal_report(config, offline_synthetic=True)
    payload = json.dumps(report)

    assert report["report_type"] == "professional_quant_terminal"
    assert report["research_only"] is True
    assert report["data"]["mode"] == "offline_synthetic"
    assert "metadata" in report
    assert "data_provenance" in report
    assert "stocks" in report
    assert report["universe"]["selected_stocks"] == ["AAPL", "MSFT", "NVDA"]
    assert set(report["stocks"]) == {"AAPL", "MSFT", "NVDA"}
    assert report["stocks"]["AAPL"]["data_used"]["ticker"] == "AAPL"
    assert report["stocks"]["AAPL"]["price_series"]
    assert report["stocks"]["AAPL"]["monte_carlo"]["parametric_normal"]["paths_sample"]
    stock = report["stocks"]["AAPL"]
    assert stock["benchmark_relative_study"]["status"] == (
        "IMPLEMENTED_WITH_DAILY_BENCHMARK_PROXY"
    )
    assert stock["momentum_liquidity_study"]["status"] == (
        "DAILY_OHLCV_PROXY_NOT_FACTOR_ZOO_REPLICATION"
    )
    assert stock["range_volatility_study"]["parkinson_volatility_annual"] >= 0
    assert stock["sharpe_inference_study"]["status"] == (
        "IMPLEMENTED_AS_DAILY_RETURN_INFERENCE_APPROXIMATION"
    )
    assert stock["var"]["historical"]["expected_shortfall"] >= stock["var"]["historical"]["var"]
    assert stock["tail_risk_backtesting_study"]["loss_sign_convention"].startswith("L_t = -R_t")
    assert stock["tail_risk_backtesting_study"]["levels"]["alpha_95"]["exceptions"] >= 0
    assert stock["execution_cost_study"]["status"] == (
        "HYPOTHETICAL_DAILY_ADV_SCENARIO_NOT_REAL_EXECUTION_MODEL"
    )
    unsupported = {
        item["research_family"]: item["what_is_not_supported"]
        for item in stock["literature_implementation_map"]
    }
    assert "Large cross-section SDF/IPCA" in unsupported["Machine-learning asset pricing"]
    assert "Triple-barrier" in unsupported["Meta-labeling and event bars"]
    assert report["options"]["AAPL"]["model_status"] == "PARAMETRIC_EDUCATIONAL_MODEL"
    assert "DEMO_SYNTHETIC_NOT_REAL_DATA" in report["warnings"]
    assert "api_key" not in payload.lower()
    assert "secret=" not in payload.lower()


def test_build_and_write_quant_terminal_report_outputs_json_and_frontier_csv(tmp_path) -> None:
    summary = build_and_write_quant_terminal_report(
        config_path="configs/quant_terminal_3_stocks.yaml",
        settings=load_settings_from_env(),
        output_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
        offline_synthetic=True,
    )

    assert summary["data_mode"] == "offline_synthetic"
    assert summary["research_only"] is True
    assert (tmp_path / "reports" / "3stocks_10y_report.json").exists()
    assert (tmp_path / "exports" / "3stocks_10y_frontier.csv").exists()


def test_report_loader_classifies_professional_quant_terminal() -> None:
    payload = {
        "report_type": "professional_quant_terminal",
        "universe": {"selected_stocks": ["AAPL", "MSFT", "NVDA"], "benchmark": "SPY"},
        "data": {"mode": "offline_synthetic", "warnings": ["warning"]},
    }

    assert classify_report(payload) == "professional_quant_terminal"
    assert summarize_report_payload(payload)["data_mode"] == "offline_synthetic"
