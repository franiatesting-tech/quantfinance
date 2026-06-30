from __future__ import annotations

import pytest

from quant_platform.research.asset_dataset import load_quant_terminal_config
from quant_platform.research.institutional_study import (
    InstitutionalStudyError,
    build_institutional_study,
    write_institutional_study,
)
from quant_platform.research.report import build_quant_terminal_report


def test_institutional_study_rejects_synthetic_data_in_strict_mode() -> None:
    config = load_quant_terminal_config("configs/quant_terminal_3_stocks.yaml")
    report = build_quant_terminal_report(config, offline_synthetic=True)

    with pytest.raises(InstitutionalStudyError, match="requires provider data"):
        build_institutional_study(report, strict_real_data=True)


def test_institutional_study_writes_tables_figures_and_markdown(tmp_path) -> None:  # noqa: ANN001
    config = load_quant_terminal_config("configs/quant_terminal_3_stocks.yaml")
    report = build_quant_terminal_report(config, offline_synthetic=True)

    metadata = write_institutional_study(
        report,
        tmp_path,
        max_table_rows=7,
        strict_real_data=False,
    )

    outputs = metadata["outputs"]
    assert metadata["research_only"] is True
    assert (tmp_path / "institutional_quant_study.json").exists()
    assert (tmp_path / "institutional_quant_study.md").exists()
    assert (tmp_path / "institutional_quant_finance_paper.md").exists()
    assert len(outputs["tables"]) >= 30
    assert len(outputs["figures"]) >= 16
    assert (tmp_path / "tables" / "02_price_sample_first_rows.csv").exists()
    assert (tmp_path / "figures" / "03_correlation_heatmap.html").exists()
    assert (tmp_path / "figures" / "16_final_decision_waterfall.html").exists()

    study = build_institutional_study(report, max_table_rows=7, strict_real_data=False)
    assert len(study["tables"]["price_sample_first_rows"]) == 7
    assert study["tables"]["portfolio_tail_risk"]
    assert study["tables"]["tail_risk_backtesting"]
    assert study["tables"]["multiple_testing_adjustments"]
    assert study["tables"]["model_confidence"]
    assert study["tables"]["portfolio_diagnostics"]
    assert study["tables"]["shrinkage_sensitivity"]
    assert study["tables"]["covariance_shrinkage_comparison"]
    assert study["tables"]["portfolio_robustness"]
    assert study["tables"]["factor_model_gap_table"]
    assert study["tables"]["execution_cost_sensitivity"]
    assert study["tables"]["final_research_decision_table"]
    assert study["tables"]["formula_catalog"]
    findings = study["tables"]["critical_findings"]
    assert any(item["finding"] == "Synthetic data mode detected" for item in findings)
    broker_gates = study["tables"]["broker_research_gates"]
    assert all(item["live_order_allowed"] is False for item in broker_gates)
