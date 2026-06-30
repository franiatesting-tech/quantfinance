from __future__ import annotations

import json

from quant_platform.research import final_academic_paper
from quant_platform.research.final_academic_paper import generate_final_academic_paper


def test_generate_final_academic_paper_writes_md_html_metadata_and_manifest(
    monkeypatch, tmp_path
) -> None:  # noqa: ANN001
    source_dir = tmp_path / "institutional"
    source_dir.mkdir()
    figures_dir = source_dir / "figures"
    figures_dir.mkdir()
    (figures_dir / "01_normalized_prices.html").write_text(
        "<html><body><h1>Figure</h1><svg></svg></body></html>",
        encoding="utf-8",
    )
    study_path = source_dir / "institutional_quant_study.json"
    study_path.write_text(json.dumps(_sample_study()), encoding="utf-8")

    monkeypatch.setattr(
        final_academic_paper,
        "export_html_report_to_pdf",
        lambda html_path, pdf_path: {"success": False, "renderer": "test"},
    )

    metadata = generate_final_academic_paper(
        study_path,
        tmp_path / "final_paper",
        include_figures=True,
        overwrite=True,
    )

    outputs = metadata["outputs"]
    html_path = tmp_path / "final_paper" / "final_institutional_quant_finance_paper.html"
    html_text = html_path.read_text(encoding="utf-8")
    md_text = (tmp_path / "final_paper" / "final_institutional_quant_finance_paper.md").read_text(
        encoding="utf-8"
    )
    assert metadata["research_only"] is True
    assert "markdown" in outputs
    assert "html" in outputs
    assert metadata["pdf_export"]["status"] == "PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER"
    assert "25. Mathematical Appendix" in md_text
    assert "Rₜ = Pₜ/Pₜ₋₁ − 1" in html_text
    assert "This is not investment advice" in html_text
    assert " compra " not in html_text.lower()
    assert (tmp_path / "final_paper" / "metadata.json").exists()
    assert (tmp_path / "final_paper" / "reproducibility_manifest.json").exists()


def _sample_study() -> dict[str, object]:
    return {
        "report_type": "institutional_state_of_art_quant_study",
        "research_only": True,
        "plain_language_explanation": "Plain explanation.",
        "audit_controls": {"strict_real_data": True, "synthetic_rejected": True},
        "source_terminal_report": {
            "data_mode": "provider_yfinance",
            "warnings": ["RISK_FREE_PROXY_UNAVAILABLE_USING_ZERO_RATE"],
        },
        "tables": {
            "data_provenance": [{"data_mode": "provider_yfinance"}],
            "asset_metrics": [
                {
                    "asset_id": "AAPL",
                    "cagr": 0.1,
                    "annualized_volatility": 0.2,
                    "max_drawdown": -0.3,
                }
            ],
            "ml_audit": [{"asset_id": "AAPL", "strict_edge_validated": False}],
            "final_research_decision_table": [
                {
                    "asset_id": "AAPL",
                    "research_only_signal": "BLOCKED_FOR_BROKER_PROMOTION",
                    "broker_promotion_allowed": False,
                }
            ],
            "critical_findings": [{"severity": "HIGH", "finding": "Risk-free proxy unavailable"}],
            "implementation_roadmap_table": [{"priority": "HIGH", "module": "risk_free_curve"}],
            "glossary": [{"term": "VaR", "meaning": "Value at Risk"}],
        },
    }
