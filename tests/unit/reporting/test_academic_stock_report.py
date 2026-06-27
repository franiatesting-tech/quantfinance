from __future__ import annotations

import json

from quant_platform.reporting.academic_stock_report import (
    build_stock_academic_report_model,
    render_stock_report_html,
    render_stock_report_markdown,
    write_stock_academic_report,
)


def test_academic_stock_report_renders_required_sections_and_outputs(tmp_path) -> None:  # noqa: ANN001
    full_report = _sample_full_report()
    model = build_stock_academic_report_model(full_report, "AAPL")

    markdown = render_stock_report_markdown(model)
    html = render_stock_report_html(model)
    metadata = write_stock_academic_report(
        model,
        tmp_path,
        formats=("md", "html"),
        include_figures=False,
    )

    assert "## 5. Data & Provenance" in markdown
    assert "## 22. Mathematical Appendix" in markdown
    assert "Research-only. This is not investment advice." in markdown
    assert "<html" in html
    assert (tmp_path / "AAPL" / "AAPL_academic_report.md").exists()
    assert (tmp_path / "AAPL" / "AAPL_academic_report.html").exists()
    assert json.loads((tmp_path / "AAPL" / "metadata.json").read_text())["asset_id"] == "AAPL"
    assert metadata["research_only"] is True


def _sample_full_report() -> dict[str, object]:
    stock = {
        "asset_id": "AAPL",
        "data_used": {
            "provider": "synthetic",
            "data_mode": "offline_synthetic",
            "frequency": "1d",
            "currency": "USD",
            "benchmark": "SPY",
            "risk_free_proxy": "^IRX",
            "risk_free_rate_annual": 0.0,
            "start_timestamp": "2024-01-01",
            "end_timestamp": "2024-01-02",
            "observations": 2,
        },
        "data_quality": {},
        "metrics": {
            "annualized_return": 0.1,
            "annualized_volatility": 0.2,
            "final_cumulative_return": 0.1,
            "max_drawdown": -0.1,
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.7,
            "beta_to_benchmark": 1.0,
            "jensen_alpha": 0.01,
            "treynor_ratio": 0.1,
        },
        "var": {},
        "monte_carlo": {},
        "backtesting_results": {},
        "options_theoretical_analytics": {},
    }
    return {
        "metadata": {"report_version": "test"},
        "data_provenance": {},
        "universe": {"selected_stocks": ["AAPL"]},
        "stocks": {"AAPL": stock},
        "bibliography": [],
    }
