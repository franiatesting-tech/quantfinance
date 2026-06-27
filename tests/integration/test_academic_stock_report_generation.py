from __future__ import annotations

from quant_platform.reporting.academic_stock_report import write_all_stock_academic_reports


def test_write_all_stock_academic_reports_generates_markdown_html_and_metadata(tmp_path) -> None:  # noqa: ANN001
    results = write_all_stock_academic_reports(
        _sample_full_report(),
        output_dir=tmp_path,
        formats=("md", "html"),
        include_figures=False,
    )

    assert len(results) == 1
    assert (tmp_path / "AAPL" / "AAPL_academic_report.md").exists()
    assert (tmp_path / "AAPL" / "AAPL_academic_report.html").exists()
    assert (tmp_path / "AAPL" / "metadata.json").exists()


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
        "metrics": {"annualized_return": 0.1, "annualized_volatility": 0.2},
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
