"""Final reproducible academic paper generator for institutional quant research."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.reporting.pdf_export import export_html_report_to_pdf

RESEARCH_ONLY_DISCLAIMER = (
    "This is not investment advice. This is a research-only quantitative signal based on "
    "historical data, assumptions and model limitations."
)

FINAL_SECTIONS = [
    "1. Title Page",
    "2. Abstract",
    "3. Plain-Language Executive Summary",
    "4. Research Questions",
    "5. Data Provenance and Audit Controls",
    "6. Literature-Based Methodology",
    "7. Data Quality and Market Data Limitations",
    "8. Return Construction and Compounding",
    "9. Single-Asset Performance Analysis",
    "10. CAPM and Factor-Model Interpretation",
    "11. Tail Risk: VaR and Expected Shortfall",
    "12. VaR Backtesting: Kupiec and Christoffersen",
    "13. Machine Learning and Predictive Validity",
    "14. Multiple Testing and Overfitting Controls",
    "15. Portfolio Construction",
    "16. Covariance Shrinkage and Robustness",
    "17. Efficient Frontier and Capital Allocation",
    "18. Execution Costs, Turnover and Capacity",
    "19. Research-Only Decision Framework",
    "20. Final Asset-Level Conclusions",
    "21. Final Portfolio-Level Conclusions",
    "22. What the Study Does Not Prove",
    "23. Required Improvements Before Broker/Paper/Live Trading",
    "24. Implementation Roadmap",
    "25. Mathematical Appendix",
    "26. Acronyms and Glossary",
    "27. Bibliography Traceability",
    "28. Reproducibility Appendix",
    "29. Operational Dependencies and Manual Configuration",
]

FORMULAS = [
    ("Simple return", "Rₜ = Pₜ/Pₜ₋₁ − 1"),
    ("Log return", "rₜ = ln(Pₜ/Pₜ₋₁)"),
    ("Portfolio return", "Rₚ,ₜ = Σᵢ wᵢ Rᵢ,ₜ"),
    ("Annual volatility", "σₐₙₙ = std(Rₜ) √252"),
    ("Sharpe", "SR = mean(Rₜ − r_f) / std(Rₜ − r_f) √252"),
    ("HAC Sharpe", "SR_HAC = SR / √(1 + 2Σₖ(1 − k/(q+1))ρₖ)"),
    ("Drawdown", "DDₜ = Vₜ / maxₛ≤ₜ(Vₛ) − 1"),
    ("CAPM beta", "βᵢ = Cov(Rᵢ, Rₘ) / Var(Rₘ)"),
    ("Historical VaR", "VaRα(L) = Qα(L), where Lₜ = −Rₜ"),
    ("Expected Shortfall", "ESα = E[L | L ≥ VaRα]"),
    ("Kupiec LR", "LR_uc = −2 ln(L(p₀) / L(p̂))"),
    ("Christoffersen LR", "LR_ind = −2 ln(L_independent / L_markov)"),
    ("Mean-variance risk", "σₚ = √(w′Σw)"),
    ("CVaR objective", "min_w ESα(−ΣᵢwᵢRᵢ,ₜ), with Σᵢwᵢ=1 and wᵢ≥0"),
    ("Concentration", "HHI = Σᵢwᵢ² and N_eff = 1/HHI"),
    ("Probabilistic Sharpe", "PSR = Φ((SR − SR*)√(T−1) / √(1 − γ₃SR + ((γ₄−1)/4)SR²))"),
    ("Execution impact", "cost ≈ notional × k × σ × √participation"),
]

BIBLIOGRAPHY_TRACE = [
    "GuKellyXiu2020; ChenPelgerZhu2024: ML asset pricing requires strict OOS evidence.",
    "LopezDePrado2018AFML; LopezDePrado2020MLAM: leakage, meta-labeling, backtest discipline.",
    "BaileyLopezPrado2014DSR; White2000RealityCheck; Hansen2005SPA: multiple-testing controls.",
    "Sharpe1964; FamaFrench1993; Carhart1997; FamaFrench2015: benchmark and factor interpretation.",
    "ArtznerDelbaenEberHeath1999; AcerbiTasche2002: coherent tail risk and Expected Shortfall.",
    "Kupiec1995VaR; Christoffersen1998IntervalForecasts: VaR exception backtesting.",
    "Markowitz1952; LedoitWolf2004; DeMiguel2009; BlackLitterman1992: portfolio construction.",
    "RockafellarUryasev2000CVaR: CVaR optimization objective.",
    "AlmgrenChriss2001; BertsimasLo1998; Gatheral2010; Kyle1985: execution and impact.",
]

OPERATIONAL_DEPENDENCIES = [
    {
        "purpose": "Run the validated project toolchain",
        "dependency": "Python 3.13 with project, dev, UI, ML and PDF extras",
        "command": "py -3.13 -m pip install -e .[dev,ui,ml,pdf]",
        "manual_configuration": "Run from quant-platform/. This installs pytest, ruff, Streamlit, Plotly, scikit-learn, scipy, WeasyPrint and Playwright bindings used by the study and UI.",
    },
    {
        "purpose": "Enable deterministic browser PDF fallback",
        "dependency": "Playwright Chromium browser binary",
        "command": "py -3.13 -m playwright install chromium",
        "manual_configuration": "Required only if WeasyPrint cannot render the HTML paper. It downloads the local Chromium binary used to print HTML to PDF.",
    },
    {
        "purpose": "Use public market data without broker credentials",
        "dependency": "yfinance provider",
        "command": "py -3.13 -m quant_platform.cli build-final-institutional-package --config configs/quant_terminal_10_stocks.yaml --provider yfinance --output-dir reports/generated/final_package --max-table-rows 20",
        "manual_configuration": "No API key is required. Internet access is required. Provider warnings remain in the paper and can block promotion gates.",
    },
    {
        "purpose": "Use licensed institutional factor data in future extensions",
        "dependency": "Fama-French/Carhart/five-factor returns or equivalent licensed factor files",
        "command": "Configure a point-in-time factor ingestion job before enabling factor-alpha gates.",
        "manual_configuration": "The current paper intentionally fails factor alpha closed until factor returns are ingested with dates, source, license and hashes.",
    },
    {
        "purpose": "Email the generated PDF",
        "dependency": "Local email client, SMTP account or manual attachment upload",
        "command": "Attach reports/generated/final_package/final_paper/final_institutional_quant_finance_paper.pdf to an email.",
        "manual_configuration": "The research CLI does not send email or store SMTP credentials. This avoids secret handling and accidental distribution.",
    },
]


class FinalAcademicPaperError(ValueError):
    """Raised when final paper generation cannot proceed safely."""


def generate_final_academic_paper(
    institutional_study_path: str | Path,
    output_dir: str | Path,
    *,
    include_figures: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate final Markdown, HTML, optional PDF, tables, figures and metadata."""

    source_path = Path(institutional_study_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Institutional study not found: {source_path}")
    study = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(study, dict):
        raise FinalAcademicPaperError("Institutional study must be a JSON object.")
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FinalAcademicPaperError(f"Output directory already contains files: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_generated_directory(output_path)
    tables_dir = output_path / "tables"
    figures_dir = output_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    table_paths = _write_final_tables(study, tables_dir)
    figure_paths = _copy_institutional_figures(source_path.parent, figures_dir) if include_figures else {}
    md_path = output_path / "final_institutional_quant_finance_paper.md"
    html_path = output_path / "final_institutional_quant_finance_paper.html"
    pdf_path = output_path / "final_institutional_quant_finance_paper.pdf"
    markdown = render_final_paper_markdown(study, table_paths, figure_paths)
    html_text = render_final_paper_html(study, table_paths, figure_paths)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    pdf_export = export_html_report_to_pdf(html_path, pdf_path)
    outputs = {"markdown": str(md_path), "html": str(html_path)}
    if pdf_export.get("success") and pdf_export.get("pdf_path"):
        outputs["pdf"] = str(pdf_export["pdf_path"])
    else:
        pdf_export["status"] = "PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER"
    manifest = _reproducibility_manifest(study, source_path, outputs, table_paths, figure_paths)
    manifest_path = output_path / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "report_type": "final_institutional_quant_finance_paper_metadata",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "not_investment_advice": True,
        "outputs": outputs,
        "tables": table_paths,
        "figures": figure_paths,
        "pdf_export": pdf_export,
        "reproducibility_manifest": str(manifest_path),
        "source_institutional_study": str(source_path),
        "warnings": _mapping(study.get("source_terminal_report")).get("warnings", []),
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def render_final_paper_markdown(
    study: dict[str, Any],
    table_paths: dict[str, str] | None = None,
    figure_paths: dict[str, str] | None = None,
) -> str:
    """Render the final paper as Markdown."""

    tables = _mapping(study.get("tables"))
    lines = [
        "# Final Institutional Quant Finance Research Paper",
        "",
        f"**{RESEARCH_ONLY_DISCLAIMER}**",
        "",
    ]
    for section in FINAL_SECTIONS:
        lines.extend(_markdown_section(section, study, tables, table_paths or {}, figure_paths or {}))
    return "\n".join(lines).strip() + "\n"


def render_final_paper_html(
    study: dict[str, Any],
    table_paths: dict[str, str] | None = None,
    figure_paths: dict[str, str] | None = None,
) -> str:
    """Render the final paper as journal-style HTML suitable for PDF export."""

    tables = _mapping(study.get("tables"))
    body = []
    for section in FINAL_SECTIONS:
        body.append(_html_section(section, study, tables, table_paths or {}, figure_paths or {}))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Final Institutional Quant Finance Paper</title>
  <style>
    @page {{ size: A4; margin: 17mm 15mm 19mm; }}
    body {{ margin:0; background:#f6f1e7; color:#101820; font-family: Georgia, 'Times New Roman', serif; line-height:1.55; }}
    main {{ max-width: 1060px; margin: 0 auto; padding: 34px 26px 72px; background:#fffaf0; }}
    h1 {{ font-family: Aptos, Segoe UI, sans-serif; font-size: 34px; letter-spacing:-0.03em; border-bottom: 4px solid #9a6b16; padding-bottom: 12px; }}
    h2 {{ font-family: Aptos, Segoe UI, sans-serif; font-size: 23px; margin-top: 34px; break-after: avoid; }}
    h2:not(:first-child) {{ break-before: page; }}
    h3 {{ font-family: Aptos, Segoe UI, sans-serif; font-size: 16px; color:#5c3b07; }}
    p {{ text-align: justify; }}
    table {{ border-collapse: collapse; width:100%; margin: 14px 0 18px; font-size: 11px; break-inside: avoid; }}
    th, td {{ border:1px solid #c9b894; padding: 6px 8px; vertical-align: top; }}
    th {{ background:#ecdfc3; font-family: Aptos, Segoe UI, sans-serif; }}
    .disclaimer {{ border:1px solid #9f1239; background:#fff1f2; color:#7f1d1d; padding: 10px 12px; border-radius: 8px; font-weight: 700; }}
    .formula {{ margin: 8px 0 12px; padding: 10px 14px; background:#ffffff; border-left:4px solid #9a6b16; font-family:'Cambria Math','Times New Roman',serif; font-size: 17px; break-inside: avoid; }}
    .plain {{ background:#f3ead8; border:1px solid #d7c39a; padding: 10px 12px; border-radius: 8px; }}
    .figure-box {{ border:1px solid #c9b894; background:#ffffff; padding: 12px; margin: 14px 0 18px; break-inside: avoid; overflow:hidden; }}
    .figure-box svg {{ max-width: 100%; }}
    .caption {{ color:#4b5563; font-size: 12px; }}
    .small {{ font-size: 12px; color:#4b5563; }}
  </style>
</head>
<body><main>
<h1>Final Institutional Quant Finance Research Paper</h1>
<div class="disclaimer">{html.escape(RESEARCH_ONLY_DISCLAIMER)}</div>
{''.join(body)}
</main></body></html>"""


def _markdown_section(
    section: str,
    study: dict[str, Any],
    tables: dict[str, Any],
    table_paths: dict[str, str],
    figure_paths: dict[str, str],
) -> list[str]:
    title = section
    plain, technical, table_names, figure_names = _section_payload(section, study, tables)
    lines = [f"## {title}", "", plain, "", technical, ""]
    if section.endswith("Mathematical Appendix"):
        for name, formula in FORMULAS:
            lines.extend([f"**{name}.** {formula}", ""])
    if section.endswith("Operational Dependencies and Manual Configuration"):
        lines.extend(["### Required setup", "", _markdown_table(OPERATIONAL_DEPENDENCIES), ""])
    for table_name in table_names:
        lines.extend([f"### Table: {table_name}", "", _markdown_table(tables.get(table_name, [])), ""])
    for figure_name in figure_names:
        path = figure_paths.get(figure_name)
        if path:
            lines.extend([f"### Figure: {figure_name}", "", f"Local HTML figure: `{path}`", ""])
    if section.endswith("Bibliography Traceability"):
        lines.extend([f"- {item}" for item in BIBLIOGRAPHY_TRACE])
        lines.append("")
    if section.endswith("Reproducibility Appendix"):
        lines.extend(["### Output Tables", ""])
        lines.extend(f"- `{name}`: `{path}`" for name, path in sorted(table_paths.items()))
        lines.extend(["", "### Output Figures", ""])
        lines.extend(f"- `{name}`: `{path}`" for name, path in sorted(figure_paths.items()))
        lines.append("")
    return lines


def _html_section(
    section: str,
    study: dict[str, Any],
    tables: dict[str, Any],
    table_paths: dict[str, str],
    figure_paths: dict[str, str],
) -> str:
    plain, technical, table_names, figure_names = _section_payload(section, study, tables)
    parts = [f"<h2>{html.escape(section)}</h2>"]
    parts.append(f"<div class='plain'><strong>Plain-language explanation.</strong> {html.escape(plain)}</div>")
    parts.append(f"<p><strong>Technical interpretation.</strong> {html.escape(technical)}</p>")
    if section.endswith("Mathematical Appendix"):
        for name, formula in FORMULAS:
            parts.append(
                f"<div class='formula'><strong>{html.escape(name)}:</strong> "
                f"{html.escape(formula)}</div>"
            )
    if section.endswith("Operational Dependencies and Manual Configuration"):
        parts.append("<h3>Required setup</h3>")
        parts.append(_html_table(OPERATIONAL_DEPENDENCIES))
    for table_name in table_names:
        parts.append(f"<h3>Table: {html.escape(table_name)}</h3>")
        parts.append(_html_table(tables.get(table_name, [])))
    for figure_name in figure_names:
        path = figure_paths.get(figure_name)
        if path:
            parts.append(f"<h3>Figure: {html.escape(figure_name)}</h3>")
            parts.append(_embedded_figure(Path(path)))
    if section.endswith("Bibliography Traceability"):
        parts.append("<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in BIBLIOGRAPHY_TRACE) + "</ul>")
    if section.endswith("Reproducibility Appendix"):
        parts.append("<h3>Output Tables</h3>" + _path_list(table_paths))
        parts.append("<h3>Output Figures</h3>" + _path_list(figure_paths))
    return "".join(parts)


def _section_payload(
    section: str,
    study: dict[str, Any],
    tables: dict[str, Any],
) -> tuple[str, str, list[str], list[str]]:
    source = _mapping(study.get("source_terminal_report"))
    controls = _mapping(study.get("audit_controls"))
    table_names: list[str] = []
    figure_names: list[str] = []
    plain = "This section explains the evidence without assuming a finance background."
    technical = "The result is interpreted under historical-data assumptions and model limitations."
    if section.endswith("Title Page"):
        plain = (
            "This is a local, reproducible institutional research report. It studies historical "
            "market data and produces research-only signals, not actions."
        )
        technical = (
            f"Source mode is {source.get('data_mode')}; strict real-data control is "
            f"{controls.get('strict_real_data')}; synthetic rejection is "
            f"{controls.get('synthetic_rejected')}."
        )
        table_names = ["data_provenance"]
    elif section.endswith("Abstract"):
        plain = _abstract_plain(tables)
        technical = _abstract_technical(tables)
    elif section.endswith("Plain-Language Executive Summary"):
        plain = str(study.get("plain_language_explanation", ""))
        technical = "Key blockers are provider warnings, risk-free proxy failure and weak broad ML evidence."
        table_names = ["critical_findings", "final_research_decision_table"]
    elif section.endswith("Research Questions"):
        plain = "The study asks what the historical data support, what they do not support, and what blocks broker promotion."
        technical = "The decision problem is constrained by data provenance, OOS evidence, tail risk, covariance robustness and execution-cost uncertainty."
    elif section.endswith("Data Provenance and Audit Controls"):
        plain = "The report records where data came from and whether warnings were raised."
        technical = "Provider mode, benchmark, aligned observations and warnings are mandatory audit fields."
        table_names = ["data_provenance"]
    elif section.endswith("Literature-Based Methodology"):
        plain = "The methods follow current academic evidence rather than chart-only heuristics."
        technical = "ML, risk, portfolio and execution blocks are mapped to peer-reviewed literature and fail closed when data are missing."
        table_names = ["method_traceability", "state_of_art_gap_analysis"]
    elif section.endswith("Data Quality and Market Data Limitations"):
        plain = "Some provider symbols failed and one fallback changed the universe."
        technical = "Fallbacks and risk-free proxy failure are treated as material blockers."
        table_names = ["critical_findings"]
    elif section.endswith("Return Construction and Compounding"):
        plain = "Prices are converted into percentage returns so assets with different prices can be compared."
        technical = "Simple returns drive the core study; log returns remain a documented modeling extension."
        table_names = ["price_sample_first_rows", "return_sample_first_rows"]
        figure_names = ["normalized_prices", "cumulative_returns"]
    elif section.endswith("Single-Asset Performance Analysis"):
        plain = "This section separates historical return from risk and drawdown pain."
        technical = "CAGR, volatility, Sharpe, Sortino, beta, Jensen proxy alpha and drawdown are descriptive statistics."
        table_names = ["asset_metrics"]
        figure_names = ["risk_return_scatter", "drawdowns"]
    elif section.endswith("CAPM and Factor-Model Interpretation"):
        plain = "Market beta is not the same as true skill or alpha."
        technical = "Current alpha is benchmark-proxy based; Fama-French/Carhart/five-factor alpha requires external factor returns."
        table_names = ["factor_model_gap_table"]
    elif section.endswith("Tail Risk: VaR and Expected Shortfall"):
        plain = "Tail risk focuses on bad days and how bad losses are after thresholds break."
        technical = "VaR is a quantile of positive losses; ES is the conditional tail mean."
        table_names = ["tail_risk", "portfolio_tail_risk", "cvar_optimized_portfolios"]
    elif section.endswith("VaR Backtesting: Kupiec and Christoffersen"):
        plain = "The study counts how often losses exceeded the VaR threshold."
        technical = "Kupiec checks unconditional coverage; Christoffersen checks exception independence."
        table_names = ["tail_risk_backtesting", "var_exception_table"]
        figure_names = ["var_exceptions_through_time", "exception_traffic_light"]
    elif section.endswith("Machine Learning and Predictive Validity"):
        plain = "Forecasts are not accepted just because they fit the past."
        technical = "The model must beat a naive baseline out of sample and show positive IC/OOS evidence."
        table_names = ["ml_audit", "model_confidence"]
        figure_names = ["ml_confidence_comparison"]
    elif section.endswith("Multiple Testing and Overfitting Controls"):
        plain = "If many models or assets are tried, the best result can be luck."
        technical = "PSR/DSR approximation applies a best-of-N Sharpe haircut; White/SPA/MCS remain future work."
        table_names = ["multiple_testing_adjustments"]
        figure_names = ["sharpe_vs_deflated_sharpe"]
    elif section.endswith("Portfolio Construction"):
        plain = "A portfolio is not just a list of strong assets; correlations decide diversification."
        technical = "Weights are long-only; risk is estimated through covariance and correlation."
        table_names = ["correlation_matrix", "correlation_pairs", "portfolio_weights", "portfolio_diagnostics"]
        figure_names = ["correlation_heatmap", "rolling_correlation_stability"]
    elif section.endswith("Covariance Shrinkage and Robustness"):
        plain = "Optimized weights can move because covariance estimates are noisy."
        technical = "Sample covariance is compared with shrinkage covariance and weight shifts are measured."
        table_names = ["covariance_shrinkage_comparison", "portfolio_robustness", "shrinkage_sensitivity"]
        figure_names = ["shrinkage_weight_comparison", "portfolio_robustness_heatmap"]
    elif section.endswith("Efficient Frontier and Capital Allocation"):
        plain = "The frontier is an approximate map of risk and return choices, not a guaranteed optimum."
        technical = "The grid frontier is in-sample and should be interpreted as a cloud of feasible long-only portfolios."
        table_names = ["frontier_sample"]
        figure_names = ["efficient_frontier"]
    elif section.endswith("Execution Costs, Turnover and Capacity"):
        plain = "Even a statistically attractive portfolio can fail after costs."
        technical = "Costs use spread and square-root impact proxies, not broker-calibrated fills."
        table_names = ["execution_cost_sensitivity"]
        figure_names = ["execution_cost_sensitivity"]
    elif section.endswith("Research-Only Decision Framework"):
        plain = "Every final output is a research-only signal and blocks live or paper broker promotion."
        technical = "Gate evidence combines provider warnings, risk-free availability, ML confidence, drawdown and VaR tests."
        table_names = ["decision_gate_evidence", "final_research_decision_table"]
        figure_names = ["decision_scores", "final_decision_waterfall"]
    elif section.endswith("Final Asset-Level Conclusions"):
        plain = _asset_plain_conclusions(tables)
        technical = "Asset conclusions are descriptive and constrained by the final decision table."
        table_names = ["asset_metrics", "final_research_decision_table"]
    elif section.endswith("Final Portfolio-Level Conclusions"):
        plain = "The most balanced portfolio cannot be promoted without robust data, factor and execution controls."
        technical = "Min-variance, max-Sharpe, CVaR and shrinkage diagnostics remain in-sample research tools."
        table_names = ["portfolio_tail_risk", "portfolio_robustness", "execution_cost_sensitivity"]
    elif section.endswith("What the Study Does Not Prove"):
        plain = "The study does not prove future performance, factor alpha, execution feasibility or suitability for any person."
        technical = "Historical data cannot establish invariance under regime change, data snooping, liquidity stress or factor adjustment."
    elif section.endswith("Required Improvements Before Broker/Paper/Live Trading"):
        plain = "Broker promotion remains blocked until missing institutional controls are implemented."
        technical = "Required controls include risk-free curve, licensed data, factor models, White/SPA/MCS, execution calibration and kill switches."
        table_names = ["implementation_roadmap_table"]
    elif section.endswith("Implementation Roadmap"):
        plain = "The roadmap prioritizes blockers before more complex models."
        technical = "Data and validation controls precede optimizer and broker workflow expansion."
        table_names = ["implementation_roadmap_table"]
    elif section.endswith("Acronyms and Glossary"):
        plain = "Definitions help non-specialists read the report without diluting rigor."
        technical = "The glossary standardizes interpretation of risk, ML and portfolio terms."
        table_names = ["glossary"]
    elif section.endswith("Bibliography Traceability"):
        plain = "Each implemented block is connected to finance literature."
        technical = "References are methodological constraints, not decorative citations."
    elif section.endswith("Reproducibility Appendix"):
        plain = "The report can be regenerated from local JSON, tables and figures."
        technical = "Metadata and manifests record source paths, hashes, warnings and generated artifacts."
    elif section.endswith("Operational Dependencies and Manual Configuration"):
        plain = "This section lists what must be installed or configured manually if local rendering, data access or distribution fails."
        technical = "The package installs analytical dependencies automatically where permitted, but browser binaries, licensed factor data and email delivery remain explicit operational steps."
    return plain, technical, table_names, figure_names


def _abstract_plain(tables: dict[str, Any]) -> str:
    assets = len(tables.get("asset_metrics", []))
    decisions = pd.DataFrame(tables.get("final_research_decision_table", []))
    blocked = int((decisions.get("broker_promotion_allowed") == False).sum()) if not decisions.empty else assets  # noqa: E712
    return (
        f"The study analyzes {assets} equities using historical provider data. It finds useful "
        f"research evidence, but {blocked} assets remain blocked for broker promotion under "
        "current controls."
    )


def _abstract_technical(tables: dict[str, Any]) -> str:
    ml = pd.DataFrame(tables.get("ml_audit", []))
    passed = int((ml.get("strict_edge_validated") == True).sum()) if not ml.empty else 0  # noqa: E712
    return (
        f"The pipeline computes descriptive returns, VaR/ES, VaR backtests, HAC Sharpe, "
        f"DSR approximation, model confidence, shrinkage robustness and execution proxies. "
        f"Only {passed} assets pass strict predictive gates in the current generated study."
    )


def _asset_plain_conclusions(tables: dict[str, Any]) -> str:
    frame = pd.DataFrame(tables.get("asset_metrics", []))
    if frame.empty:
        return "No asset-level metrics are available."
    best_cagr = _extreme_label(frame, "cagr", largest=True)
    worst_drawdown = _extreme_label(frame, "max_drawdown", largest=False)
    highest_vol = _extreme_label(frame, "annualized_volatility", largest=True)
    return (
        f"Historically, {best_cagr} stands out on compounded return, {highest_vol} stands "
        f"out on volatility, and {worst_drawdown} stands out on drawdown severity. These are "
        "descriptive observations, not promotion decisions."
    )


def _extreme_label(frame: pd.DataFrame, column: str, *, largest: bool) -> str:
    if column not in frame.columns or "asset_id" not in frame.columns:
        return "unavailable"
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return "unavailable"
    idx = values.idxmax() if largest else values.idxmin()
    return f"{frame.loc[idx, 'asset_id']} ({column}={values.loc[idx]:.4f})"


def _write_final_tables(study: dict[str, Any], tables_dir: Path) -> dict[str, str]:
    paths = {}
    tables = _mapping(study.get("tables"))
    for index, (name, rows) in enumerate(tables.items(), start=1):
        path = tables_dir / f"{index:02d}_{name}.csv"
        pd.DataFrame(rows if isinstance(rows, list) else []).to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def _copy_institutional_figures(source_dir: Path, figures_dir: Path) -> dict[str, str]:
    source_figures = source_dir / "figures"
    if not source_figures.exists():
        return {}
    paths = {}
    for path in sorted(source_figures.glob("*.html")):
        target = figures_dir / path.name
        shutil.copy2(path, target)
        name = path.stem.split("_", 1)[-1]
        paths[name] = str(target)
    return paths


def _reproducibility_manifest(
    study: dict[str, Any],
    source_path: Path,
    outputs: dict[str, str],
    tables: dict[str, str],
    figures: dict[str, str],
) -> dict[str, Any]:
    files = {**outputs, **{f"table:{k}": v for k, v in tables.items()}, **{f"figure:{k}": v for k, v in figures.items()}}
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "source_institutional_study": str(source_path),
        "source_sha256": _hash_file(source_path),
        "source_terminal_report": study.get("source_terminal_report", {}),
        "audit_controls": study.get("audit_controls", {}),
        "files": {name: {"path": path, "sha256": _hash_file(Path(path))} for name, path in files.items()},
    }


def _embedded_figure(path: Path) -> str:
    if not path.exists():
        return f"<p class='small'>Figure file unavailable: {html.escape(str(path))}</p>"
    text = path.read_text(encoding="utf-8", errors="ignore")
    body = _extract_body(text)
    return (
        "<div class='figure-box'>"
        + body
        + f"<p class='caption'>Source figure: {html.escape(path.name)}</p></div>"
    )


def _extract_body(text: str) -> str:
    lower = text.lower()
    start = lower.find("<body")
    if start == -1:
        return text
    start = lower.find(">", start)
    end = lower.rfind("</body>")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start + 1 : end]


def _markdown_table(rows: object, *, max_rows: int = 20) -> str:
    if not isinstance(rows, list) or not rows:
        return "No rows available."
    clean_rows = [_mapping(row) for row in rows[:max_rows]]
    columns = list(dict.fromkeys(column for row in clean_rows for column in row))
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in clean_rows:
        body.append("| " + " | ".join(_md_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def _html_table(rows: object, *, max_rows: int = 20) -> str:
    if not isinstance(rows, list) or not rows:
        return "<p>No rows available.</p>"
    clean_rows = [_mapping(row) for row in rows[:max_rows]]
    columns = list(dict.fromkeys(column for row in clean_rows for column in row))
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in clean_rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(_json_safe(row.get(column))))}</td>" for column in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _path_list(paths: dict[str, str]) -> str:
    if not paths:
        return "<p>No paths available.</p>"
    return "<ul>" + "".join(
        f"<li><strong>{html.escape(name)}</strong>: {html.escape(path)}</li>"
        for name, path in sorted(paths.items())
    ) + "</ul>"


def _clear_generated_directory(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if pd.notna(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _md_cell(value: object) -> str:
    text = str(_json_safe(value))
    return text.replace("|", "\\|").replace("\n", " ")[:240]
