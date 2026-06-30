"""Institutional multi-asset study package for auditable quant research."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.research.portfolio_optimization import long_only_weight_grid
from quant_platform.research.state_of_art.covariance_shrinkage import (
    covariance_shrinkage_comparison,
)
from quant_platform.research.state_of_art.execution_costs import execution_cost_sensitivity
from quant_platform.research.state_of_art.factor_models import (
    factor_model_gap_table as build_factor_model_gap_table,
)
from quant_platform.research.state_of_art.final_research_gates import final_research_decision
from quant_platform.research.state_of_art.model_confidence import model_confidence_table
from quant_platform.research.state_of_art.multiple_testing import (
    multiple_testing_adjustment_table,
)
from quant_platform.research.state_of_art.portfolio_robustness import (
    portfolio_robustness_summary,
)
from quant_platform.research.state_of_art.tail_risk_backtesting import (
    backtest_var,
    var_exceptions,
)


class InstitutionalStudyError(ValueError):
    """Raised when an institutional study cannot be built safely."""


def build_institutional_study(
    terminal_report: dict[str, Any],
    *,
    max_table_rows: int = 20,
    strict_real_data: bool = True,
) -> dict[str, Any]:
    """Build a JSON-safe institutional study from a terminal report.

    The function does not invent data. It only summarizes observed series and model outputs
    already present in the terminal report. In strict mode it rejects synthetic report modes.
    """

    if max_table_rows < 1:
        raise InstitutionalStudyError("max_table_rows must be >= 1.")
    _validate_terminal_report(terminal_report, strict_real_data=strict_real_data)
    stocks = _mapping(terminal_report.get("stocks"))
    prices = _wide_stock_series(stocks, "price_series", "price")
    returns = _wide_stock_series(stocks, "simple_returns", "return")
    portfolio = _mapping(terminal_report.get("portfolio"))
    optimization = _mapping(terminal_report.get("optimization"))
    correlation = _matrix_frame(portfolio.get("correlation_matrix"))
    covariance = _matrix_frame(portfolio.get("annualized_covariance_matrix"))
    if correlation.empty and not returns.empty:
        correlation = returns.corr()
    if covariance.empty and not returns.empty:
        covariance = returns.cov() * 252.0
    critical_findings = _critical_findings(terminal_report, stocks)
    asset_metrics = _asset_metrics_table(stocks)
    tail_risk = _tail_risk_table(stocks)
    ml_audit = _ml_audit_table(stocks)
    tail_backtesting = _tail_risk_backtesting_table(returns, stocks)
    model_confidence = model_confidence_table(_records(ml_audit))
    final_decisions = _final_research_decision_table(
        terminal_report,
        asset_metrics,
        ml_audit,
        tail_backtesting,
        model_confidence,
    )
    tables = {
        "data_provenance": _provenance_table(terminal_report),
        "price_sample_first_rows": _records(_sample_wide_frame(prices, max_table_rows)),
        "return_sample_first_rows": _records(_sample_wide_frame(returns, max_table_rows)),
        "asset_metrics": _records(asset_metrics),
        "tail_risk": _records(tail_risk),
        "portfolio_tail_risk": _records(_portfolio_tail_risk_table(returns, terminal_report)),
        "cvar_optimized_portfolios": _records(_cvar_optimized_portfolios_table(returns)),
        "autocorrelation_adjusted_sharpe": _records(
            _autocorrelation_adjusted_sharpe_table(returns, terminal_report)
        ),
        "tail_risk_backtesting": _records(tail_backtesting),
        "var_exception_table": _records(
            _var_exception_table(returns, tail_risk, max_table_rows)
        ),
        "multiple_testing_adjustments": _records(
            multiple_testing_adjustment_table(
                _records(asset_metrics),
                number_of_trials=max(len(stocks), 1),
            )
        ),
        "ml_audit": _records(ml_audit),
        "model_confidence": _records(model_confidence),
        "decision_signals": _records(_decision_table(stocks)),
        "broker_research_gates": _records(_broker_gate_table(stocks)),
        "decision_gate_evidence": _records(
            _decision_gate_evidence_table(
                terminal_report,
                asset_metrics,
                ml_audit,
                tail_backtesting,
                model_confidence,
            )
        ),
        "final_research_decision_table": _records(final_decisions),
        "correlation_matrix": _records(_matrix_to_table(correlation)),
        "correlation_pairs": _records(_correlation_pair_table(correlation, max_table_rows)),
        "annual_regime_stability": _records(_annual_regime_stability_table(returns)),
        "annualized_covariance_matrix": _records(_matrix_to_table(covariance)),
        "portfolio_weights": _records(_portfolio_weight_table(terminal_report)),
        "portfolio_diagnostics": _records(_portfolio_diagnostics_table(terminal_report)),
        "shrinkage_sensitivity": _records(_shrinkage_sensitivity_table(returns)),
        "covariance_shrinkage_comparison": _records(
            _covariance_shrinkage_comparison_table(returns, terminal_report)
        ),
        "portfolio_robustness": _records(_portfolio_robustness_table(returns)),
        "factor_model_gap_table": _records(_factor_model_gap_table(stocks)),
        "execution_cost_sensitivity": _records(
            _execution_cost_sensitivity_table(terminal_report, returns)
        ),
        "frontier_sample": _records(_frontier_table(optimization, max_table_rows)),
        "state_of_art_gap_analysis": _state_of_art_gap_analysis_table(),
        "implementation_roadmap_table": _implementation_roadmap_table(),
        "critical_findings": critical_findings,
        "glossary": _glossary_table(),
        "formula_catalog": _formula_catalog_table(),
        "method_traceability": _method_traceability(),
    }
    study = {
        "report_type": "institutional_state_of_art_quant_study",
        "report_version": "institutional_study_v1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "not_investment_advice": True,
        "source_terminal_report": {
            "report_type": terminal_report.get("report_type"),
            "generated_at": terminal_report.get("generated_at"),
            "data_mode": _data_mode(terminal_report),
            "warnings": terminal_report.get("warnings", []),
        },
        "audit_controls": _audit_controls(terminal_report, strict_real_data=strict_real_data),
        "universe": terminal_report.get("universe", {}),
        "study_design": _study_design(),
        "plain_language_explanation": _plain_language_explanation(),
        "tables": tables,
        "critical_project_changes": _project_change_recommendations(critical_findings),
        "limitations": _study_limitations(terminal_report),
    }
    return _json_safe(study)


def write_institutional_study(
    terminal_report: dict[str, Any],
    output_dir: str | Path,
    *,
    max_table_rows: int = 20,
    strict_real_data: bool = True,
) -> dict[str, Any]:
    """Write study JSON, Markdown, CSV tables and HTML/SVG figures."""

    study = build_institutional_study(
        terminal_report,
        max_table_rows=max_table_rows,
        strict_real_data=strict_real_data,
    )
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    figures_dir = output_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_generated_files(tables_dir)
    _clear_generated_files(figures_dir)
    table_paths = _write_tables(study["tables"], tables_dir)
    figure_paths = _write_figures(terminal_report, study, figures_dir)
    json_path = output_path / "institutional_quant_study.json"
    md_path = output_path / "institutional_quant_study.md"
    paper_path = output_path / "institutional_quant_finance_paper.md"
    json_path.write_text(json.dumps(study, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(
        render_institutional_study_markdown(study, table_paths, figure_paths),
        encoding="utf-8",
    )
    paper_path.write_text(
        render_institutional_academic_paper_markdown(study, table_paths, figure_paths),
        encoding="utf-8",
    )
    metadata = {
        "report_type": "institutional_quant_study_metadata",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "strict_real_data": strict_real_data,
        "outputs": {
            "json": str(json_path),
            "markdown": str(md_path),
            "academic_paper_markdown": str(paper_path),
            "tables": table_paths,
            "figures": figure_paths,
        },
        "source_data_mode": _data_mode(terminal_report),
        "critical_findings_count": len(study["tables"]["critical_findings"]),
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def render_institutional_study_markdown(
    study: dict[str, Any],
    table_paths: dict[str, str] | None = None,
    figure_paths: dict[str, str] | None = None,
) -> str:
    """Render the institutional study as readable Markdown."""

    tables = _mapping(study.get("tables"))
    controls = _mapping(study.get("audit_controls"))
    lines = [
        "# Institutional State-of-the-Art Quant Study",
        "",
        "**Research-only. This is not investment advice. No broker order is generated.**",
        "",
        "## Executive Summary",
        "",
        _paragraph(study.get("plain_language_explanation")),
        "",
        "## Audit Controls",
        "",
        _markdown_table([controls]),
        "",
        "## Study Design",
        "",
        _markdown_table(study.get("study_design", [])),
        "",
        "## Data Provenance",
        "",
        _markdown_table(tables.get("data_provenance", [])),
        "",
        "## First Data Rows",
        "",
        "Prices are shown as the first rows of the aligned close-price matrix. Returns are "
        "computed as `R_t = P_t / P_{t-1} - 1`, so they start one row after prices.",
        "",
        "### Price Sample",
        "",
        _markdown_table(tables.get("price_sample_first_rows", [])),
        "",
        "### Return Sample",
        "",
        _markdown_table(tables.get("return_sample_first_rows", [])),
        "",
        "## Asset Metrics",
        "",
        _markdown_table(tables.get("asset_metrics", [])),
        "",
        "## Correlation And Portfolio Construction",
        "",
        "Correlation measures whether assets tend to move together. Portfolio construction "
        "uses this because two assets with similar standalone returns can have very different "
        "portfolio value if their drawdowns happen at different times.",
        "",
        "### Correlation Matrix",
        "",
        _markdown_table(tables.get("correlation_matrix", [])),
        "",
        "### Portfolio Weights",
        "",
        _markdown_table(tables.get("portfolio_weights", [])),
        "",
        "## Tail Risk And Validation",
        "",
        "VaR estimates a loss threshold. Expected Shortfall estimates the average loss after "
        "that threshold is crossed. This project uses the explicit loss convention `L_t = -R_t`.",
        "",
        _markdown_table(tables.get("tail_risk", [])),
        "",
        "### Portfolio Tail Risk",
        "",
        _markdown_table(tables.get("portfolio_tail_risk", [])),
        "",
        "### CVaR-Optimized Portfolios",
        "",
        _markdown_table(tables.get("cvar_optimized_portfolios", [])),
        "",
        "### VaR Backtesting",
        "",
        _markdown_table(tables.get("tail_risk_backtesting", [])),
        "",
        "### VaR Exception Rows",
        "",
        _markdown_table(tables.get("var_exception_table", [])),
        "",
        "## Predictive Model Audit",
        "",
        "A forecast is only useful if it beats simple baselines out of sample. The ML table "
        "therefore emphasizes OOS R-squared, information coefficient, directional accuracy "
        "and naive-baseline comparison.",
        "",
        _markdown_table(tables.get("ml_audit", [])),
        "",
        "## Autocorrelation-Adjusted Sharpe",
        "",
        _markdown_table(tables.get("autocorrelation_adjusted_sharpe", [])),
        "",
        "## Multiple Testing And Model Confidence",
        "",
        _markdown_table(tables.get("multiple_testing_adjustments", [])),
        "",
        _markdown_table(tables.get("model_confidence", [])),
        "",
        "## Annual Regime Stability",
        "",
        _markdown_table(tables.get("annual_regime_stability", [])),
        "",
        "## Robust Portfolio Diagnostics",
        "",
        _markdown_table(tables.get("covariance_shrinkage_comparison", [])),
        "",
        _markdown_table(tables.get("portfolio_robustness", [])),
        "",
        "## Factor And Execution Gaps",
        "",
        _markdown_table(tables.get("factor_model_gap_table", [])),
        "",
        _markdown_table(tables.get("execution_cost_sensitivity", [])),
        "",
        "## Research-Only Broker Gates",
        "",
        "These gates translate statistics into research workflow states. They are not BUY/SELL "
        "orders and do not produce live sizing.",
        "",
        _markdown_table(tables.get("broker_research_gates", [])),
        "",
        _markdown_table(tables.get("decision_gate_evidence", [])),
        "",
        _markdown_table(tables.get("final_research_decision_table", [])),
        "",
        "## Critical Findings And Required Project Changes",
        "",
        _markdown_table(tables.get("critical_findings", [])),
        "",
        "## Method Traceability",
        "",
        _markdown_table(tables.get("method_traceability", [])),
        "",
        "## Implementation Roadmap",
        "",
        _markdown_table(tables.get("implementation_roadmap_table", [])),
        "",
        "## Figures",
        "",
    ]
    for name, path in sorted((figure_paths or {}).items()):
        lines.append(f"- `{name}`: `{path}`")
    if table_paths:
        lines.extend(["", "## CSV Tables", ""])
        for name, path in sorted(table_paths.items()):
            lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Limitations", "", _markdown_table(study.get("limitations", [])), ""])
    return "\n".join(lines).strip() + "\n"


def render_institutional_academic_paper_markdown(
    study: dict[str, Any],
    table_paths: dict[str, str] | None = None,
    figure_paths: dict[str, str] | None = None,
) -> str:
    """Render a detailed academic paper from the institutional study payload."""

    tables = _mapping(study.get("tables"))
    source = _mapping(study.get("source_terminal_report"))
    controls = _mapping(study.get("audit_controls"))
    lines = [
        "# An Auditable State-of-the-Art Quantitative Finance Study For Multi-Asset "
        "Equity Research",
        "",
        "## Abstract",
        "",
        _paper_abstract(study),
        "",
        "## Keywords",
        "",
        "Quantitative finance; empirical asset pricing; portfolio construction; VaR; "
        "Expected Shortfall; CVaR optimization; HAC Sharpe; walk-forward validation; "
        "broker research gates; model risk.",
        "",
        "## 1. Research Scope And Non-Advice Statement",
        "",
        "This paper studies a universe of public equities using observed provider data. "
        "It is explicitly research-only: no live broker order, personalized advice, margin "
        "instruction or position-sizing command is produced. A broker-style gate is used "
        "only as a research control: it says whether evidence is strong enough to continue "
        "toward future paper trading, not whether to buy or sell.",
        "",
        "## 2. Glossary Of Economic And Statistical Terms",
        "",
        _markdown_table(tables.get("glossary", [])),
        "",
        "## 3. Data Provenance And Audit Controls",
        "",
        "A quantitative result is only meaningful if the exact input data are known. The "
        "pipeline therefore records provider mode, benchmark, time window, warnings and "
        "strict synthetic-data rejection. Registered datasets also carry a SHA-256 hash, "
        "so later tampering of `data.csv` is detected before calculations run.",
        "",
        _markdown_table(tables.get("data_provenance", [])),
        "",
        "Audit-control summary:",
        "",
        _markdown_table([controls]),
        "",
        "## 4. Mathematical Convention And Formula Catalogue",
        "",
        "The project uses positive losses for risk calculations: `L_t = -R_t`. A negative "
        "return therefore becomes a positive loss. This avoids sign ambiguity in VaR and "
        "Expected Shortfall.",
        "",
        _markdown_table(tables.get("formula_catalog", [])),
        "",
        "## 5. Empirical Design From The Recent Literature",
        "",
        _literature_design_text(),
        "",
        _markdown_table(tables.get("method_traceability", [])),
        "",
        "## 6. Data Tables Used In The Study",
        "",
        "The full aligned matrices can be long. The paper shows only the first rows, while "
        "CSV outputs keep the generated tables separately. Prices are provider-adjusted "
        "daily close prices. Returns start one observation later because they need both "
        "`P_t` and `P_{t-1}`.",
        "",
        "### 6.1 First Price Rows",
        "",
        _markdown_table(tables.get("price_sample_first_rows", [])),
        "",
        "### 6.2 First Return Rows",
        "",
        _markdown_table(tables.get("return_sample_first_rows", [])),
        "",
        "## 7. Single-Asset Results",
        "",
        _single_asset_results_text(tables.get("asset_metrics", [])),
        "",
        _markdown_table(tables.get("asset_metrics", [])),
        "",
        "## 8. Tail-Risk Results",
        "",
        "VaR answers: how large is the loss threshold at a given confidence level? ES "
        "answers the more important question: if we are already beyond that threshold, "
        "how bad is the average tail loss? This is why ES/CVaR is central in modern risk "
        "management and portfolio optimization.",
        "",
        _markdown_table(tables.get("tail_risk", [])),
        "",
        "### 8.1 Portfolio Tail Risk",
        "",
        _markdown_table(tables.get("portfolio_tail_risk", [])),
        "",
        "### 8.2 Empirical CVaR-Optimized Portfolios",
        "",
        "The following table minimizes empirical Expected Shortfall over a long-only weight "
        "grid. It is an auditable approximation of the Rockafellar-Uryasev objective, not "
        "yet a production convex optimizer with turnover, liquidity and walk-forward "
        "constraints.",
        "",
        _markdown_table(tables.get("cvar_optimized_portfolios", [])),
        "",
        "### 8.3 VaR Backtesting",
        "",
        "A VaR number is not enough. The study therefore records realized exceptions and "
        "applies Kupiec unconditional-coverage and Christoffersen independence diagnostics. "
        "This remains an in-sample diagnostic when the VaR threshold is estimated on the "
        "same sample, so it supports review but does not certify a production risk model.",
        "",
        _markdown_table(tables.get("tail_risk_backtesting", [])),
        "",
        _markdown_table(tables.get("var_exception_table", [])),
        "",
        "## 9. Prediction And Machine-Learning Audit",
        "",
        "The model layer is deliberately conservative. It does not accept in-sample fit as "
        "evidence. A predictive signal must improve naive baselines out of sample, show "
        "positive OOS R-squared, positive information coefficient and non-trivial "
        "directional accuracy. This follows the spirit of Gu-Kelly-Xiu and the "
        "overfitting literature.",
        "",
        _markdown_table(tables.get("ml_audit", [])),
        "",
        "## 10. Autocorrelation-Adjusted Performance Inference",
        "",
        "Daily returns are not guaranteed to be independent. A naive annualized Sharpe can "
        "therefore overstate evidence when serial correlation inflates the long-run variance "
        "of the mean. The table reports Lo-style autocorrelation-adjusted Sharpe ratios and "
        "Newey-West/HAC mean t-statistics for assets and available portfolio rules.",
        "",
        _markdown_table(tables.get("autocorrelation_adjusted_sharpe", [])),
        "",
        "## 11. Multiple Testing And Model Confidence",
        "",
        "Selecting the best historical asset or strategy creates data-snooping risk. The "
        "paper therefore adds a Bailey-Lopez de Prado style PSR/DSR approximation and a "
        "model-confidence table. White Reality Check, Hansen SPA and full Model Confidence "
        "Set bootstraps remain documented future work, so any favorable result remains "
        "research-only.",
        "",
        _markdown_table(tables.get("multiple_testing_adjustments", [])),
        "",
        _markdown_table(tables.get("model_confidence", [])),
        "",
        "## 12. Correlation, Diversification And Portfolio Construction",
        "",
        "Correlation is economically important because diversification is not about owning "
        "many tickers; it is about owning exposures that do not collapse together. The "
        "covariance matrix combines volatility and correlation and enters the Markowitz "
        "risk formula `sigma_p = sqrt(w' Sigma w)`.",
        "",
        "### 12.1 Correlation Matrix",
        "",
        _markdown_table(tables.get("correlation_matrix", [])),
        "",
        "### 12.2 Highest Correlation Pairs",
        "",
        _markdown_table(tables.get("correlation_pairs", [])),
        "",
        "### 12.3 Annual Regime Stability",
        "",
        "Diversification can disappear in stressed years. This annual diagnostic reports "
        "equal-weight performance and average pairwise correlation by calendar year, so the "
        "paper does not rely on one full-sample correlation matrix alone.",
        "",
        _markdown_table(tables.get("annual_regime_stability", [])),
        "",
        "### 12.4 Portfolio Weights And Concentration",
        "",
        _markdown_table(tables.get("portfolio_weights", [])),
        "",
        _markdown_table(tables.get("portfolio_diagnostics", [])),
        "",
        "### 12.5 Shrinkage Sensitivity",
        "",
        "Sample covariance matrices are noisy. The paper therefore includes a diagonal "
        "shrinkage sensitivity `Sigma_delta=(1-delta)Sigma+delta diag(Sigma)`. This is "
        "not claimed to be a full Ledoit-Wolf optimal shrinkage estimator; it is a robust "
        "diagnostic showing how allocations react when estimated correlations are reduced.",
        "",
        _markdown_table(tables.get("shrinkage_sensitivity", [])),
        "",
        "### 12.6 Covariance Shrinkage Comparison And Robustness",
        "",
        "The following diagnostics compare sample covariance against shrinkage covariance "
        "and quantify how much optimized weights move. Large shifts indicate estimation "
        "fragility rather than stable economic evidence.",
        "",
        _markdown_table(tables.get("covariance_shrinkage_comparison", [])),
        "",
        _markdown_table(tables.get("portfolio_robustness", [])),
        "",
        "## 13. Factor-Model And Execution-Cost Gaps",
        "",
        "Market-only beta is not enough to claim alpha. The factor table deliberately fails "
        "closed until Fama-French/Carhart/five-factor returns are loaded. Execution costs "
        "are also research proxies, not broker-calibrated estimates, because no order book, "
        "broker fills or intraday volume curve are available.",
        "",
        _markdown_table(tables.get("factor_model_gap_table", [])),
        "",
        _markdown_table(tables.get("execution_cost_sensitivity", [])),
        "",
        "## 14. Broker-Style Research Gates",
        "",
        "A broker system needs a final decision layer. In this project the layer is still "
        "research-only: it blocks live orders and labels what further evidence would be "
        "needed. This is intentionally stricter than a portfolio chart because model risk, "
        "data warnings and execution costs can invalidate attractive historical returns.",
        "",
        _markdown_table(tables.get("broker_research_gates", [])),
        "",
        _markdown_table(tables.get("decision_gate_evidence", [])),
        "",
        _markdown_table(tables.get("final_research_decision_table", [])),
        "",
        "## 15. Figures Produced By The Study",
        "",
        _figure_list(figure_paths or {}),
        "",
        "## 16. Critical Findings",
        "",
        _markdown_table(tables.get("critical_findings", [])),
        "",
        "## 17. Roadmap For A Stronger Institutional Study",
        "",
        _roadmap_text(),
        "",
        "## 18. Implementation Plan",
        "",
        _implementation_plan_text(),
        "",
        "## 19. Conclusion",
        "",
        _paper_conclusion(tables, source),
        "",
        "## Appendix A. Generated CSV And Figure Artifacts",
        "",
        _artifact_list(table_paths or {}, figure_paths or {}),
    ]
    return "\n".join(lines).strip() + "\n"


def _paper_abstract(study: dict[str, Any]) -> str:
    source = _mapping(study.get("source_terminal_report"))
    controls = _mapping(study.get("audit_controls"))
    tables = _mapping(study.get("tables"))
    assets = _asset_count(tables.get("asset_metrics", []))
    findings = len(tables.get("critical_findings", [])) if isinstance(
        tables.get("critical_findings"), list
    ) else 0
    return (
        f"This paper presents an auditable quantitative-finance study over {assets} equity "
        f"assets using source mode `{source.get('data_mode')}`. The study computes observed "
        "returns, drawdowns, CAPM-style benchmark sensitivity, VaR, Expected Shortfall, "
        "empirical CVaR-optimized portfolios, walk-forward predictive diagnostics, "
        "autocorrelation-adjusted Sharpe inference, regime-stability diagnostics, correlation "
        "matrices, portfolio weights, concentration diagnostics and covariance-shrinkage "
        "sensitivity. Strict controls "
        f"reject synthetic data by default (`strict_real_data={controls.get('strict_real_data')}`) "
        f"and the final review produced {findings} critical findings. The main result is "
        "that historical performance alone is not sufficient for broker promotion: data "
        "warnings, weak predictive evidence, drawdown risk and in-sample portfolio "
        "optimization materially limit operational use."
    )


def _literature_design_text() -> str:
    return (
        "The study translates recent quantitative-finance literature into implementable "
        "controls. Gu, Kelly and Xiu motivate strict out-of-sample validation for ML asset "
        "pricing. Bailey and Lopez de Prado motivate explicit overfitting controls and "
        "deflated performance interpretation. Fama-French and Carhart motivate benchmark "
        "and factor controls before calling a result alpha. Acerbi and Tasche motivate ES "
        "as a coherent tail-risk metric, and Rockafellar-Uryasev motivate optimizing the "
        "same tail-loss objective directly. Lo and Newey-West motivate autocorrelation-robust "
        "performance inference. Markowitz gives the mean-variance framework, while Ledoit-Wolf "
        "and DeMiguel show why naive sample covariance and optimized weights can be unstable. "
        "Almgren-Chriss and Gatheral motivate execution-cost and market-impact controls before "
        "translating a signal into an order."
    )


def _single_asset_results_text(rows: object) -> str:
    frame = pd.DataFrame(rows if isinstance(rows, list) else [])
    if frame.empty:
        return "No single-asset metrics were available."
    best_cagr = _row_with_extreme(frame, "cagr", largest=True)
    worst_drawdown = _row_with_extreme(frame, "max_drawdown", largest=False)
    highest_vol = _row_with_extreme(frame, "annualized_volatility", largest=True)
    return (
        "The single-asset table separates return from risk. A high CAGR is not sufficient "
        "if it arrives with extreme drawdowns or high volatility. In this run, the highest "
        f"CAGR row is `{best_cagr}`, the most severe drawdown row is `{worst_drawdown}`, "
        f"and the highest volatility row is `{highest_vol}`. These labels summarize the "
        "table; they are not recommendations."
    )


def _figure_list(paths: dict[str, str]) -> str:
    if not paths:
        return "No figures were generated."
    descriptions = {
        "normalized_prices": "Compares relative price growth after normalizing starts.",
        "cumulative_returns": "Shows total compounded return paths.",
        "correlation_heatmap": "Visualizes common movement and diversification risk.",
        "risk_return_scatter": "Places each asset in volatility-return space.",
        "drawdowns": "Shows historical peak-to-trough losses through time.",
        "efficient_frontier": "Shows the approximate long-only grid frontier.",
        "decision_scores": "Summarizes research-only decision scores.",
        "shrinkage_weight_comparison": "Compares sample-covariance and shrinkage weights.",
        "rolling_correlation_stability": "Shows whether diversification changes by regime.",
        "var_exceptions_through_time": "Counts displayed VaR breaches through time.",
        "exception_traffic_light": "Summarizes VaR exception rates by asset.",
        "sharpe_vs_deflated_sharpe": "Shows multiple-testing Sharpe haircut.",
        "ml_confidence_comparison": "Ranks model-confidence status by asset.",
        "portfolio_robustness_heatmap": "Shows optimized-weight fragility.",
        "execution_cost_sensitivity": "Shows research proxy transaction-cost sensitivity.",
        "final_decision_waterfall": "Counts final research-only decision states.",
    }
    return "\n".join(
        f"- `{name}`: `{path}`. {descriptions.get(name, 'Generated study figure.')}"
        for name, path in sorted(paths.items())
    )


def _roadmap_text() -> str:
    return "\n".join(
        [
            "1. **Data audit upgrade.** Add raw-response hashes, provider request snapshots, "
            "calendar-aware coverage, split/dividend audit and point-in-time metadata.",
            "2. **Factor-model upgrade.** Ingest Fama-French-Carhart and five-factor returns, "
            "then estimate residual alpha with t-statistics and HAC/Newey-West errors.",
            "3. **Cross-sectional ML upgrade.** Move from single-asset OHLCV models to a panel "
            "with point-in-time features, nested walk-forward validation and model-confidence "
            "sets.",
            "4. **Portfolio optimizer upgrade.** Replace grid diagnostics with tested convex "
            "CVaR, Ledoit-Wolf shrinkage, HRP, Black-Litterman and turnover-aware constraints.",
            "5. **Tail-risk upgrade.** Add EVT/POT, stress regimes, ES backtesting proxies and "
            "scenario attribution by asset contribution.",
            "6. **Execution upgrade.** Add intraday bars, bid-ask spreads, ADV capacity curves, "
            "broker fill logs and calibrated market-impact models.",
            "7. **Broker-readiness upgrade.** Keep live trading disabled until paper-trading, "
            "risk limits, kill switches, audit logs and compliance review are implemented.",
        ]
    )


def _implementation_plan_text() -> str:
    return "\n".join(
        [
            "1. Extend provider ingestion to store raw hashes and provider metadata per symbol.",
            "2. Add a `factors/` module with benchmark, SMB, HML, RMW, CMA and MOM alignment.",
            "3. Add `portfolio/shrinkage.py` with a tested Ledoit-Wolf estimator and PSD checks.",
            "4. Add `portfolio/cvar.py` with a Rockafellar-Uryasev linear-program formulation "
            "to replace the current auditable grid diagnostic.",
            "5. Add `validation/model_comparison.py` for SPA/MCS-style model comparison.",
            "6. Add `execution/impact.py` with calibrated spread, slippage and capacity curves.",
            "7. Add report sections that fail closed when required data are unavailable.",
        ]
    )


def _paper_conclusion(tables: dict[str, Any], source: dict[str, Any]) -> str:
    gates = pd.DataFrame(tables.get("broker_research_gates", []))
    ml = pd.DataFrame(tables.get("ml_audit", []))
    allowed = 0
    if not gates.empty and "live_order_allowed" in gates.columns:
        allowed = int((gates["live_order_allowed"] == True).sum())  # noqa: E712
    passed_ml = 0
    if not ml.empty and "strict_edge_validated" in ml.columns:
        passed_ml = int((ml["strict_edge_validated"] == True).sum())  # noqa: E712
    return (
        f"Using `{source.get('data_mode')}` data, the study finds that {passed_ml} assets "
        "passed strict predictive gates, but broker promotion remains blocked because the "
        f"research gate allowed {allowed} live orders. This is the correct conservative "
        "outcome for an audit-grade quant process: the presence of historical return, "
        "positive Sharpe or attractive optimized weights is not enough. A production-grade "
        "broker decision requires clean risk-free data, stronger out-of-sample evidence, "
        "factor-adjusted alpha, robust covariance estimation, tail-risk constraints and "
        "execution-cost calibration."
    )


def _artifact_list(table_paths: dict[str, str], figure_paths: dict[str, str]) -> str:
    lines = ["### Tables"]
    if table_paths:
        lines.extend(f"- `{name}`: `{path}`" for name, path in sorted(table_paths.items()))
    else:
        lines.append("No CSV table paths available.")
    lines.append("")
    lines.append("### Figures")
    if figure_paths:
        lines.extend(f"- `{name}`: `{path}`" for name, path in sorted(figure_paths.items()))
    else:
        lines.append("No figure paths available.")
    return "\n".join(lines)


def _asset_count(rows: object) -> int:
    if not isinstance(rows, list):
        return 0
    return len(
        {str(_mapping(row).get("asset_id")) for row in rows if _mapping(row).get("asset_id")}
    )


def _row_with_extreme(frame: pd.DataFrame, column: str, *, largest: bool) -> str:
    if column not in frame.columns or "asset_id" not in frame.columns:
        return "unavailable"
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.dropna().empty:
        return "unavailable"
    idx = values.idxmax() if largest else values.idxmin()
    return f"{frame.loc[idx, 'asset_id']} ({column}={values.loc[idx]:.4f})"


def _validate_terminal_report(report: dict[str, Any], *, strict_real_data: bool) -> None:
    if not isinstance(report, dict) or report.get("report_type") != "professional_quant_terminal":
        raise InstitutionalStudyError(
            "terminal_report must be a professional_quant_terminal JSON object."
        )
    if not isinstance(report.get("stocks"), dict) or not report["stocks"]:
        raise InstitutionalStudyError("terminal_report has no per-stock study payload.")
    mode = _data_mode(report)
    if strict_real_data and not mode.startswith("provider_"):
        raise InstitutionalStudyError(
            f"Strict study mode requires provider data, but terminal report mode is {mode!r}."
        )


def _data_mode(report: dict[str, Any]) -> str:
    data = _mapping(report.get("data"))
    provenance = _mapping(report.get("data_provenance"))
    return str(provenance.get("data_mode") or data.get("mode") or "UNKNOWN")


def _risk_free_daily(report: dict[str, Any]) -> float:
    provenance = _mapping(report.get("data_provenance"))
    annual = _safe_float(provenance.get("risk_free_rate_annual"))
    if annual is None:
        return 0.0
    return (1.0 + annual) ** (1.0 / 252.0) - 1.0


def _risk_free_annual(report: dict[str, Any]) -> float:
    provenance = _mapping(report.get("data_provenance"))
    annual = _safe_float(provenance.get("risk_free_rate_annual"))
    return float(annual) if annual is not None else 0.0


def _audit_controls(report: dict[str, Any], *, strict_real_data: bool) -> dict[str, Any]:
    mode = _data_mode(report)
    warnings = [str(item) for item in report.get("warnings", [])]
    return {
        "strict_real_data": strict_real_data,
        "source_data_mode": mode,
        "synthetic_rejected": strict_real_data,
        "research_only": True,
        "warnings_count": len(warnings),
        "has_provider_partial_failures": "PROVIDER_PARTIAL_SYMBOL_FAILURES" in warnings,
        "has_risk_free_proxy_warning": any(
            item.startswith("RISK_FREE_PROXY_") for item in warnings
        ),
        "no_broker_order_generated": True,
    }


def _study_design() -> list[dict[str, Any]]:
    return [
        {
            "block": "Ingestion and provenance",
            "purpose": "Prove what data were used before calculating returns.",
            "math_or_rule": (
                "available_at >= timestamp; positive finite OHLC; no synthetic in strict mode"
            ),
            "bibliography": "LopezDePrado2018; data-quality/MLOps checklist",
        },
        {
            "block": "Returns and compounding",
            "purpose": "Transform prices into measurable gains/losses.",
            "math_or_rule": "R_t = P_t/P_{t-1}-1; wealth_t = product(1+R_t)",
            "bibliography": "Standard empirical asset-pricing convention",
        },
        {
            "block": "Benchmark and factor proxy",
            "purpose": "Avoid calling beta exposure alpha.",
            "math_or_rule": "beta_i = Cov(R_i,R_m)/Var(R_m); active return = R_i-R_b",
            "bibliography": "Sharpe1964; FamaFrench1993; Carhart1997",
        },
        {
            "block": "Tail risk",
            "purpose": "Measure bad outcomes, not only average return.",
            "math_or_rule": "VaR_alpha(L)=quantile_alpha(L); ES_alpha=E[L|L>=VaR]",
            "bibliography": (
                "Artzner1999; AcerbiTasche2002; RockafellarUryasev2000; "
                "Kupiec1995; Christoffersen1998"
            ),
        },
        {
            "block": "Performance inference",
            "purpose": "Avoid overstating Sharpe evidence under serial correlation.",
            "math_or_rule": "SR_HAC = SR / sqrt(1 + 2 sum_k w_k rho_k)",
            "bibliography": "Lo2002; NeweyWest1987",
        },
        {
            "block": "Prediction validation",
            "purpose": "Reject forecasts that do not beat naive baselines out of sample.",
            "math_or_rule": "walk-forward split; OOS R2; IC; directional accuracy edge",
            "bibliography": "GuKellyXiu2020; BaileyLopezDePrado2014; Hansen2005",
        },
        {
            "block": "Portfolio construction",
            "purpose": (
                "Use correlations and risk to combine assets instead of ranking tickers alone."
            ),
            "math_or_rule": "mu_p=w'mu; sigma_p=sqrt(w'Sigma w); long-only constraints",
            "bibliography": "Markowitz1952; LedoitWolf2004; DeMiguel2009",
        },
        {
            "block": "Regime stability",
            "purpose": "Detect whether diversification changes across calendar years.",
            "math_or_rule": "annual average pairwise correlation and equal-weight drawdown",
            "bibliography": "AngTimmermann2012; LopezDePrado2018",
        },
    ]


def _plain_language_explanation() -> str:
    return (
        "El estudio empieza con precios reales normalizados y auditados. Despues convierte "
        "precios en retornos porque el retorno permite comparar acciones con precios distintos. "
        "A continuacion mide riesgo normal y riesgo de cola, compara cada activo contra un "
        "benchmark, prueba si los modelos predictivos superan reglas simples fuera de muestra, "
        "y finalmente mira correlaciones para construir carteras. El objetivo no es adivinar el "
        "futuro, sino decidir que evidencia historica es suficientemente limpia, robusta y "
        "economicamente interpretable para seguir investigando antes de cualquier integracion con "
        "un broker."
    )


def _provenance_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    provenance = _mapping(report.get("data_provenance"))
    data = _mapping(report.get("data"))
    provider_metadata = _mapping(
        provenance.get("provider_metadata") or data.get("provider_metadata")
    )
    return [
        {
            "data_mode": _data_mode(report),
            "provider": provenance.get("provider") or provider_metadata.get("provider"),
            "venue": provenance.get("venue") or provider_metadata.get("venue"),
            "frequency": provenance.get("frequency"),
            "start_timestamp": provenance.get("start_timestamp"),
            "end_timestamp": provenance.get("end_timestamp"),
            "aligned_observations": provenance.get("aligned_observations"),
            "benchmark": provenance.get("benchmark"),
            "risk_free_proxy": provenance.get("risk_free_proxy"),
            "risk_free_rate_annual": provenance.get("risk_free_rate_annual"),
            "successful_symbols": _join(provider_metadata.get("successful_symbols")),
            "failed_symbols": json.dumps(
                provider_metadata.get("failed_symbols", {}), sort_keys=True
            ),
            "warnings": _join(report.get("warnings", [])),
        }
    ]


def _wide_stock_series(stocks: dict[str, Any], series_key: str, value_key: str) -> pd.DataFrame:
    frames = []
    for symbol, payload in sorted(stocks.items()):
        rows = _mapping(payload).get(series_key, [])
        if not isinstance(rows, list) or not rows:
            continue
        frame = pd.DataFrame(rows)
        if "timestamp" not in frame.columns or value_key not in frame.columns:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame[str(symbol)] = pd.to_numeric(frame[value_key], errors="coerce")
        frames.append(frame.loc[:, ["timestamp", str(symbol)]].dropna())
    if not frames:
        return pd.DataFrame()
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on="timestamp", how="outer")
    return wide.sort_values("timestamp").set_index("timestamp")


def _sample_wide_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    sample = frame.head(max_rows).copy()
    sample.insert(0, "timestamp", [pd.Timestamp(index).isoformat() for index in sample.index])
    return sample.reset_index(drop=True)


def _asset_metrics_table(stocks: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, payload in sorted(stocks.items()):
        metrics = _mapping(_mapping(payload).get("metrics"))
        rows.append(
            {
                "asset_id": symbol,
                "observations": metrics.get("observations"),
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "annualized_volatility": metrics.get("annualized_volatility"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "beta_to_benchmark": metrics.get("beta_to_benchmark"),
                "jensen_alpha": metrics.get("jensen_alpha"),
                "treynor_ratio": metrics.get("treynor_ratio"),
                "skewness": metrics.get("skewness"),
                "kurtosis": metrics.get("kurtosis"),
            }
        )
    return pd.DataFrame(rows)


def _tail_risk_table(stocks: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, payload in sorted(stocks.items()):
        var_payload = _mapping(_mapping(payload).get("var"))
        for model_name in ("historical", "parametric_normal", "monte_carlo"):
            model = _mapping(var_payload.get(model_name))
            if not model:
                continue
            rows.append(
                {
                    "asset_id": symbol,
                    "model": model_name,
                    "var_95_loss": model.get("var"),
                    "expected_shortfall_95_loss": model.get("expected_shortfall"),
                    "loss_convention": var_payload.get("loss_sign_convention", "L_t = -R_t"),
                    "model_status": model.get("model_status"),
                }
            )
    return pd.DataFrame(rows)


def _ml_audit_table(stocks: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, payload in sorted(stocks.items()):
        stock = _mapping(payload)
        ml = _mapping(stock.get("ml_forecasting"))
        reliability = _mapping(stock.get("predictive_reliability_audit"))
        gates = _mapping(ml.get("approval_gates"))
        status = str(ml.get("status", ""))
        strict_edge = ml.get("strict_predictive_edge_validated")
        if strict_edge is None:
            strict_edge = gates.get("strict_predictive_edge_validated")
        if strict_edge is None:
            strict_edge = status == "MODEL_EDGE_PASSED_STRICT_DIAGNOSTIC_GATES"
        rows.append(
            {
                "asset_id": symbol,
                "status": status,
                "audit_rating": reliability.get("rating"),
                "oos_r_squared": ml.get("oos_r_squared"),
                "information_coefficient": ml.get("information_coefficient"),
                "directional_accuracy": ml.get("directional_accuracy"),
                "baseline_directional_accuracy": ml.get("baseline_directional_accuracy"),
                "rmse": ml.get("rmse"),
                "baseline_rmse": ml.get("baseline_rmse"),
                "strict_edge_validated": bool(strict_edge),
                "failed_criteria": _join(reliability.get("failed_criteria", [])),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(stocks: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, payload in sorted(stocks.items()):
        signal = _mapping(_mapping(payload).get("decision_signal"))
        rows.append(
            {
                "asset_id": symbol,
                "research_signal": signal.get("signal"),
                "confidence": signal.get("confidence"),
                "score": signal.get("score"),
                "suggested_research_action": signal.get("suggested_research_action"),
                "limits_triggered": _join(signal.get("limits_triggered", [])),
                "positive_drivers": _join(signal.get("drivers_positive", []), limit=3),
                "negative_drivers": _join(signal.get("drivers_negative", []), limit=3),
            }
        )
    return pd.DataFrame(rows)


def _broker_gate_table(stocks: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, payload in sorted(stocks.items()):
        signal = _mapping(_mapping(payload).get("decision_signal"))
        limits = {str(item) for item in signal.get("limits_triggered", [])}
        confidence = str(signal.get("confidence", "LOW"))
        research_signal = str(signal.get("signal", "INSUFFICIENT_DATA"))
        if confidence != "HIGH":
            gate = "NO_LIVE_ORDER_LOW_CONFIDENCE"
        elif limits.intersection(
            {
                "MAX_DRAWDOWN_LIMIT_BREACHED",
                "VAR_95_LIMIT_BREACHED",
                "ES_95_LIMIT_BREACHED",
            }
        ):
            gate = "RISK_REVIEW_REQUIRED"
        elif "PREDICTIVE_EDGE_NOT_VALIDATED" in limits:
            gate = "NO_PROMOTION_PREDICTIVE_EDGE_NOT_VALIDATED"
        elif research_signal == "FAVORABLE":
            gate = "PAPER_ONLY_CANDIDATE"
        else:
            gate = "WATCHLIST_RESEARCH_ONLY"
        rows.append(
            {
                "asset_id": symbol,
                "research_signal": research_signal,
                "confidence": confidence,
                "broker_research_gate": gate,
                "live_order_allowed": False,
                "reason": _broker_gate_reason(gate),
            }
        )
    return pd.DataFrame(rows)


def _broker_gate_reason(gate: str) -> str:
    reasons = {
        "NO_LIVE_ORDER_LOW_CONFIDENCE": "Evidence is not strong enough for broker integration.",
        "RISK_REVIEW_REQUIRED": "Tail risk or drawdown breached configured research limits.",
        "NO_PROMOTION_PREDICTIVE_EDGE_NOT_VALIDATED": "Forecast did not pass strict OOS gates.",
        "PAPER_ONLY_CANDIDATE": "Only eligible for future paper-trading validation.",
        "WATCHLIST_RESEARCH_ONLY": "Keep monitoring without operational action.",
    }
    return reasons.get(gate, "Research-only gate.")


def _matrix_frame(rows: object) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "asset_id" not in frame.columns:
        return pd.DataFrame()
    frame = frame.set_index("asset_id")
    return frame.apply(pd.to_numeric, errors="coerce")


def _matrix_to_table(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    frame = matrix.copy()
    frame.insert(0, "asset_id", [str(index) for index in frame.index])
    return frame.reset_index(drop=True)


def _portfolio_weight_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    weight_sources = _portfolio_weight_sources(report)
    for portfolio_name, weights in weight_sources.items():
        for asset_id, weight in sorted(weights.items()):
            rows.append(
                {
                    "portfolio": portfolio_name,
                    "asset_id": asset_id,
                    "weight": weight,
                    "long_only": True,
                }
            )
    return pd.DataFrame(rows)


def _portfolio_weight_sources(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    portfolio = _mapping(report.get("portfolio"))
    optimization = _mapping(report.get("optimization"))
    raw_sources = {
        "equal_weight": _mapping(_mapping(portfolio.get("equal_weight")).get("weights")),
        "inverse_volatility": _mapping(
            _mapping(portfolio.get("inverse_volatility")).get("weights")
        ),
        "min_variance": _mapping(_mapping(optimization.get("min_variance")).get("weights")),
        "max_sharpe": _mapping(_mapping(optimization.get("max_sharpe")).get("weights")),
    }
    clean_sources: dict[str, dict[str, float]] = {}
    for name, weights in raw_sources.items():
        clean = {
            str(asset): float(weight) for asset, weight in weights.items() if weight is not None
        }
        total = sum(clean.values())
        if total > 0:
            clean_sources[name] = {asset: weight / total for asset, weight in clean.items()}
    return clean_sources


def _portfolio_diagnostics_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, weights in _portfolio_weight_sources(report).items():
        values = np.array(list(weights.values()), dtype=float)
        hhi = float(np.sum(values**2)) if len(values) else float("nan")
        rows.append(
            {
                "portfolio": name,
                "nonzero_assets": int(np.sum(values > 1e-12)),
                "max_weight": float(np.max(values)) if len(values) else None,
                "min_positive_weight": float(np.min(values[values > 1e-12]))
                if np.any(values > 1e-12)
                else None,
                "herfindahl_hirschman_index": hhi,
                "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0 else None,
                "concentration_warning": bool(np.max(values) > 0.35) if len(values) else None,
            }
        )
    return pd.DataFrame(rows)


def _portfolio_tail_risk_table(returns: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    rows = []
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean_returns.empty:
        return pd.DataFrame()
    for name, weights in _portfolio_weight_sources(report).items():
        aligned_weights = pd.Series(weights, dtype=float).reindex(clean_returns.columns).fillna(0.0)
        if aligned_weights.sum() <= 0:
            continue
        aligned_weights = aligned_weights / aligned_weights.sum()
        series = clean_returns.mul(aligned_weights, axis=1).sum(axis=1)
        wealth = (1.0 + series).cumprod()
        losses = -series
        for alpha in (0.95, 0.99):
            var = float(losses.quantile(alpha))
            tail = losses[losses >= var]
            rows.append(
                {
                    "portfolio": name,
                    "alpha": alpha,
                    "observations": int(len(series)),
                    "annualized_mean_return": float(series.mean() * 252.0),
                    "annualized_volatility": float(series.std(ddof=1) * math.sqrt(252.0)),
                    "historical_var_loss": var,
                    "historical_expected_shortfall_loss": float(tail.mean())
                    if len(tail)
                    else None,
                    "max_drawdown": _max_drawdown_from_wealth(wealth),
                    "loss_convention": "L_t = -R_t",
                }
            )
    return pd.DataFrame(rows)


def _tail_risk_backtesting_table(returns: pd.DataFrame, stocks: dict[str, Any]) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        return pd.DataFrame()
    rows = []
    for symbol, payload in sorted(stocks.items()):
        if str(symbol) not in clean.columns:
            continue
        historical = _mapping(_mapping(_mapping(payload).get("var")).get("historical"))
        var_95 = _safe_float(historical.get("var"))
        if var_95 is None:
            continue
        result = backtest_var(clean[str(symbol)], var_95, alpha=0.95, label="historical_var_95")
        result["asset_id"] = str(symbol)
        result["var_95_loss"] = var_95
        result["research_interpretation"] = _tail_backtest_interpretation(result)
        rows.append(result)
    return pd.DataFrame(rows)


def _tail_backtest_interpretation(row: dict[str, Any]) -> str:
    if row.get("kupiec_status") == "PASS" and row.get("christoffersen_status") == "PASS":
        return "VaR exceptions are not statistically rejected in this in-sample diagnostic."
    return "VaR model requires review before any broker or paper-trading promotion."


def _var_exception_table(
    returns: pd.DataFrame,
    tail_risk: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or tail_risk.empty:
        return pd.DataFrame()
    rows = []
    historical = tail_risk[tail_risk.get("model") == "historical"]
    for _, risk_row in historical.iterrows():
        symbol = str(risk_row.get("asset_id"))
        if symbol not in clean.columns:
            continue
        var_95 = _safe_float(risk_row.get("var_95_loss"))
        if var_95 is None:
            continue
        exceptions = var_exceptions(clean[symbol], var_95)
        losses = -clean[symbol].reindex(exceptions.index)
        for timestamp, is_exception in exceptions.items():
            if not bool(is_exception):
                continue
            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamp).isoformat(),
                    "asset_id": symbol,
                    "loss": float(losses.loc[timestamp]),
                    "var_95_loss": var_95,
                    "excess_loss_over_var": float(losses.loc[timestamp] - var_95),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["timestamp", "asset_id"]).head(max_rows)


def _cvar_optimized_portfolios_table(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return pd.DataFrame()
    grid = long_only_weight_grid(tuple(str(column) for column in clean.columns), step=0.05)
    effective_step = float(grid.attrs.get("effective_step", 0.05))
    rows = []
    for alpha in (0.95, 0.99):
        best: dict[str, Any] | None = None
        for _, weight_row in grid.iterrows():
            weights = weight_row.reindex(clean.columns).astype(float)
            portfolio_returns = clean.mul(weights, axis=1).sum(axis=1)
            losses = -portfolio_returns
            var = float(losses.quantile(alpha))
            tail = losses[losses >= var]
            es = float(tail.mean()) if len(tail) else float("nan")
            annual_return = float(portfolio_returns.mean() * 252.0)
            annual_vol = float(portfolio_returns.std(ddof=1) * math.sqrt(252.0))
            wealth = (1.0 + portfolio_returns).cumprod()
            hhi = float(np.sum(weights.to_numpy(dtype=float) ** 2))
            candidate = {
                "objective": "minimize_empirical_expected_shortfall",
                "alpha": alpha,
                "observations": int(len(portfolio_returns)),
                "historical_var_loss": var,
                "historical_expected_shortfall_loss": es,
                "annualized_mean_return": annual_return,
                "annualized_volatility": annual_vol,
                "max_drawdown": _max_drawdown_from_wealth(wealth),
                "max_weight": float(weights.max()),
                "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0 else None,
                "weights_json": json.dumps(weights.to_dict(), sort_keys=True),
                "method": f"long_only_grid_search_effective_step_{effective_step:g}",
                "literature": "RockafellarUryasev2000; AcerbiTasche2002",
                "limitation": (
                    "Empirical-grid CVaR optimizer; no convex solver, turnover, liquidity, "
                    "or walk-forward retraining yet."
                ),
            }
            if best is None or es < float(best["historical_expected_shortfall_loss"]):
                best = candidate
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows)


def _autocorrelation_adjusted_sharpe_table(
    returns: pd.DataFrame,
    report: dict[str, Any],
) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        return pd.DataFrame()
    rf_daily = _risk_free_daily(report)
    series_map: dict[str, pd.Series] = {
        str(column): clean[str(column)] for column in clean.columns
    }
    for portfolio_name, weights in _portfolio_weight_sources(report).items():
        aligned_weights = pd.Series(weights, dtype=float).reindex(clean.columns).fillna(0.0)
        if aligned_weights.sum() > 0:
            aligned_weights = aligned_weights / aligned_weights.sum()
            series_map[f"portfolio:{portfolio_name}"] = clean.mul(aligned_weights, axis=1).sum(
                axis=1
            )
    rows = []
    for name, series in sorted(series_map.items()):
        excess = (series.astype(float) - rf_daily).dropna()
        if len(excess) < 30 or float(excess.std(ddof=1)) == 0.0:
            continue
        naive_sharpe = float(excess.mean() / excess.std(ddof=1) * math.sqrt(252.0))
        for lag in (5, 21):
            adjustment = _hac_sharpe_adjustment(excess, lag)
            rows.append(
                {
                    "series": name,
                    "observations": int(len(excess)),
                    "lag_days": lag,
                    "naive_annualized_sharpe": naive_sharpe,
                    "autocorrelation_variance_inflation": adjustment["variance_inflation"],
                    "lo_adjusted_annualized_sharpe": adjustment["adjusted_sharpe"],
                    "hac_mean_t_stat": adjustment["hac_mean_t_stat"],
                    "autocorrelations_json": json.dumps(
                        adjustment["autocorrelations"], sort_keys=True
                    ),
                    "literature": "Lo2002; NeweyWest1987",
                }
            )
    return pd.DataFrame(rows)


def _hac_sharpe_adjustment(series: pd.Series, lag: int) -> dict[str, Any]:
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    centered = clean - mean
    gamma0 = float(np.mean(centered.to_numpy(dtype=float) ** 2))
    variance_inflation = 1.0
    hac_long_run_variance = gamma0
    autocorrelations: dict[str, float] = {}
    for k in range(1, min(lag, n - 1) + 1):
        left = centered.iloc[k:].to_numpy(dtype=float)
        right = centered.iloc[:-k].to_numpy(dtype=float)
        gamma_k = float(np.mean(left * right)) if len(left) else 0.0
        rho_k = gamma_k / gamma0 if gamma0 > 0 else 0.0
        weight = 1.0 - k / (lag + 1.0)
        variance_inflation += 2.0 * weight * rho_k
        hac_long_run_variance += 2.0 * weight * gamma_k
        autocorrelations[f"lag_{k}"] = float(rho_k)
    variance_inflation = max(float(variance_inflation), 1e-12)
    adjusted_sharpe = float(mean / std * math.sqrt(252.0 / variance_inflation))
    hac_se_mean = math.sqrt(max(hac_long_run_variance, 1e-18) / n)
    hac_t = mean / hac_se_mean if hac_se_mean > 0 else None
    return {
        "variance_inflation": variance_inflation,
        "adjusted_sharpe": adjusted_sharpe,
        "hac_mean_t_stat": hac_t,
        "autocorrelations": autocorrelations,
    }


def _annual_regime_stability_table(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return pd.DataFrame()
    rows = []
    for year, frame in clean.groupby(clean.index.year):
        if len(frame) < 30:
            continue
        corr = frame.corr()
        pair_values = _upper_triangle_values(corr)
        equal_weight_returns = frame.mean(axis=1)
        wealth = (1.0 + equal_weight_returns).cumprod()
        rows.append(
            {
                "year": int(year),
                "observations": int(len(frame)),
                "equal_weight_return": float((1.0 + equal_weight_returns).prod() - 1.0),
                "equal_weight_annualized_volatility": float(
                    equal_weight_returns.std(ddof=1) * math.sqrt(252.0)
                ),
                "equal_weight_max_drawdown": _max_drawdown_from_wealth(wealth),
                "average_pairwise_correlation": float(np.mean(pair_values)),
                "max_pairwise_correlation": float(np.max(pair_values)),
                "min_pairwise_correlation": float(np.min(pair_values)),
                "regime_comment": _regime_comment(float(np.mean(pair_values))),
            }
        )
    return pd.DataFrame(rows)


def _upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    indices = np.triu_indices_from(values, k=1)
    clean = values[indices]
    clean = clean[np.isfinite(clean)]
    return clean if clean.size else np.array([float("nan")])


def _regime_comment(avg_corr: float) -> str:
    if not np.isfinite(avg_corr):
        return "insufficient finite correlations"
    if avg_corr >= 0.60:
        return "high common regime; diversification likely weak"
    if avg_corr >= 0.40:
        return "moderate common regime"
    return "more diversified regime"


def _covariance_shrinkage_comparison_table(
    returns: pd.DataFrame,
    report: dict[str, Any],
) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return pd.DataFrame()
    try:
        return covariance_shrinkage_comparison(
            clean,
            risk_free_rate_annual=_risk_free_annual(report),
            grid_step=0.05,
        )
    except ValueError:
        return pd.DataFrame()


def _portfolio_robustness_table(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return pd.DataFrame()
    try:
        return portfolio_robustness_summary(clean, grid_step=0.05)
    except ValueError:
        return pd.DataFrame()


def _factor_model_gap_table(stocks: dict[str, Any]) -> pd.DataFrame:
    return build_factor_model_gap_table(tuple(sorted(str(symbol) for symbol in stocks)))


def _execution_cost_sensitivity_table(
    report: dict[str, Any],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    sources = _portfolio_weight_sources(report)
    if "equal_weight" not in sources:
        return pd.DataFrame()
    target_names = [name for name in ("min_variance", "max_sharpe") if name in sources]
    if not target_names:
        return pd.DataFrame()
    stocks = _mapping(report.get("stocks"))
    adv_by_asset = _adv_by_asset(stocks)
    volatility_by_asset = _volatility_by_asset(stocks, returns)
    rows = []
    old_weights = pd.Series(sources["equal_weight"], dtype=float)
    for target_name in target_names:
        new_weights = pd.Series(sources[target_name], dtype=float)
        table = execution_cost_sensitivity(
            old_weights,
            new_weights,
            notional=1_000_000.0,
            spread_bps=5.0,
            impact_coefficient=0.10,
            adv_by_asset=adv_by_asset,
            volatility_by_asset=volatility_by_asset,
        )
        table.insert(0, "rebalance_scenario", f"equal_weight_to_{target_name}")
        table["notional_assumption"] = 1_000_000.0
        table["calibration_status"] = "RESEARCH_PROXY_NOT_BROKER_CALIBRATED"
        rows.extend(table.to_dict(orient="records"))
    return pd.DataFrame(rows)


def _adv_by_asset(stocks: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for symbol, payload in sorted(stocks.items()):
        stock = _mapping(payload)
        execution = _mapping(stock.get("execution_cost_study"))
        momentum = _mapping(stock.get("momentum_liquidity_study"))
        adv = _safe_float(
            execution.get("average_daily_dollar_volume_20d")
            or momentum.get("average_dollar_volume_20d")
        )
        if adv is not None and adv > 0:
            values[str(symbol)] = adv
    return values


def _volatility_by_asset(stocks: dict[str, Any], returns: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for symbol, payload in sorted(stocks.items()):
        metrics = _mapping(_mapping(payload).get("metrics"))
        volatility = _safe_float(metrics.get("annualized_volatility"))
        if volatility is None and str(symbol) in returns.columns:
            volatility = float(returns[str(symbol)].std(ddof=1) * math.sqrt(252.0))
        if volatility is not None and volatility > 0:
            values[str(symbol)] = volatility
    return values


def _decision_gate_evidence_table(
    report: dict[str, Any],
    asset_metrics: pd.DataFrame,
    ml_audit: pd.DataFrame,
    tail_backtesting: pd.DataFrame,
    model_confidence: pd.DataFrame,
) -> pd.DataFrame:
    data_warnings = bool(report.get("warnings"))
    risk_free_available = not _has_risk_free_proxy_warning(report)
    rows = []
    for asset_id in sorted(str(item) for item in asset_metrics.get("asset_id", [])):
        metric = _row_by_asset(asset_metrics, asset_id)
        ml = _row_by_asset(ml_audit, asset_id)
        tail = _row_by_asset(tail_backtesting, asset_id)
        confidence = _row_by_asset(model_confidence, asset_id)
        severe_drawdown = _safe_float(metric.get("max_drawdown"))
        severe = bool(severe_drawdown is not None and severe_drawdown < -0.35)
        tail_passed = _tail_backtest_passed(tail)
        rows.append(
            {
                "asset_id": asset_id,
                "provider_warnings_present": data_warnings,
                "risk_free_proxy_available": risk_free_available,
                "strict_predictive_edge_validated": bool(ml.get("strict_edge_validated")),
                "model_confidence_status": confidence.get("model_confidence_status"),
                "severe_drawdown_flag": severe,
                "tail_backtest_passed": tail_passed,
                "broker_promotion_blocked": True,
                "why_blocked": _join(
                    _blocking_reasons(
                        data_warnings=data_warnings,
                        risk_free_available=risk_free_available,
                        predictive_edge=bool(ml.get("strict_edge_validated")),
                        severe_drawdown=severe,
                        tail_passed=tail_passed,
                        model_confidence_status=confidence.get("model_confidence_status"),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _final_research_decision_table(
    report: dict[str, Any],
    asset_metrics: pd.DataFrame,
    ml_audit: pd.DataFrame,
    tail_backtesting: pd.DataFrame,
    model_confidence: pd.DataFrame,
) -> pd.DataFrame:
    data_warnings = bool(report.get("warnings"))
    risk_free_available = not _has_risk_free_proxy_warning(report)
    rows = []
    for asset_id in sorted(str(item) for item in asset_metrics.get("asset_id", [])):
        metric = _row_by_asset(asset_metrics, asset_id)
        ml = _row_by_asset(ml_audit, asset_id)
        tail = _row_by_asset(tail_backtesting, asset_id)
        confidence = _row_by_asset(model_confidence, asset_id)
        drawdown = _safe_float(metric.get("max_drawdown"))
        severe_drawdown = bool(drawdown is not None and drawdown < -0.35)
        decision = final_research_decision(
            data_warnings=data_warnings,
            risk_free_proxy_available=risk_free_available,
            predictive_edge_validated=bool(ml.get("strict_edge_validated")),
            severe_drawdown=severe_drawdown,
            tail_backtest_passed=_tail_backtest_passed(tail),
            model_confidence_status=str(confidence.get("model_confidence_status")),
        )
        decision["asset_id"] = asset_id
        decision["confidence"] = confidence.get("model_confidence_status")
        decision["main_positive_drivers"] = _positive_drivers(metric, ml, tail, confidence)
        decision["main_negative_drivers"] = _join(
            _blocking_reasons(
                data_warnings=data_warnings,
                risk_free_available=risk_free_available,
                predictive_edge=bool(ml.get("strict_edge_validated")),
                severe_drawdown=severe_drawdown,
                tail_passed=_tail_backtest_passed(tail),
                model_confidence_status=confidence.get("model_confidence_status"),
            )
        )
        decision["required_improvements"] = (
            "Resolve provider/risk-free warnings; add factor data; validate ML persistence; "
            "calibrate execution costs; run paper-trading controls before any broker promotion."
        )
        rows.append(decision)
    return pd.DataFrame(rows)


def _row_by_asset(frame: pd.DataFrame, asset_id: str) -> dict[str, Any]:
    if frame.empty or "asset_id" not in frame.columns:
        return {}
    matches = frame[frame["asset_id"].astype(str) == str(asset_id)]
    if matches.empty:
        return {}
    return _mapping(matches.iloc[0].to_dict())


def _has_risk_free_proxy_warning(report: dict[str, Any]) -> bool:
    return any(str(item).startswith("RISK_FREE_PROXY_") for item in report.get("warnings", []))


def _tail_backtest_passed(row: dict[str, Any]) -> bool | None:
    if not row:
        return None
    if row.get("kupiec_status") == "PASS" and row.get("christoffersen_status") == "PASS":
        return True
    return False


def _blocking_reasons(
    *,
    data_warnings: bool,
    risk_free_available: bool,
    predictive_edge: bool,
    severe_drawdown: bool,
    tail_passed: bool | None,
    model_confidence_status: object,
) -> list[str]:
    reasons = []
    if data_warnings:
        reasons.append("provider warnings/fallbacks present")
    if not risk_free_available:
        reasons.append("risk-free proxy unavailable")
    if not predictive_edge:
        reasons.append("strict predictive edge not validated")
    if severe_drawdown:
        reasons.append("severe historical drawdown")
    if tail_passed is False:
        reasons.append("VaR backtest review required")
    if model_confidence_status != "MODEL_CONFIDENCE_ACCEPTED":
        reasons.append("model confidence not accepted")
    return reasons


def _positive_drivers(
    metric: dict[str, Any],
    ml: dict[str, Any],
    tail: dict[str, Any],
    confidence: dict[str, Any],
) -> str:
    drivers = []
    cagr = _safe_float(metric.get("cagr"))
    sharpe = _safe_float(metric.get("sharpe_ratio"))
    if cagr is not None and cagr > 0:
        drivers.append(f"positive historical CAGR {cagr:.2%}")
    if sharpe is not None and sharpe > 1.0:
        drivers.append(f"historical Sharpe above 1 ({sharpe:.2f})")
    if bool(ml.get("strict_edge_validated")):
        drivers.append("strict OOS predictive gate passed")
    if _tail_backtest_passed(tail):
        drivers.append("VaR exception diagnostics passed")
    if confidence.get("model_confidence_status") == "MODEL_CONFIDENCE_ACCEPTED":
        drivers.append("model confidence accepted")
    return _join(drivers) if drivers else "No sufficient positive driver for promotion."


def _state_of_art_gap_analysis_table() -> list[dict[str, Any]]:
    return [
        {
            "area": "risk_free_proxy",
            "status": "BLOCKING_GAP",
            "literature": "Sharpe1964; Lo2002",
            "required_improvement": "Ingest reliable Treasury/bill curve or fixed daily RF series.",
            "implement_now": "DOCUMENT_AND_BLOCK",
        },
        {
            "area": "factor_models",
            "status": "FACTOR_DATA_REQUIRED",
            "literature": "FamaFrench1993; Carhart1997; FamaFrench2015",
            "required_improvement": "Load point-in-time factor returns before alpha claims.",
            "implement_now": "SCAFFOLD_FAIL_CLOSED",
        },
        {
            "area": "tail_risk_backtesting",
            "status": "IMPLEMENTED_NOW",
            "literature": "Kupiec1995; Christoffersen1998",
            "required_improvement": "Expose VaR exceptions and coverage tests in final paper.",
            "implement_now": "YES",
        },
        {
            "area": "multiple_testing",
            "status": "PARTIAL_IMPLEMENTATION_DSR_APPROXIMATION",
            "literature": "BaileyLopezPrado2014; White2000; Hansen2005",
            "required_improvement": "Add full White Reality Check, SPA and MCS bootstrap later.",
            "implement_now": "PARTIAL",
        },
        {
            "area": "portfolio_robustness",
            "status": "IMPLEMENTED_NOW",
            "literature": "LedoitWolf2004; DeMiguel2009",
            "required_improvement": "Compare sample vs shrinkage weights and concentration.",
            "implement_now": "YES",
        },
        {
            "area": "execution_costs",
            "status": "RESEARCH_PROXY_NOT_BROKER_CALIBRATED",
            "literature": "AlmgrenChriss2001; BertsimasLo1998; Gatheral2010",
            "required_improvement": (
                "Calibrate spreads, impact and capacity from broker/intraday data."
            ),
            "implement_now": "PARTIAL",
        },
    ]


def _implementation_roadmap_table() -> list[dict[str, Any]]:
    return [
        {
            "priority": "HIGH",
            "module": "risk_free_curve",
            "action": "Add reliable Treasury/bill curve ingestion and daily RF alignment.",
            "blocks": "Sharpe, Treynor, Jensen alpha and broker promotion.",
        },
        {
            "priority": "HIGH",
            "module": "factor_models",
            "action": "Load Fama-French/Carhart/five-factor returns from external CSV/provider.",
            "blocks": "Factor-adjusted alpha and academic publication claims.",
        },
        {
            "priority": "HIGH",
            "module": "execution",
            "action": (
                "Add bid-ask, intraday volume curves, fills, slippage and impact calibration."
            ),
            "blocks": "Paper/live broker workflow and capacity estimates.",
        },
        {
            "priority": "MEDIUM",
            "module": "model_selection",
            "action": "Implement White Reality Check, Hansen SPA and Model Confidence Set.",
            "blocks": "Multiple-testing robust model promotion.",
        },
        {
            "priority": "MEDIUM",
            "module": "portfolio_optimizer",
            "action": "Replace grid CVaR with convex LP and turnover-aware constraints.",
            "blocks": "Institutional optimizer deployment.",
        },
    ]


def _correlation_pair_table(correlation: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if correlation.empty:
        return pd.DataFrame()
    rows = []
    assets = [str(asset) for asset in correlation.index]
    for left_index, left_asset in enumerate(assets):
        for right_asset in assets[left_index + 1 :]:
            value = _safe_float(correlation.loc[left_asset, right_asset])
            if value is None:
                continue
            rows.append(
                {
                    "asset_a": left_asset,
                    "asset_b": right_asset,
                    "correlation": value,
                    "abs_correlation": abs(value),
                    "diversification_comment": _correlation_comment(value),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("abs_correlation", ascending=False).head(max_rows)


def _correlation_comment(value: float) -> str:
    if value >= 0.75:
        return "high common-movement risk"
    if value >= 0.50:
        return "moderate common-movement risk"
    if value >= 0.25:
        return "partial diversification"
    return "stronger diversification potential"


def _shrinkage_sensitivity_table(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return pd.DataFrame()
    mu = clean.mean() * 252.0
    sample_cov = clean.cov() * 252.0
    diagonal_target = pd.DataFrame(
        np.diag(np.diag(sample_cov.to_numpy(dtype=float))),
        index=sample_cov.index,
        columns=sample_cov.columns,
    )
    rows = []
    for shrinkage_delta in (0.0, 0.25, 0.5, 0.75, 1.0):
        shrunk_cov = (1.0 - shrinkage_delta) * sample_cov + shrinkage_delta * diagonal_target
        weights = _long_only_min_variance_weights(shrunk_cov)
        if weights.empty:
            continue
        variance = float(weights.to_numpy(dtype=float) @ shrunk_cov.to_numpy(dtype=float) @ weights)
        expected_return = float(weights.dot(mu.reindex(weights.index)))
        volatility = math.sqrt(max(variance, 0.0))
        sharpe_no_rf = expected_return / volatility if volatility > 0 else None
        hhi = float(np.sum(weights.to_numpy(dtype=float) ** 2))
        rows.append(
            {
                "shrinkage_delta": shrinkage_delta,
                "target": "diagonal_covariance_target",
                "annualized_return_estimate": expected_return,
                "annualized_volatility_estimate": volatility,
                "sharpe_no_risk_free": sharpe_no_rf,
                "max_weight": float(weights.max()),
                "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0 else None,
                "weights_json": json.dumps(weights.to_dict(), sort_keys=True),
                "method_warning": (
                    "Sensitivity analysis only; this is not an optimal Ledoit-Wolf shrinkage "
                    "intensity estimator."
                ),
            }
        )
    return pd.DataFrame(rows)


def _long_only_min_variance_weights(covariance: pd.DataFrame) -> pd.Series:
    assets = tuple(str(asset) for asset in covariance.index)
    grid = long_only_weight_grid(assets, step=0.05, max_weight=1.0)
    if grid.empty:
        return pd.Series(dtype=float)
    sigma = covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
    best_variance = float("inf")
    best_weights: pd.Series | None = None
    for _, row in grid.iterrows():
        weights = row.reindex(assets).astype(float)
        variance = float(weights.to_numpy(dtype=float) @ sigma @ weights.to_numpy(dtype=float))
        if np.isfinite(variance) and variance < best_variance:
            best_variance = variance
            best_weights = weights
    if best_weights is None:
        return pd.Series(dtype=float)
    return best_weights / float(best_weights.sum())


def _max_drawdown_from_wealth(wealth: pd.Series) -> float | None:
    clean = wealth.astype(float).dropna()
    if clean.empty:
        return None
    running_max = clean.cummax()
    drawdowns = clean / running_max - 1.0
    return float(drawdowns.min())


def _frontier_table(optimization: dict[str, Any], max_rows: int) -> pd.DataFrame:
    rows = []
    frontier = optimization.get("frontier", [])
    frontier_rows = frontier[:max_rows] if isinstance(frontier, list) else []
    for row in frontier_rows:
        item = _mapping(row)
        rows.append(
            {
                "annualized_return": item.get("annualized_return"),
                "annualized_volatility": item.get("annualized_volatility"),
                "sharpe_ratio": item.get("sharpe_ratio"),
                "weights_json": json.dumps(item.get("weights", {}), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _critical_findings(report: dict[str, Any], stocks: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    mode = _data_mode(report)
    warnings = [str(item) for item in report.get("warnings", [])]
    if not mode.startswith("provider_"):
        findings.append(
            _finding(
                "CRITICAL",
                "Synthetic data mode detected",
                "Conclusions from synthetic data are not market evidence.",
                (
                    "Run provider ingestion successfully and reject synthetic fallback for final "
                    "studies."
                ),
            )
        )
    if warnings:
        findings.append(
            _finding(
                "HIGH",
                "Provider or data warnings are present",
                _join(warnings),
                (
                    "Resolve failed symbols, fallback substitutions and risk-free proxy warnings "
                    "before promotion."
                ),
            )
        )
    if any(item.startswith("RISK_FREE_PROXY_") for item in warnings):
        findings.append(
            _finding(
                "HIGH",
                "Risk-free proxy unavailable",
                "Sharpe, Jensen alpha, Treynor and decision scores use a fallback rate.",
                "Ingest a reliable Treasury/bill curve or fixed daily risk-free series.",
            )
        )
    ml_table = _ml_audit_table(stocks)
    if not ml_table.empty:
        validated = int((ml_table.get("strict_edge_validated") == True).sum())  # noqa: E712
        if validated < len(ml_table):
            findings.append(
                _finding(
                    "HIGH",
                    "Predictive edge is not broadly validated",
                    f"Only {validated}/{len(ml_table)} assets passed strict predictive gates.",
                    (
                        "Do not convert forecasts into broker sizing until OOS gates pass "
                        "persistently."
                    ),
                )
            )
    metrics = _asset_metrics_table(stocks)
    if "max_drawdown" in metrics.columns:
        drawdown_breaches = metrics[pd.to_numeric(metrics["max_drawdown"], errors="coerce") < -0.35]
        if not drawdown_breaches.empty:
            findings.append(
                _finding(
                    "MEDIUM",
                    "Several assets breach drawdown limits",
                    _join(drawdown_breaches["asset_id"].astype(str).tolist()),
                    "Broker workflow needs portfolio-level drawdown constraints and risk budgets.",
                )
            )
    findings.extend(
        [
            _finding(
                "MEDIUM",
                "Daily OHLCV cannot replicate intraday microstructure studies",
                "Execution cost and liquidity blocks are ADV/range proxies only.",
                "Add intraday bars, bid-ask spreads, order book depth and broker fill logs.",
            ),
            _finding(
                "MEDIUM",
                "Factor alpha is still benchmark-proxy based",
                "The report controls benchmark beta but not official Fama-French-Carhart factors.",
                (
                    "Ingest factor returns and point-in-time fundamentals before claiming "
                    "factor-adjusted alpha."
                ),
            ),
            _finding(
                "MEDIUM",
                "Portfolio optimizer is in-sample and grid based",
                "Mean/covariance estimates are noisy and the frontier is approximate.",
                (
                    "Add covariance shrinkage, walk-forward optimization, CVaR/ES constraints "
                    "and turnover costs."
                ),
            ),
        ]
    )
    return findings


def _finding(severity: str, finding: str, evidence: str, required_change: str) -> dict[str, str]:
    return {
        "severity": severity,
        "finding": finding,
        "evidence": evidence,
        "required_project_change": required_change,
    }


def _project_change_recommendations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "priority": item.get("severity"),
            "change": item.get("required_project_change"),
            "why": item.get("finding"),
        }
        for item in findings
    ]


def _method_traceability() -> list[dict[str, Any]]:
    return [
        {
            "study_family": "ML asset pricing",
            "implemented_here": "Walk-forward daily-return ML diagnostics with baselines.",
            "state_of_art_reference": "GuKellyXiu2020; ChenPelgerZhu2024",
            "current_limitation": "No large CRSP/Compustat-style cross-section or SDF replication.",
        },
        {
            "study_family": "Overfitting control",
            "implemented_here": "OOS gates, PSR/DSR proxy in stock reports, no random splits.",
            "state_of_art_reference": "BaileyLopezDePrado2014; Hansen2005; White2000",
            "current_limitation": "SPA/Reality Check is not yet integrated into model selection.",
        },
        {
            "study_family": "Tail risk",
            "implemented_here": "Historical/parametric/MC VaR and ES with rolling VaR exceptions.",
            "state_of_art_reference": (
                "AcerbiTasche2002; RockafellarUryasev2000; Kupiec1995; Christoffersen1998"
            ),
            "current_limitation": "EVT/POT and formal ES backtesting are future work.",
        },
        {
            "study_family": "Performance inference",
            "implemented_here": "Lo-style Sharpe adjustment and HAC/Newey-West mean t-statistics.",
            "state_of_art_reference": "Lo2002; NeweyWest1987",
            "current_limitation": "No multiple-testing correction or factor-residual Sharpe yet.",
        },
        {
            "study_family": "Portfolio construction",
            "implemented_here": (
                "Correlation, covariance, equal-weight, inverse-vol, min-var, max-Sharpe, "
                "empirical CVaR grid optimization and annual regime stability."
            ),
            "state_of_art_reference": (
                "Markowitz1952; LedoitWolf2004; DeMiguel2009; AngTimmermann2012"
            ),
            "current_limitation": (
                "Convex CVaR, optimal shrinkage, HRP, Black-Litterman and turnover-aware "
                "walk-forward optimization are not final."
            ),
        },
        {
            "study_family": "Execution and costs",
            "implemented_here": "ADV-based spread/impact sensitivity scenarios.",
            "state_of_art_reference": "AlmgrenChriss2001; Gatheral2010",
            "current_limitation": (
                "No broker fills, intraday volume curves or order-book calibration."
            ),
        },
    ]


def _glossary_table() -> list[dict[str, str]]:
    return [
        {
            "term": "OHLCV",
            "meaning": "Open, High, Low, Close, Volume.",
            "why_it_matters": "Es la unidad minima diaria usada para precios, retornos y volumen.",
        },
        {
            "term": "Return / Retorno",
            "meaning": "Cambio porcentual de precio entre dos fechas.",
            "why_it_matters": "Permite comparar activos con precios nominales distintos.",
        },
        {
            "term": "CAGR",
            "meaning": "Compound Annual Growth Rate; crecimiento anual compuesto.",
            "why_it_matters": "Resume el crecimiento historico anualizado de una inversion.",
        },
        {
            "term": "Volatility / Volatilidad",
            "meaning": "Desviacion tipica anualizada de retornos.",
            "why_it_matters": "Mide dispersion historica, no perdida maxima garantizada.",
        },
        {
            "term": "Sharpe Ratio",
            "meaning": "Retorno excedente por unidad de volatilidad.",
            "why_it_matters": "Compara rentabilidad ajustada por riesgo bajo supuestos fuertes.",
        },
        {
            "term": "HAC / Newey-West",
            "meaning": "Error estandar robusto a autocorrelacion y heterocedasticidad.",
            "why_it_matters": "Evita exagerar t-statistics cuando los retornos no son iid.",
        },
        {
            "term": "Sortino Ratio",
            "meaning": "Retorno excedente por unidad de volatilidad negativa.",
            "why_it_matters": "Penaliza principalmente caidas, no subidas.",
        },
        {
            "term": "Max Drawdown",
            "meaning": "Peor caida desde maximo acumulado hasta minimo posterior.",
            "why_it_matters": "Aproxima el dolor historico de mantener una posicion.",
        },
        {
            "term": "VaR",
            "meaning": "Value at Risk; umbral de perdida para un nivel de confianza.",
            "why_it_matters": "Indica una frontera de perdida, pero no la severidad mas alla.",
        },
        {
            "term": "ES / CVaR",
            "meaning": "Expected Shortfall / Conditional VaR; perdida media en la cola.",
            "why_it_matters": "Es mas informativo que VaR para eventos extremos.",
        },
        {
            "term": "Regime Stability",
            "meaning": "Estabilidad de retornos, volatilidad y correlaciones entre periodos.",
            "why_it_matters": "Una cartera diversificada en promedio puede concentrarse en crisis.",
        },
        {
            "term": "OOS",
            "meaning": "Out of Sample; datos no usados para ajustar el modelo.",
            "why_it_matters": "Evita confundir ajuste historico con capacidad predictiva.",
        },
        {
            "term": "IC",
            "meaning": "Information Coefficient; correlacion entre prediccion y resultado.",
            "why_it_matters": "Mide si el ranking/senal predictiva tiene direccion correcta.",
        },
        {
            "term": "Shrinkage",
            "meaning": "Regularizacion de una matriz de covarianza hacia un objetivo estable.",
            "why_it_matters": "Reduce pesos extremos causados por ruido de estimacion.",
        },
        {
            "term": "HHI",
            "meaning": "Herfindahl-Hirschman Index; suma de pesos al cuadrado.",
            "why_it_matters": "Mide concentracion efectiva de una cartera.",
        },
    ]


def _formula_catalog_table() -> list[dict[str, str]]:
    return [
        {
            "concept": "Simple return",
            "formula": "R_t = P_t/P_{t-1} - 1",
            "interpretation": "Ganancia o perdida porcentual entre dos cierres consecutivos.",
        },
        {
            "concept": "Portfolio return",
            "formula": "R_{p,t} = sum_i w_i R_{i,t}",
            "interpretation": "Retorno agregado ponderado por pesos de cartera.",
        },
        {
            "concept": "Annualized volatility",
            "formula": "sigma_ann = std(R_t) sqrt(252)",
            "interpretation": "Volatilidad diaria escalada a un ano bursatil aproximado.",
        },
        {
            "concept": "Sharpe ratio",
            "formula": "SR = mean(R_t-r_f) / std(R_t-r_f) sqrt(252)",
            "interpretation": "Compensacion historica por unidad de riesgo total.",
        },
        {
            "concept": "Autocorrelation-adjusted Sharpe",
            "formula": "SR_HAC = SR / sqrt(1 + 2 sum_k (1-k/(q+1)) rho_k)",
            "interpretation": "Sharpe penalizado si los retornos tienen dependencia serial.",
        },
        {
            "concept": "Maximum drawdown",
            "formula": "DD_t = V_t / max_{s<=t}(V_s) - 1",
            "interpretation": "Caida relativa desde el maximo acumulado.",
        },
        {
            "concept": "CAPM beta",
            "formula": "beta_i = Cov(R_i,R_m) / Var(R_m)",
            "interpretation": "Sensibilidad historica al benchmark de mercado.",
        },
        {
            "concept": "Historical VaR",
            "formula": "VaR_alpha(L) = quantile_alpha(L), L_t=-R_t",
            "interpretation": "Perdida historica que solo se supera en la cola 1-alpha.",
        },
        {
            "concept": "Expected Shortfall",
            "formula": "ES_alpha = E[L | L >= VaR_alpha]",
            "interpretation": "Perdida media condicionada a estar en la cola.",
        },
        {
            "concept": "Empirical CVaR objective",
            "formula": "min_w ES_alpha(-sum_i w_i R_{i,t}), sum_i w_i=1, w_i>=0",
            "interpretation": "Busca pesos long-only con menor perdida media de cola observada.",
        },
        {
            "concept": "Mean-variance risk",
            "formula": "sigma_p = sqrt(w' Sigma w)",
            "interpretation": "Riesgo de cartera que combina volatilidad y correlaciones.",
        },
        {
            "concept": "Diagonal covariance shrinkage",
            "formula": "Sigma_delta = (1-delta)Sigma + delta diag(Sigma)",
            "interpretation": "Analisis de sensibilidad que reduce correlaciones estimadas.",
        },
        {
            "concept": "HHI concentration",
            "formula": "HHI = sum_i w_i^2; N_eff = 1/HHI",
            "interpretation": "Numero efectivo de posiciones tras concentracion de pesos.",
        },
        {
            "concept": "Average pairwise correlation",
            "formula": "rho_bar = 2/(N(N-1)) sum_{i<j} rho_ij",
            "interpretation": "Resumen anual de common movement y fragilidad de diversificacion.",
        },
        {
            "concept": "OOS R-squared",
            "formula": "R2_OOS = 1 - SSE_model/SSE_baseline",
            "interpretation": "Mejora predictiva fuera de muestra contra referencia ingenua.",
        },
    ]


def _study_limitations(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "limitation": "Historical results are not forecasts.",
            "impact": "A strong past Sharpe can disappear under regime change.",
        },
        {
            "limitation": "Daily OHLCV is not institutional execution data.",
            "impact": "Slippage and capacity are approximate until broker/intraday data are added.",
        },
        {
            "limitation": "Universe can have survivorship bias.",
            "impact": "Current tickers may overstate historical investability.",
        },
        {
            "limitation": f"Source data mode: {_data_mode(report)}.",
            "impact": (
                "Strict final studies should use provider data and fail on synthetic fallback."
            ),
        },
    ]


def _write_tables(tables: dict[str, Any], tables_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for index, (name, rows) in enumerate(tables.items(), start=1):
        frame = pd.DataFrame(rows if isinstance(rows, list) else [])
        path = tables_dir / f"{index:02d}_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def _clear_generated_files(directory: Path) -> None:
    """Remove stale generated files from a known output subdirectory."""

    for child in directory.iterdir():
        if child.is_file():
            child.unlink()


def _write_figures(
    report: dict[str, Any],
    study: dict[str, Any],
    figures_dir: Path,
) -> dict[str, str]:
    stocks = _mapping(report.get("stocks"))
    prices = _wide_stock_series(stocks, "price_series", "price")
    cumulative = _wide_stock_series(stocks, "cumulative_returns", "cumulative_return")
    drawdowns = _wide_stock_series(stocks, "drawdown_series", "drawdown")
    returns = _wide_stock_series(stocks, "simple_returns", "return")
    portfolio = _mapping(report.get("portfolio"))
    optimization = _mapping(report.get("optimization"))
    correlation = _matrix_frame(portfolio.get("correlation_matrix"))
    metrics = _asset_metrics_table(stocks)
    decisions = _decision_table(stocks)
    tables = _mapping(study.get("tables"))
    paths = {
        "normalized_prices": figures_dir / "01_normalized_prices.html",
        "cumulative_returns": figures_dir / "02_cumulative_returns.html",
        "correlation_heatmap": figures_dir / "03_correlation_heatmap.html",
        "risk_return_scatter": figures_dir / "04_risk_return_scatter.html",
        "drawdowns": figures_dir / "05_drawdowns.html",
        "efficient_frontier": figures_dir / "06_efficient_frontier.html",
        "decision_scores": figures_dir / "07_decision_scores.html",
        "shrinkage_weight_comparison": figures_dir / "08_shrinkage_weight_comparison.html",
        "rolling_correlation_stability": figures_dir / "09_rolling_correlation_stability.html",
        "var_exceptions_through_time": figures_dir / "10_var_exceptions_through_time.html",
        "exception_traffic_light": figures_dir / "11_exception_traffic_light.html",
        "sharpe_vs_deflated_sharpe": figures_dir / "12_sharpe_vs_deflated_sharpe.html",
        "ml_confidence_comparison": figures_dir / "13_ml_confidence_comparison.html",
        "portfolio_robustness_heatmap": figures_dir / "14_portfolio_robustness_heatmap.html",
        "execution_cost_sensitivity": figures_dir / "15_execution_cost_sensitivity.html",
        "final_decision_waterfall": figures_dir / "16_final_decision_waterfall.html",
    }
    _write_html(
        paths["normalized_prices"],
        _line_chart_html(prices, "Normalized close prices", normalize=True),
    )
    _write_html(paths["cumulative_returns"], _line_chart_html(cumulative, "Cumulative returns"))
    _write_html(
        paths["correlation_heatmap"],
        _heatmap_html(correlation, "Return correlation matrix"),
    )
    _write_html(paths["risk_return_scatter"], _scatter_html(metrics, "Risk-return map"))
    _write_html(paths["drawdowns"], _line_chart_html(drawdowns, "Drawdowns"))
    _write_html(
        paths["efficient_frontier"],
        _frontier_html(optimization, "Efficient frontier grid"),
    )
    _write_html(paths["decision_scores"], _bar_html(decisions, "Research signal scores"))
    _write_html(
        paths["shrinkage_weight_comparison"],
        _shrinkage_weight_comparison_html(
            pd.DataFrame(tables.get("covariance_shrinkage_comparison", []))
        ),
    )
    _write_html(
        paths["rolling_correlation_stability"],
        _rolling_correlation_html(returns, pd.DataFrame(tables.get("annual_regime_stability", []))),
    )
    _write_html(
        paths["var_exceptions_through_time"],
        _var_exceptions_html(pd.DataFrame(tables.get("var_exception_table", []))),
    )
    _write_html(
        paths["exception_traffic_light"],
        _exception_traffic_light_html(pd.DataFrame(tables.get("tail_risk_backtesting", []))),
    )
    _write_html(
        paths["sharpe_vs_deflated_sharpe"],
        _sharpe_deflated_html(pd.DataFrame(tables.get("multiple_testing_adjustments", []))),
    )
    _write_html(
        paths["ml_confidence_comparison"],
        _ml_confidence_html(pd.DataFrame(tables.get("model_confidence", []))),
    )
    _write_html(
        paths["portfolio_robustness_heatmap"],
        _portfolio_robustness_html(pd.DataFrame(tables.get("portfolio_robustness", []))),
    )
    _write_html(
        paths["execution_cost_sensitivity"],
        _execution_cost_html(pd.DataFrame(tables.get("execution_cost_sensitivity", []))),
    )
    _write_html(
        paths["final_decision_waterfall"],
        _final_decision_waterfall_html(
            pd.DataFrame(tables.get("final_research_decision_table", []))
        ),
    )
    return {name: str(path) for name, path in paths.items()}


def _shrinkage_weight_comparison_html(frame: pd.DataFrame) -> str:
    if frame.empty or "weights_json" not in frame.columns:
        return "<h1>Shrinkage weight comparison</h1><p>No shrinkage weights available.</p>"
    rows = frame[frame["optimizer_objective"] == "min_variance"]
    if rows.empty:
        rows = frame.head(2)
    bars = []
    for _, row in rows.iterrows():
        weights = _json_loads_mapping(row.get("weights_json"))
        label = str(row.get("covariance_estimator", "estimator"))
        for asset, weight in weights.items():
            bars.append({"label": f"{label}:{asset}", "value": _safe_float(weight) or 0.0})
    return _simple_bar_chart_html(
        pd.DataFrame(bars),
        "Shrinkage weight comparison",
        label_col="label",
        value_col="value",
        y_label="Portfolio weight",
    )


def _rolling_correlation_html(returns: pd.DataFrame, annual_regime: pd.DataFrame) -> str:
    if not annual_regime.empty and "average_pairwise_correlation" in annual_regime.columns:
        frame = annual_regime.loc[:, ["year", "average_pairwise_correlation"]].rename(
            columns={"year": "label", "average_pairwise_correlation": "value"}
        )
        return _simple_bar_chart_html(
            frame,
            "Annual pairwise-correlation stability",
            label_col="label",
            value_col="value",
            y_label="Average pairwise correlation",
        )
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return "<h1>Rolling correlation stability</h1><p>No return data available.</p>"
    values = []
    for timestamp in clean.index[126::21]:
        window = clean.loc[:timestamp].tail(126)
        pair_values = _upper_triangle_values(window.corr())
        values.append(
            {"label": pd.Timestamp(timestamp).date().isoformat(), "value": np.mean(pair_values)}
        )
    return _simple_bar_chart_html(
        pd.DataFrame(values),
        "Rolling 126-day average correlation",
        label_col="label",
        value_col="value",
        y_label="Average pairwise correlation",
    )


def _var_exceptions_html(frame: pd.DataFrame) -> str:
    if frame.empty or "timestamp" not in frame.columns:
        return "<h1>VaR exceptions through time</h1><p>No exception rows available.</p>"
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    counts = data.dropna(subset=["timestamp"]).groupby(data["timestamp"].dt.year).size()
    plot = pd.DataFrame({"label": counts.index.astype(str), "value": counts.to_numpy(dtype=float)})
    return _simple_bar_chart_html(
        plot,
        "VaR exceptions through time",
        label_col="label",
        value_col="value",
        y_label="Exception count in displayed rows",
    )


def _exception_traffic_light_html(frame: pd.DataFrame) -> str:
    if frame.empty or "exception_rate" not in frame.columns:
        return "<h1>Exception traffic light</h1><p>No VaR backtesting rows available.</p>"
    plot = frame.loc[:, ["asset_id", "exception_rate"]].rename(
        columns={"asset_id": "label", "exception_rate": "value"}
    )
    return _simple_bar_chart_html(
        plot,
        "VaR exception traffic light",
        label_col="label",
        value_col="value",
        y_label="Exception rate",
    )


def _sharpe_deflated_html(frame: pd.DataFrame) -> str:
    if frame.empty or "deflated_sharpe" not in frame.columns:
        return "<h1>Sharpe vs deflated Sharpe</h1><p>No DSR rows available.</p>"
    rows = []
    for _, row in frame.iterrows():
        asset = str(row.get("asset_id", "asset"))
        rows.append({"label": f"{asset}:observed", "value": row.get("observed_sharpe")})
        rows.append({"label": f"{asset}:deflated", "value": row.get("deflated_sharpe")})
    return _simple_bar_chart_html(
        pd.DataFrame(rows),
        "Observed Sharpe vs DSR approximation",
        label_col="label",
        value_col="value",
        y_label="Sharpe units",
    )


def _ml_confidence_html(frame: pd.DataFrame) -> str:
    if frame.empty or "model_confidence_status" not in frame.columns:
        return "<h1>ML confidence comparison</h1><p>No confidence rows available.</p>"
    mapping = {
        "MODEL_CONFIDENCE_ACCEPTED": 3,
        "MODEL_CONFIDENCE_WEAK": 2,
        "MODEL_CONFIDENCE_REQUIRES_MORE_TESTS": 1,
        "MODEL_CONFIDENCE_REJECTED": 0,
    }
    plot = pd.DataFrame(
        {
            "label": frame["asset_id"].astype(str),
            "value": [mapping.get(str(status), 0) for status in frame["model_confidence_status"]],
        }
    )
    return _simple_bar_chart_html(
        plot,
        "ML confidence comparison",
        label_col="label",
        value_col="value",
        y_label="Status score: rejected=0, accepted=3",
    )


def _portfolio_robustness_html(frame: pd.DataFrame) -> str:
    if frame.empty or "weight_l1_shift" not in frame.columns:
        return "<h1>Portfolio robustness heatmap</h1><p>No robustness rows available.</p>"
    rows = []
    for _, row in frame.iterrows():
        rows.append({"label": f"{row.get('objective')}:L1", "value": row.get("weight_l1_shift")})
        rows.append(
            {"label": f"{row.get('objective')}:max", "value": row.get("max_abs_weight_shift")}
        )
    return _simple_bar_chart_html(
        pd.DataFrame(rows),
        "Portfolio robustness weight shifts",
        label_col="label",
        value_col="value",
        y_label="Absolute weight shift",
    )


def _execution_cost_html(frame: pd.DataFrame) -> str:
    if frame.empty or "total_estimated_cost" not in frame.columns:
        return "<h1>Execution cost sensitivity</h1><p>No cost sensitivity rows available.</p>"
    data = frame[frame["asset_id"].astype(str) != "PORTFOLIO_TOTAL"].copy()
    data = data.loc[:, ["rebalance_scenario", "asset_id", "total_estimated_cost"]]
    data["label"] = data["rebalance_scenario"].astype(str) + ":" + data["asset_id"].astype(str)
    data = data.rename(columns={"total_estimated_cost": "value"})
    return _simple_bar_chart_html(
        data,
        "Execution cost sensitivity",
        label_col="label",
        value_col="value",
        y_label="Estimated cost in dollars",
    )


def _final_decision_waterfall_html(frame: pd.DataFrame) -> str:
    if frame.empty or "research_only_signal" not in frame.columns:
        return "<h1>Final decision waterfall</h1><p>No final decision rows available.</p>"
    counts = frame["research_only_signal"].astype(str).value_counts().sort_index()
    plot = pd.DataFrame({"label": counts.index, "value": counts.to_numpy(dtype=float)})
    return _simple_bar_chart_html(
        plot,
        "Final research-only decision waterfall",
        label_col="label",
        value_col="value",
        y_label="Asset count",
    )


def _simple_bar_chart_html(
    frame: pd.DataFrame,
    title: str,
    *,
    label_col: str,
    value_col: str,
    y_label: str,
) -> str:
    if frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return f"<h1>{_escape(title)}</h1><p>No data available.</p>"
    clean = frame.copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce").fillna(0.0)
    clean = clean.head(40)
    width, height = 980, 500
    left, top, right, bottom = 86, 42, 24, 112
    plot_w = width - left - right
    plot_h = height - top - bottom
    minimum = min(float(clean[value_col].min()), 0.0)
    maximum = max(float(clean[value_col].max()), 0.0)
    if math.isclose(minimum, maximum):
        maximum = minimum + 1.0
    bar_w = plot_w / max(len(clean), 1)
    zero_y = top + (maximum / (maximum - minimum)) * plot_h
    bars = []
    for position, (_, row) in enumerate(clean.iterrows()):
        value = float(row[value_col])
        x = left + position * bar_w + 4
        y_value = top + (1.0 - ((value - minimum) / (maximum - minimum))) * plot_h
        y = min(y_value, zero_y)
        height_value = abs(zero_y - y_value)
        color = "#0f766e" if value >= 0 else "#b91c1c"
        label = _escape(str(row[label_col]))
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 8, 3):.2f}" '
            f'height="{height_value:.2f}" fill="{color}" />'
            f'<text x="{x + bar_w / 2 - 4:.2f}" y="{height - 36}" text-anchor="middle" '
            f'font-size="10" transform="rotate(-42 {x + bar_w / 2 - 4:.2f},{height - 36})">'
            f"{label}</text>"
        )
    axis = _svg_axes(width, height, left, top, plot_w, plot_h, minimum, maximum)
    return f"""<h1>{_escape(title)}</h1>
<p style="max-width:900px;">{_escape(y_label)}</p>
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
{axis}
<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#111820" />
{''.join(bars)}
</svg>"""


def _json_loads_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return _mapping(result)


def _write_html(path: Path, body: str) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Quant Study Figure</title></head>
<body style="font-family:Arial,sans-serif;background:#f7f2e8;color:#111820;">
{body}
<p style="font-size:12px;color:#6b7280;">Research-only figure. No investment advice.</p>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def _line_chart_html(frame: pd.DataFrame, title: str, *, normalize: bool = False) -> str:
    clean = frame.copy()
    if clean.empty:
        return f"<h1>{_escape(title)}</h1><p>No data available.</p>"
    if normalize:
        clean = clean.apply(_normalize_series)
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    clean = _downsample(clean, 420)
    if clean.empty:
        return f"<h1>{_escape(title)}</h1><p>No finite data available.</p>"
    values = clean.to_numpy(dtype=float)
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return f"<h1>{_escape(title)}</h1><p>No finite data available.</p>"
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    width, height = 980, 460
    left, top, right, bottom = 72, 34, 24, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = _palette()
    polylines = []
    labels = []
    for idx, column in enumerate(clean.columns[:12]):
        series = clean[column].astype(float)
        points = []
        finite_positions = np.where(np.isfinite(series.to_numpy(dtype=float)))[0]
        for pos in finite_positions:
            x = left + (pos / max(len(clean) - 1, 1)) * plot_w
            y = top + (1.0 - ((float(series.iloc[pos]) - y_min) / (y_max - y_min))) * plot_h
            points.append(f"{x:.2f},{y:.2f}")
        if points:
            color = colors[idx % len(colors)]
            polylines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
                'stroke-width="1.8" />'
            )
            labels.append(
                f'<span style="color:{color};margin-right:14px;">&#9632; '
                f"{_escape(str(column))}</span>"
            )
    axis = _svg_axes(width, height, left, top, plot_w, plot_h, y_min, y_max)
    svg = f"""<h1>{_escape(title)}</h1>
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
{axis}
{''.join(polylines)}
</svg>
<div>{''.join(labels)}</div>"""
    return svg


def _heatmap_html(matrix: pd.DataFrame, title: str) -> str:
    if matrix.empty:
        return f"<h1>{_escape(title)}</h1><p>No correlation data available.</p>"
    assets = [str(item) for item in matrix.index]
    n = len(assets)
    cell = 46
    left = 120
    top = 90
    width = left + n * cell + 40
    height = top + n * cell + 40
    rects = []
    for i, row_asset in enumerate(assets):
        for j, col_asset in enumerate(assets):
            value = _safe_float(matrix.loc[row_asset, col_asset])
            color = _corr_color(value)
            x = left + j * cell
            y = top + i * cell
            label = "" if value is None else f"{value:.2f}"
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" />'
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                f'font-size="11">{label}</text>'
            )
    labels = []
    for idx, asset in enumerate(assets):
        labels.append(
            f'<text x="{left - 8}" y="{top + idx * cell + cell / 2 + 4}" '
            f'text-anchor="end" font-size="12">{_escape(asset)}</text>'
        )
        labels.append(
            f'<text x="{left + idx * cell + cell / 2}" y="{top - 12}" '
            f'text-anchor="middle" font-size="12" transform="rotate(-45 '
            f'{left + idx * cell + cell / 2},{top - 12})">{_escape(asset)}</text>'
        )
    return f"""<h1>{_escape(title)}</h1>
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
{''.join(labels)}{''.join(rects)}
</svg>"""


def _scatter_html(metrics: pd.DataFrame, title: str) -> str:
    if metrics.empty:
        return f"<h1>{_escape(title)}</h1><p>No metrics available.</p>"
    frame = metrics.copy()
    frame["annualized_volatility"] = pd.to_numeric(frame["annualized_volatility"], errors="coerce")
    frame["cagr"] = pd.to_numeric(frame["cagr"], errors="coerce")
    frame = frame.dropna(subset=["annualized_volatility", "cagr"])
    if frame.empty:
        return f"<h1>{_escape(title)}</h1><p>No finite risk-return points available.</p>"
    return _point_chart_html(
        frame,
        title,
        x_col="annualized_volatility",
        y_col="cagr",
        label_col="asset_id",
        x_label="Annualized volatility",
        y_label="CAGR",
    )


def _frontier_html(optimization: dict[str, Any], title: str) -> str:
    frontier = optimization.get("frontier", [])
    if not isinstance(frontier, list) or not frontier:
        return f"<h1>{_escape(title)}</h1><p>No frontier data available.</p>"
    frame = pd.DataFrame(frontier)
    frame = frame.rename(columns={"annualized_volatility": "risk", "annualized_return": "return"})
    frame["portfolio"] = [f"p{i}" for i in range(len(frame))]
    frame = _downsample(frame.loc[:, ["risk", "return", "portfolio"]], 600)
    return _point_chart_html(
        frame,
        title,
        x_col="risk",
        y_col="return",
        label_col="portfolio",
        x_label="Annualized volatility",
        y_label="Annualized return",
        label_points=False,
    )


def _bar_html(decisions: pd.DataFrame, title: str) -> str:
    if decisions.empty or "score" not in decisions.columns:
        return f"<h1>{_escape(title)}</h1><p>No decision scores available.</p>"
    frame = decisions.copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    width, height = 980, 460
    left, top, right, bottom = 80, 36, 24, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_abs = max(float(frame["score"].abs().max()), 1.0)
    bar_w = plot_w / max(len(frame), 1)
    zero_y = top + plot_h / 2
    bars = []
    for idx, row in frame.iterrows():
        score = float(row["score"])
        x = left + idx * bar_w + 5
        h = abs(score) / max_abs * (plot_h / 2 - 10)
        y = zero_y - h if score >= 0 else zero_y
        color = "#0f766e" if score >= 0 else "#b91c1c"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 10, 4):.2f}" '
            f'height="{h:.2f}" fill="{color}" />'
            f'<text x="{x + bar_w / 2 - 5:.2f}" y="{height - 28}" text-anchor="middle" '
            f'font-size="11" transform="rotate(-35 {x + bar_w / 2 - 5:.2f},{height - 28})">'
            f'{_escape(str(row.get("asset_id", "")))}</text>'
        )
    return f"""<h1>{_escape(title)}</h1>
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
<line x1="{left}" y1="{zero_y}" x2="{width-right}" y2="{zero_y}" stroke="#111820" />
{''.join(bars)}
</svg>"""


def _point_chart_html(
    frame: pd.DataFrame,
    title: str,
    *,
    x_col: str,
    y_col: str,
    label_col: str,
    x_label: str,
    y_label: str,
    label_points: bool = True,
) -> str:
    clean = frame.copy()
    clean[x_col] = pd.to_numeric(clean[x_col], errors="coerce")
    clean[y_col] = pd.to_numeric(clean[y_col], errors="coerce")
    clean = clean.dropna(subset=[x_col, y_col])
    if clean.empty:
        return f"<h1>{_escape(title)}</h1><p>No finite points available.</p>"
    width, height = 980, 460
    left, top, right, bottom = 76, 36, 32, 66
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = _range_with_padding(clean[x_col])
    y_min, y_max = _range_with_padding(clean[y_col])
    points = []
    for _, row in clean.iterrows():
        x = left + ((float(row[x_col]) - x_min) / (x_max - x_min)) * plot_w
        y = top + (1.0 - ((float(row[y_col]) - y_min) / (y_max - y_min))) * plot_h
        label = _escape(str(row.get(label_col, "")))
        text = (
            f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-size="11">{label}</text>'
            if label_points
            else ""
        )
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#0f766e" />{text}')
    axis = _svg_axes(width, height, left, top, plot_w, plot_h, y_min, y_max)
    return f"""<h1>{_escape(title)}</h1>
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
{axis}
<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="12">
{_escape(x_label)}</text>
<text x="18" y="{top + plot_h / 2}" text-anchor="middle" font-size="12"
 transform="rotate(-90 18,{top + plot_h / 2})">{_escape(y_label)}</text>
{''.join(points)}
</svg>"""


def _svg_axes(
    width: int,
    height: int,
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    y_min: float,
    y_max: float,
) -> str:
    return (
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="#fffaf0" stroke="#c9b894" />'
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11">{y_max:.2f}</text>'
        f'<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" '
        f'font-size="11">{y_min:.2f}</text>'
        f'<text x="{left}" y="{height - 22}" text-anchor="start" font-size="11">start</text>'
        f'<text x="{width - 24}" y="{height - 22}" text-anchor="end" font-size="11">end</text>'
    )


def _normalize_series(series: pd.Series) -> pd.Series:
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan)
    first = clean.dropna().iloc[0] if not clean.dropna().empty else np.nan
    if not np.isfinite(first) or first == 0:
        return clean * np.nan
    return clean / first


def _downsample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    positions = np.linspace(0, len(frame) - 1, max_rows).round().astype(int)
    return frame.iloc[sorted(set(int(pos) for pos in positions))]


def _range_with_padding(series: pd.Series) -> tuple[float, float]:
    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    pad = (maximum - minimum) * 0.05
    return minimum - pad, maximum + pad


def _corr_color(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "#e5e7eb"
    clipped = max(-1.0, min(1.0, float(value)))
    if clipped >= 0:
        intensity = int(245 - 105 * clipped)
        return f"rgb({intensity},{245},{230})"
    intensity = int(245 + 10 * clipped)
    return f"rgb({245},{intensity},{intensity})"


def _palette() -> list[str]:
    return [
        "#0f766e",
        "#b45309",
        "#1d4ed8",
        "#be123c",
        "#6d28d9",
        "#047857",
        "#92400e",
        "#0369a1",
        "#a21caf",
        "#4d7c0f",
        "#7f1d1d",
        "#334155",
    ]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _markdown_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "No rows available."
    clean_rows = [_mapping(row) for row in rows]
    columns = list(dict.fromkeys(column for row in clean_rows for column in row))
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in clean_rows:
        body.append("| " + " | ".join(_md_cell(row.get(column)) for column in columns) + " |")
    return "\n".join([header, separator, *body])


def _md_cell(value: object) -> str:
    text = str(_json_safe(value))
    return text.replace("|", "\\|").replace("\n", " ")[:240]


def _paragraph(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value or "")


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _join(value: object, *, limit: int | None = None) -> str:
    if isinstance(value, dict):
        items = [f"{key}:{val}" for key, val in value.items()]
    elif isinstance(value, (list, tuple, set, pd.Series)):
        items = [str(item) for item in list(value)]
    elif value is None:
        items = []
    else:
        items = [str(value)]
    if limit is not None:
        items = items[:limit]
    return "; ".join(items)


def _safe_float(value: object) -> float | None:
    try:
        clean = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return clean if np.isfinite(clean) else None


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
