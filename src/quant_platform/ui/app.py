"""Streamlit application for local read-only research inspection."""

from __future__ import annotations

import importlib
import json
from typing import Any

from quant_platform.config.settings import load_settings_from_env
from quant_platform.ui.actions import action_catalog, build_ui_status, safe_command_catalog
from quant_platform.ui.explainers import all_explainers
from quant_platform.ui.formulas import all_formulas
from quant_platform.ui.report_loader import load_dataset_quality_report, read_json_report
from quant_platform.ui.view_models import (
    backtest_metric_rows,
    build_platform_snapshot,
    dataset_quality_rows,
    final_package_artifacts,
    institutional_table,
    report_series,
    transaction_cost_components,
)

_charts_mod = importlib.import_module("quant_platform.ui.charts")
backtest_metric_figure = _charts_mod.backtest_metric_figure
data_quality_warnings_figure = _charts_mod.data_quality_warnings_figure
drawdown_curve_figure = _charts_mod.drawdown_curve_figure
equity_curve_figure = _charts_mod.equity_curve_figure
profile_comparison_figure = _charts_mod.profile_comparison_figure
provider_status_figure = _charts_mod.provider_status_figure
quality_coverage_figure = _charts_mod.quality_coverage_figure
returns_histogram_figure = _charts_mod.returns_histogram_figure
terminal_backtest_equity_figure = _charts_mod.terminal_backtest_equity_figure
terminal_correlation_heatmap = _charts_mod.terminal_correlation_heatmap
terminal_exposure_figure = _charts_mod.terminal_exposure_figure
terminal_frontier_figure = _charts_mod.terminal_frontier_figure
terminal_monte_carlo_fan_figure = _charts_mod.terminal_monte_carlo_fan_figure
terminal_options_figure = _charts_mod.terminal_options_figure
terminal_price_figure = _charts_mod.terminal_price_figure
terminal_var_figure = _charts_mod.terminal_var_figure
transaction_cost_breakdown_figure = _charts_mod.transaction_cost_breakdown_figure
universe_mix_figure = _charts_mod.universe_mix_figure
var_es_conceptual_figure = _charts_mod.var_es_conceptual_figure

SECTIONS = (
    "Inicio",
    "Quant Terminal",
    "Academic Reports",
    "Flujo conceptual",
    "Estado del sistema",
    "Providers",
    "Universo",
    "Datasets",
    "Calidad de datos",
    "Backtests",
    "Comparacion de perfiles",
    "Formulas",
    "Acciones permitidas",
    "Riesgos y limites",
)


def main() -> None:
    """Run the Streamlit UI without starting data downloads, backtests, or trading."""

    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="Institutional Quant Research Terminal",
        page_icon="IQ",
        layout="wide",
    )
    _inject_style(st)

    st.markdown(
        """
        <div class="hero">
          <div>
            <p class="eyebrow">Research-only institutional quant analytics</p>
            <h1>Institutional Quant Research Terminal</h1>
            <p class="hero-copy">
              Reproducible state-of-the-art research package: provenance, returns,
              portfolio construction, tail risk, ML confidence, robustness gates and final paper.
            </p>
          </div>
          <div class="hero-badge">No trading<br/>No advice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_safety_strip(st)

    control_cols = st.columns([1.2, 1.2, 1])
    report_dir = control_cols[0].text_input("Report directory", value="reports/generated")
    registry_dir = "data/registry"
    universe_config = "configs/universe_etfs_crypto_daily.yaml"
    risk_profiles = "configs/risk_profiles.yaml"

    settings = load_settings_from_env()
    snapshot = build_platform_snapshot(
        settings=settings,
        registry_dir=registry_dir,
        report_dir=report_dir,
        universe_config_path=universe_config,
        risk_profiles_path=risk_profiles,
    )
    status = build_ui_status(
        settings=settings,
        registry_dir=registry_dir,
        report_dir=report_dir,
        universe_config_path=universe_config,
        risk_profiles_path=risk_profiles,
    )

    if control_cols[2].button("Regenerate command", use_container_width=True):
        st.code(
            "py -3.13 -m quant_platform.cli build-final-institutional-package "
            "--config configs/quant_terminal_10_stocks.yaml --provider yfinance "
            "--output-dir reports/generated/final_package --max-table-rows 20",
            language="powershell",
        )
    if not _render_institutional_terminal(st, pd, snapshot, status):
        _render_stock_research_terminal(st, pd, snapshot, status)


def _render_header(st) -> None:  # noqa: ANN001
    st.markdown(
        """
        <div class="hero">
          <div>
            <p class="eyebrow">Professional Quant Finance Research</p>
            <h1>Quant Platform Terminal</h1>
            <p class="hero-copy">
              Terminal local para analizar tres acciones, portfolio, Monte Carlo, VaR,
              backtesting, opciones, renta fija, tipos y exposicion. No compra ni vende nada.
            </p>
          </div>
          <div class="hero-badge">Research-only<br/>No trading</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_safety_strip(st) -> None:  # noqa: ANN001
    cols = st.columns(3)
    cols[0].success("Research-only / No trading")
    cols[1].info("No se muestran claves")
    cols[2].warning("Backtest no es prediccion asegurada")


def _render_stock_research_terminal(
    st, pd, snapshot: dict[str, Any], status: dict[str, Any]
) -> None:  # noqa: ANN001
    reports = [
        row
        for row in snapshot["reports"]
        if row.get("report_type") == "professional_quant_terminal"
    ]
    if not reports:
        _empty_state(st, "No hay quant terminal report local.", _terminal_commands())
        return

    report_labels = [
        f"{row['name']} | {row.get('summary', {}).get('data_mode')}" for row in reports
    ]
    selected_report = st.selectbox("Select report", report_labels)
    report_row = reports[report_labels.index(selected_report)]
    report = read_json_report(report_row["path"])
    stocks = report.get("stocks", {})
    if not isinstance(stocks, dict) or not stocks:
        st.error("Selected report has no per-stock payload.")
        return

    stock_cols = st.columns([1, 2.4])
    asset_id = stock_cols[0].selectbox("Ticker", sorted(stocks))
    stock = stocks[asset_id]
    stock_cols[1].markdown(
        "<div class='ticker-note'>Single-ticker research dossier: data, risk, ML, "
        "options and downloadable PDF. Research-only, no trading action.</div>",
        unsafe_allow_html=True,
    )

    metrics = _mapping(stock.get("metrics"))
    var_hist = _mapping(_mapping(stock.get("var")).get("historical"))
    ml = _mapping(stock.get("ml_forecasting"))
    predictive_audit = _mapping(stock.get("predictive_reliability_audit"))
    signal = _mapping(stock.get("decision_signal"))
    data_used = _mapping(stock.get("data_used"))
    _kpi_row(
        st,
        [
            ("Last price", _money(data_used.get("last_close"), data_used.get("currency", "USD"))),
            ("Cumulative return", _pct(metrics.get("final_cumulative_return"))),
            ("CAGR", _pct(metrics.get("cagr"))),
            ("Volatility", _pct(metrics.get("annualized_volatility"))),
            ("Sharpe", _num(metrics.get("sharpe_ratio"))),
            ("Sortino", _num(metrics.get("sortino_ratio"))),
            ("Treynor", _num(metrics.get("treynor_ratio"))),
            ("Jensen alpha", _pct(metrics.get("jensen_alpha"))),
            ("Max drawdown", _pct(metrics.get("max_drawdown"))),
            ("VaR 95", _pct(var_hist.get("var"))),
            ("ES 95", _pct(var_hist.get("expected_shortfall"))),
            ("ML dir. accuracy", _pct(ml.get("directional_accuracy"))),
            ("Predictive audit", str(predictive_audit.get("rating", "N/A"))),
            ("Decision signal", str(signal.get("signal", "INSUFFICIENT_DATA"))),
        ],
    )

    st.markdown("### 1. Research Thread")
    st.write(
        "El dossier conecta el hilo completo del estudio: datos OHLCV, transformacion a "
        "retornos, performance, riesgo, CAPM, VaR/ES, Monte Carlo, ML walk-forward, "
        "backtesting educativo y Black-Scholes-Merton. La clasificacion final resume "
        "evidencia historica bajo limites explicitos, no una instruccion operativa."
    )
    action_text = signal.get("suggested_research_action", "Review model assumptions.")
    st.info(f"Research-only action classification: {action_text}")
    st.warning(
        "Predictive model audit: "
        f"{predictive_audit.get('rating', 'N/A')} - "
        f"{predictive_audit.get('interpretation', 'N/A')}"
    )

    chart_cols = st.columns(2)
    chart_cols[0].plotly_chart(
        _stock_line_figure(stock, "price_series", "price", asset_id, "Price history"),
        use_container_width=True,
    )
    chart_cols[1].plotly_chart(
        _stock_line_figure(
            stock, "cumulative_returns", "cumulative_return", asset_id, "Cumulative returns"
        ),
        use_container_width=True,
    )

    st.markdown("### 2. Data Quality & Market Risk")
    data_cols = st.columns([1, 1, 1.4])
    data_cols[0].dataframe(
        _safe_df(pd, [_mapping(stock.get("data_used"))]), use_container_width=True
    )
    data_cols[1].dataframe(
        _safe_df(pd, [_mapping(stock.get("data_quality"))]), use_container_width=True
    )
    data_cols[2].plotly_chart(
        _stock_line_figure(stock, "drawdown_series", "drawdown", asset_id, "Drawdown"),
        use_container_width=True,
    )

    risk_cols = st.columns(2)
    risk_cols[0].plotly_chart(_stock_returns_histogram(stock), use_container_width=True)
    risk_cols[1].plotly_chart(_stock_var_figure(stock), use_container_width=True)

    st.markdown("### 3. ML, Monte Carlo & Backtest")
    ml_cols = st.columns([1, 1])
    ml_cols[0].dataframe(
        _safe_df(pd, [_ml_summary_row(ml)]), use_container_width=True, hide_index=True
    )
    ml_cols[1].plotly_chart(_stock_ml_prediction_figure(stock), use_container_width=True)
    st.dataframe(
        _safe_df(pd, _predictive_audit_rows(predictive_audit)),
        use_container_width=True,
        hide_index=True,
    )
    mc_cols = st.columns(2)
    mc_cols[0].plotly_chart(_stock_mc_fan_figure(stock), use_container_width=True)
    mc_cols[1].plotly_chart(_stock_backtest_figure(stock), use_container_width=True)

    st.markdown("### 4. Black-Scholes-Merton Options Study")
    opt_cols = st.columns([1.4, 1])
    opt_cols[0].plotly_chart(_stock_options_payoff_figure(stock), use_container_width=True)
    opt_cols[1].dataframe(
        _safe_df(pd, [_options_summary_row(_mapping(stock.get("options_theoretical_analytics")))]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 5. Academic PDF")
    _render_stock_report_links(st, snapshot, asset_id)

    with st.expander("Developer diagnostics"):
        st.json({"ui_status": status, "selected_report": report_row, "selected_stock": asset_id})


def _render_institutional_terminal(
    st, pd, snapshot: dict[str, Any], status: dict[str, Any]
) -> bool:  # noqa: ANN001
    artifacts = final_package_artifacts(snapshot["reports"])
    study_row = artifacts.get("study")
    package_row = artifacts.get("package")
    paper_row = artifacts.get("paper")
    if not study_row and not package_row:
        return False

    study_path = _institutional_study_path(package_row, study_row, artifacts.get("study_metadata"))
    if not study_path:
        _empty_state(
            st,
            "Final package metadata exists, but the study JSON path is missing.",
            _terminal_commands(),
        )
        return True
    study = read_json_report(study_path)
    source = _mapping(study.get("source_terminal_report"))
    controls = _mapping(study.get("audit_controls"))
    findings = institutional_table(study, "critical_findings")
    decisions = institutional_table(study, "final_research_decision_table")

    st.markdown("### Institutional Research Package")
    _kpi_row(
        st,
        [
            ("Data mode", str(source.get("data_mode", "unknown"))),
            ("Strict real data", str(controls.get("strict_real_data", "unknown"))),
            ("Synthetic rejected", str(controls.get("synthetic_rejected", "unknown"))),
            ("Warnings", str(len(source.get("warnings", [])))),
            ("Critical findings", str(len(findings))),
            ("Assets", str(len(institutional_table(study, "asset_metrics")))),
        ],
    )
    st.info(
        "Terminal read-only: state-of-the-art research controls, no broker order generation, "
        "no investment advice."
    )

    tabs = st.tabs(
        [
            "Overview",
            "Asset Results",
            "Portfolio",
            "Tail Risk",
            "ML Confidence",
            "Robustness",
            "Decision Gates",
            "Final Paper",
            "Methodology",
        ]
    )
    with tabs[0]:
        st.write(study.get("plain_language_explanation", "No executive summary available."))
        st.dataframe(_safe_df(pd, findings), use_container_width=True, hide_index=True)
    with tabs[1]:
        _render_study_table(st, pd, study, "asset_metrics")
        st.dataframe(_safe_df(pd, decisions), use_container_width=True, hide_index=True)
    with tabs[2]:
        for name in ("portfolio_weights", "portfolio_diagnostics", "frontier_sample"):
            _render_study_table(st, pd, study, name)
    with tabs[3]:
        for name in (
            "tail_risk",
            "portfolio_tail_risk",
            "tail_risk_backtesting",
            "var_exception_table",
        ):
            _render_study_table(st, pd, study, name)
    with tabs[4]:
        for name in ("ml_audit", "model_confidence", "multiple_testing_adjustments"):
            _render_study_table(st, pd, study, name)
    with tabs[5]:
        for name in (
            "covariance_shrinkage_comparison",
            "portfolio_robustness",
            "execution_cost_sensitivity",
            "factor_model_gap_table",
        ):
            _render_study_table(st, pd, study, name)
    with tabs[6]:
        for name in (
            "decision_gate_evidence",
            "final_research_decision_table",
            "implementation_roadmap_table",
        ):
            _render_study_table(st, pd, study, name)
    with tabs[7]:
        _render_final_paper_links(st, package_row, paper_row)
    with tabs[8]:
        for name in (
            "data_provenance",
            "method_traceability",
            "state_of_art_gap_analysis",
            "glossary",
        ):
            _render_study_table(st, pd, study, name)

    with st.expander("Developer diagnostics"):
        st.json(
            {
                "ui_status": status,
                "package": package_row,
                "paper": paper_row,
                "study_path": study_path,
            }
        )
    return True


def _render_study_table(st, pd, study: dict[str, Any], name: str) -> None:  # noqa: ANN001
    st.markdown(f"#### {name}")
    st.dataframe(
        _safe_df(pd, institutional_table(study, name)),
        use_container_width=True,
        hide_index=True,
    )


def _institutional_study_path(
    package_row: dict[str, Any] | None,
    study_row: dict[str, Any] | None,
    study_metadata_row: dict[str, Any] | None,
) -> str | None:
    if study_row:
        return str(study_row.get("path"))
    for row in (package_row, study_metadata_row):
        if not row:
            continue
        metadata = read_json_report(row["path"])
        outputs = _mapping(metadata.get("outputs"))
        institutional_outputs = _mapping(outputs.get("institutional_study"))
        path = institutional_outputs.get("json") or outputs.get("json")
        if path:
            return str(path)
    return None


def _render_final_paper_links(
    st, package_row: dict[str, Any] | None, paper_row: dict[str, Any] | None
) -> None:  # noqa: ANN001
    from pathlib import Path

    metadata = None
    if paper_row:
        metadata = read_json_report(paper_row["path"])
    elif package_row:
        package = read_json_report(package_row["path"])
        outputs = _mapping(package.get("outputs"))
        paper_outputs = _mapping(outputs.get("final_paper"))
        metadata = {"outputs": paper_outputs, "pdf_export": {}}
    if not metadata:
        st.info("No final academic paper metadata found yet.")
        return
    outputs = _mapping(metadata.get("outputs"))
    st.markdown(
        "<div class='paper-card'>Final institutional paper artifacts</div>",
        unsafe_allow_html=True,
    )
    st.write(f"Markdown: `{outputs.get('markdown', 'not generated')}`")
    st.write(f"HTML: `{outputs.get('html', 'not generated')}`")
    pdf_path = outputs.get("pdf")
    if not pdf_path:
        st.warning(_mapping(metadata.get("pdf_export")).get("status", "PDF not generated"))
        st.code("py -3.13 -m pip install '.[pdf]'", language="powershell")
        st.code("py -3.13 -m playwright install chromium", language="powershell")
        return
    pdf_file = Path(str(pdf_path))
    if not pdf_file.exists():
        st.warning(f"PDF metadata exists but file is missing: {pdf_file}")
        return
    st.download_button(
        "Download final institutional PDF",
        data=pdf_file.read_bytes(),
        file_name=pdf_file.name,
        mime="application/pdf",
        use_container_width=True,
    )


def _stock_line_figure(
    stock: dict[str, Any], series_key: str, value_key: str, asset_id: str, title: str
):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    rows = [row for row in stock.get(series_key, []) if isinstance(row, dict)]
    figure = go.Figure()
    if rows:
        figure.add_trace(
            go.Scatter(
                x=[row.get("timestamp") for row in rows],
                y=[row.get(value_key) for row in rows],
                mode="lines",
                name=asset_id,
            )
        )
    figure.update_layout(
        template="plotly_dark", title=title, xaxis_title="Date", yaxis_title=value_key
    )
    return figure


def _stock_returns_histogram(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    rows = [row for row in stock.get("simple_returns", []) if isinstance(row, dict)]
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=[row.get("return") for row in rows], nbinsx=60, name="Returns"))
    figure.update_layout(template="plotly_dark", title="Returns distribution", xaxis_title="Return")
    return figure


def _stock_var_figure(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    var_payload = _mapping(stock.get("var"))
    rows = []
    for model in ("historical", "parametric_normal", "monte_carlo"):
        payload = _mapping(var_payload.get(model))
        rows.append({"model": model, "metric": "VaR", "value": payload.get("var")})
        rows.append({"model": model, "metric": "ES", "value": payload.get("expected_shortfall")})
    figure = go.Figure()
    for metric in ("VaR", "ES"):
        selected = [row for row in rows if row["metric"] == metric]
        figure.add_trace(
            go.Bar(
                name=metric,
                x=[row["model"] for row in selected],
                y=[row["value"] for row in selected],
            )
        )
    figure.update_layout(template="plotly_dark", title="VaR / ES comparison", barmode="group")
    return figure


def _stock_mc_paths_figure(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    normal_mc = _mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal"))
    rows = _rows(normal_mc.get("paths_sample"))
    figure = go.Figure()
    for path_id in sorted({row.get("path_id") for row in rows})[:15]:
        path_rows = [row for row in rows if row.get("path_id") == path_id]
        figure.add_trace(
            go.Scatter(
                x=[row.get("step") for row in path_rows],
                y=[row.get("value") for row in path_rows],
                mode="lines",
                name=f"path {path_id}",
                opacity=0.35,
            )
        )
    figure.update_layout(template="plotly_dark", title="Monte Carlo paths", xaxis_title="Step")
    return figure


def _stock_mc_fan_figure(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    normal_mc = _mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal"))
    rows = _rows(normal_mc.get("fan_chart"))
    figure = go.Figure()
    for percentile in ("p5", "p25", "p50", "p75", "p95"):
        figure.add_trace(
            go.Scatter(
                x=[row.get("step") for row in rows],
                y=[row.get(percentile) for row in rows],
                mode="lines",
                name=percentile.upper(),
            )
        )
    figure.update_layout(template="plotly_dark", title="Monte Carlo percentile fan")
    return figure


def _stock_ml_prediction_figure(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    rows = _rows(_mapping(stock.get("ml_forecasting")).get("prediction_rows"))
    figure = go.Figure()
    for key, name in (("actual_return", "Actual"), ("model_prediction", "Predicted")):
        figure.add_trace(
            go.Scatter(
                x=[row.get("timestamp") for row in rows],
                y=[row.get(key) for row in rows],
                mode="lines",
                name=name,
            )
        )
    figure.update_layout(template="plotly_dark", title="ML predicted vs actual returns")
    return figure


def _stock_backtest_figure(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    backtests = _mapping(stock.get("backtesting_results"))
    figure = go.Figure()
    for name, payload in backtests.items():
        rows = _rows(_mapping(payload).get("equity_curve"))
        figure.add_trace(
            go.Scatter(
                x=[row.get("timestamp") for row in rows],
                y=[row.get("equity") for row in rows],
                mode="lines",
                name=str(name),
            )
        )
    figure.update_layout(template="plotly_dark", title="Backtest equity curves")
    return figure


def _stock_options_payoff_figure(stock: dict[str, Any]):  # noqa: ANN201
    go = _charts_mod._plotly_go()  # noqa: SLF001
    options = _mapping(stock.get("options_theoretical_analytics"))
    rows = _rows(options.get("payoff_profile"))
    protective = _rows(options.get("protective_put_payoff"))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row.get("underlying_price") for row in rows],
            y=[row.get("payoff") for row in rows],
            mode="lines",
            name="Call payoff",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row.get("underlying_price") for row in protective],
            y=[row.get("net_payoff") for row in protective],
            mode="lines",
            name="Protective put net payoff",
        )
    )
    figure.update_layout(template="plotly_dark", title="Options payoff")
    return figure


def _render_stock_report_links(st, snapshot: dict[str, Any], asset_id: str) -> None:  # noqa: ANN001
    import base64
    from pathlib import Path

    reports = [
        row
        for row in snapshot["reports"]
        if row.get("report_type") == "academic_stock_report"
        and row.get("summary", {}).get("asset_id") == asset_id
    ]
    if not reports:
        st.info("No academic report metadata found for this stock yet.")
        st.code(
            "py -3 -m quant_platform.cli generate-stock-academic-report "
            f"--asset {asset_id} --terminal-report "
            "reports/generated/quant_terminal/10stocks_10y_report.json "
            "--format md --format html --format pdf --include-figures --overwrite",
            language="powershell",
        )
        return
    metadata = read_json_report(reports[-1]["path"])
    outputs = _mapping(metadata.get("outputs"))
    st.markdown("<div class='paper-card'>Academic report artifacts</div>", unsafe_allow_html=True)
    st.write(f"Markdown: `{outputs.get('md', 'not generated')}`")
    st.write(f"HTML: `{outputs.get('html', 'not generated')}`")
    pdf_path = outputs.get("pdf")
    if not pdf_path:
        st.warning(metadata.get("pdf_export", {}).get("status", "PDF not generated"))
        return
    pdf_file = Path(str(pdf_path))
    if not pdf_file.exists():
        st.warning(f"PDF metadata exists but file is missing: {pdf_file}")
        return
    pdf_bytes = pdf_file.read_bytes()
    st.download_button(
        "Download academic PDF",
        data=pdf_bytes,
        file_name=pdf_file.name,
        mime="application/pdf",
        use_container_width=True,
    )
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    st.components.v1.html(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="760" '
        'style="border:1px solid rgba(214,179,90,0.35);border-radius:18px;"></iframe>',
        height=790,
    )


def _ml_summary_row(ml: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": ml.get("status"),
        "rmse": ml.get("rmse"),
        "mae": ml.get("mae"),
        "oos_r_squared": ml.get("oos_r_squared"),
        "directional_accuracy": ml.get("directional_accuracy"),
        "baseline_directional_accuracy": ml.get("baseline_directional_accuracy"),
        "directional_accuracy_edge": ml.get("directional_accuracy_edge_vs_naive"),
        "information_coefficient": ml.get("information_coefficient"),
        "strategy_sharpe": ml.get("strategy_sharpe"),
    }


def _predictive_audit_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = audit.get("criteria", [])
    if not isinstance(criteria, list):
        return []
    return [item for item in criteria if isinstance(item, dict)]


def _options_summary_row(options: dict[str, Any]) -> dict[str, Any]:
    call_diag = _mapping(options.get("call_diagnostics"))
    return {
        "spot": options.get("spot"),
        "strike": options.get("strike"),
        "volatility": options.get("volatility"),
        "BSM call": options.get("black_scholes_call"),
        "BSM put": options.get("black_scholes_put"),
        "CRR call": options.get("binomial_call"),
        "CRR put": options.get("binomial_put"),
        "parity gap": options.get("put_call_parity_gap"),
        "call time value": call_diag.get("time_value"),
        "call breakeven": call_diag.get("breakeven_at_maturity"),
    }


def _render_home(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ANN001
    settings = snapshot["settings"]
    provider_summary = snapshot["provider_summary"]
    universe = snapshot["universe"]
    capital = settings["safety"]["initial_capital"]
    frequency = settings["research"]["default_frequency"]

    st.subheader("Inicio")
    _kpi_row(
        st,
        [
            ("Modo", "Research-only"),
            ("Capital simulado", _money(capital, settings["safety"]["base_currency"])),
            ("Frecuencia", str(frequency)),
            ("Providers listos", str(provider_summary["available"])),
            ("Datasets locales", str(len(snapshot["datasets"]))),
            ("Reports locales", str(len(snapshot["reports"]))),
        ],
    )
    st.markdown(
        """
        Esta consola muestra el estado de la plataforma, datos locales, calidad, backtests y
        formulas. Es una interfaz de inspeccion: no lanza ordenes, no hace trading y no ejecuta
        descargas de red automaticamente.
        """
    )
    if any(_is_demo_artifact(item) for item in [*snapshot["datasets"], *snapshot["reports"]]):
        st.warning("Se detectaron artefactos demo marcados como DEMO_SYNTHETIC_NOT_REAL_DATA.")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(provider_status_figure(snapshot["providers"]), use_container_width=True)
    with right:
        st.plotly_chart(universe_mix_figure(universe), use_container_width=True)

    st.markdown("### Comandos seguros desde terminal")
    st.caption("La UI no ejecuta estos comandos por ti; se muestran para ejecucion explicita.")
    st.dataframe(_safe_df(pd, safe_command_catalog()), use_container_width=True, hide_index=True)
    with st.expander("Estado UI JSON publico"):
        st.json(status)


def _render_quant_terminal(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Professional Quant Terminal")
    reports = [
        row
        for row in snapshot["reports"]
        if row.get("report_type") == "professional_quant_terminal"
    ]
    if not reports:
        _empty_state(st, "No hay professional quant terminal report local.", _terminal_commands())
        st.info(
            "Genera el JSON primero. La UI no descarga datos ni ejecuta simulaciones por si sola."
        )
        return
    report_row = _select_report(st, reports, "Professional terminal report")
    report = read_json_report(report_row["path"])
    universe = report.get("universe", {})
    data = report.get("data", {})
    warnings = data.get("warnings", []) if isinstance(data, dict) else []
    _kpi_row(
        st,
        [
            ("Stocks", ", ".join(universe.get("selected_stocks", []))),
            ("Benchmark", str(universe.get("benchmark", "N/A"))),
            ("Data mode", str(data.get("mode", "unknown"))),
            ("Observations", str(data.get("aligned_observations", "N/A"))),
            ("Warnings", str(len(warnings))),
        ],
    )
    if warnings:
        st.warning(" | ".join(str(item) for item in warnings))

    tabs = st.tabs(
        [
            "A Stock Detail",
            "B Portfolio",
            "C Monte Carlo",
            "D VaR",
            "E Backtesting",
            "F Options",
            "G Fixed Income & Rates",
            "H Hedging & Exposure",
            "I Spreadsheet & Methods",
        ]
    )
    with tabs[0]:
        _render_terminal_stock_detail(st, pd, report)
    with tabs[1]:
        _render_terminal_portfolio(st, pd, report)
    with tabs[2]:
        _render_terminal_monte_carlo(st, pd, report)
    with tabs[3]:
        _render_terminal_var(st, pd, report)
    with tabs[4]:
        _render_terminal_backtesting(st, pd, report)
    with tabs[5]:
        _render_terminal_options(st, pd, report)
    with tabs[6]:
        _render_terminal_fixed_income_rates(st, pd, report)
    with tabs[7]:
        _render_terminal_hedging_exposure(st, pd, report)
    with tabs[8]:
        _render_terminal_methods(st, pd, report)


def _render_academic_reports(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Academic Reports")
    reports = [
        row for row in snapshot["reports"] if row.get("report_type") == "academic_stock_report"
    ]
    if not reports:
        _empty_state(
            st, "No hay informes academicos por stock generados.", _academic_report_commands()
        )
        st.info(
            "La UI solo inspecciona documentos existentes. Usa el CLI explicito para generarlos."
        )
        return
    labels = [
        f"{row.get('summary', {}).get('asset_id', row['name'])} | {row['name']}" for row in reports
    ]
    selected = st.selectbox("Stock academic report", labels)
    report_row = reports[labels.index(selected)]
    metadata = read_json_report(report_row["path"])
    summary = metadata.get("summary", "")
    outputs = metadata.get("outputs", {}) if isinstance(metadata.get("outputs"), dict) else {}
    figures = metadata.get("figures", {}) if isinstance(metadata.get("figures"), dict) else {}
    _kpi_row(
        st,
        [
            ("Stock", str(metadata.get("asset_id", "N/A"))),
            ("Formats", ", ".join(str(fmt) for fmt in metadata.get("formats", []))),
            ("Figures", str(len(figures))),
            ("Research-only", str(metadata.get("research_only", True))),
        ],
    )
    st.markdown("### Resumen")
    st.write(summary)
    st.markdown("### Rutas")
    st.dataframe(_safe_df(pd, outputs), use_container_width=True, hide_index=True)
    if outputs.get("html"):
        st.info(
            "Para abrir el HTML local, abre esta ruta desde el navegador o explorador de archivos:"
        )
        st.code(str(outputs["html"]), language="text")
    st.markdown("### Figuras generadas")
    figure_rows = [{"figure": name, "path": path} for name, path in figures.items()]
    st.dataframe(_safe_df(pd, figure_rows), use_container_width=True, hide_index=True)
    st.markdown("### Regenerar de forma explicita")
    for command in _academic_report_commands(asset=str(metadata.get("asset_id", "AAPL"))):
        st.code(command, language="powershell")
    with st.expander("Metadata JSON"):
        st.json(metadata)


def _render_terminal_stock_detail(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### A. Stock Individual")
    assets = report.get("single_assets", {})
    symbols = sorted(str(symbol) for symbol in assets) if isinstance(assets, dict) else []
    if not symbols:
        st.warning("No stock analytics in report.")
        return
    symbol = st.selectbox("Stock", symbols)
    payload = assets[symbol]
    _kpi_row(
        st,
        [
            ("Annual return", _format_metric(payload.get("annualized_return"))),
            ("Annual vol", _format_metric(payload.get("annualized_volatility"))),
            ("Sharpe", _format_metric(payload.get("sharpe_ratio"))),
            ("Beta", _format_metric(payload.get("beta_to_benchmark"))),
            ("Max DD", _format_metric(payload.get("max_drawdown"))),
        ],
    )
    st.plotly_chart(terminal_price_figure(report), use_container_width=True)
    metric_payload = {key: value for key, value in payload.items() if key != "series"}
    st.dataframe(_safe_df(pd, metric_payload), use_container_width=True, hide_index=True)


def _render_terminal_portfolio(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### B. Portfolio 3 Stocks")
    portfolio = report.get("portfolio", {})
    optimization = report.get("optimization", {})
    equal_weight = portfolio.get("equal_weight", {}) if isinstance(portfolio, dict) else {}
    max_sharpe = optimization.get("max_sharpe", {}) if isinstance(optimization, dict) else {}
    _kpi_row(
        st,
        [
            ("EW return", _format_metric(equal_weight.get("annualized_return"))),
            ("EW vol", _format_metric(equal_weight.get("annualized_volatility"))),
            ("EW Sharpe", _format_metric(equal_weight.get("sharpe_ratio"))),
            ("Max-Sharpe vol", _format_metric(max_sharpe.get("annualized_volatility"))),
            ("Grid portfolios", str(optimization.get("portfolio_count", "N/A"))),
        ],
    )
    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(terminal_correlation_heatmap(report), use_container_width=True)
    with right:
        st.plotly_chart(terminal_frontier_figure(report), use_container_width=True)
    st.markdown("#### Optimized weights")
    st.dataframe(
        _safe_df(
            pd,
            [
                {
                    "portfolio": "min_variance",
                    **optimization.get("min_variance", {}).get("weights", {}),
                },
                {
                    "portfolio": "max_sharpe",
                    **optimization.get("max_sharpe", {}).get("weights", {}),
                },
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_terminal_monte_carlo(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### C. Monte Carlo")
    mc = report.get("monte_carlo", {})
    _kpi_row(
        st,
        [
            ("Paths", str(mc.get("path_count", "N/A"))),
            ("Horizon", str(mc.get("horizon_days", "N/A"))),
            ("Terminal mean", _format_metric(mc.get("terminal_mean"))),
            ("Terminal p05", _format_metric(mc.get("terminal_p05"))),
            ("Terminal p95", _format_metric(mc.get("terminal_p95"))),
        ],
    )
    st.plotly_chart(terminal_monte_carlo_fan_figure(report), use_container_width=True)
    st.caption("Modelo parametrico/educativo; no predice precios futuros.")


def _render_terminal_var(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### D. VaR")
    st.plotly_chart(terminal_var_figure(report), use_container_width=True)
    st.dataframe(
        _safe_df(pd, _flatten_var_rows(report.get("var", {}))),
        use_container_width=True,
        hide_index=True,
    )
    st.info("Convencion: VaR y ES se muestran como perdidas positivas, no retornos negativos.")


def _render_terminal_backtesting(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### E. Backtesting")
    st.plotly_chart(terminal_backtest_equity_figure(report), use_container_width=True)
    rows = []
    for strategy, payload in report.get("backtesting", {}).items():
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        rows.append({"strategy": strategy, **metrics})
    st.dataframe(_safe_df(pd, rows), use_container_width=True, hide_index=True)
    st.caption("El motor aplica execution lag t+1 para reducir look-ahead bias.")


def _render_terminal_options(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### F. Options")
    st.plotly_chart(terminal_options_figure(report), use_container_width=True)
    options = report.get("options", {})
    rows = []
    for symbol, payload in options.items():
        rows.append(
            {
                "symbol": symbol,
                **{key: value for key, value in payload.items() if key != "payoff_profile"},
            }
        )
    st.dataframe(_safe_df(pd, rows), use_container_width=True, hide_index=True)
    st.warning("Sin option chain real: Black-Scholes, Greeks y CRR son modelos parametrizados.")


def _render_terminal_fixed_income_rates(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### G. Fixed Income & Rates")
    st.dataframe(
        _safe_df(pd, report.get("fixed_income", {})), use_container_width=True, hide_index=True
    )
    rates = report.get("rates_derivatives", {})
    st.dataframe(_safe_df(pd, rates.get("swap", {})), use_container_width=True, hide_index=True)
    st.dataframe(
        _safe_df(pd, rates.get("sofr_futures", {})), use_container_width=True, hide_index=True
    )
    st.warning("Pricing real requiere curvas, calendarios, convenciones y contract specs reales.")


def _render_terminal_hedging_exposure(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### H. Hedging & Exposure")
    st.plotly_chart(terminal_exposure_figure(report), use_container_width=True)
    st.dataframe(_safe_df(pd, report.get("hedging", {})), use_container_width=True, hide_index=True)
    exposure = report.get("exposure", {})
    st.dataframe(
        _safe_df(pd, {key: value for key, value in exposure.items() if key != "profile"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Hedge y exposure son parametrizados; no hay contratos ni CSA reales.")


def _render_terminal_methods(st, pd, report: dict[str, Any]) -> None:  # noqa: ANN001
    st.markdown("### I. Spreadsheet / Export / Bibliography")
    exports = report.get("exports", {})
    st.info(f"Frontier CSV: `{exports.get('frontier_csv', 'N/A')}`")
    st.dataframe(_safe_df(pd, report.get("methods", [])), use_container_width=True, hide_index=True)
    st.dataframe(_safe_df(pd, report.get("safety", {})), use_container_width=True, hide_index=True)
    with st.expander("Professional terminal report JSON"):
        st.json(report)


def _render_conceptual_flow(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Flujo conceptual")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", color="#d6b35a"];
          Providers -> OHLCV -> "Quality checks" -> Registry -> Returns;
          Returns -> Weights -> Backtest -> "Risk metrics" -> Reports -> "Visual UI";
        }
        """
    )
    st.dataframe(
        _safe_df(pd, snapshot["conceptual_flow"]),
        use_container_width=True,
        hide_index=True,
    )
    _concept_cards(st)


def _render_system_status(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ANN001
    st.subheader("Estado del sistema")
    _kpi_row(
        st,
        [
            ("UI available", str(status["ui_available"])),
            ("Streamlit", str(status["streamlit_installed"])),
            ("Plotly", str(status["plotly_installed"])),
            ("Network auto-run", str(status["network_auto_run"])),
            ("No secrets", str(status["no_secrets_exposed"])),
        ],
    )
    st.dataframe(_safe_df(pd, [status]), use_container_width=True, hide_index=True)
    with st.expander("Public settings snapshot"):
        st.json(snapshot["settings"])


def _render_providers(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Providers")
    st.write(
        "Configured significa que existe credencial local. Enabled significa que la politica "
        "local permite usar el provider. configured_but_disabled es seguro: la clave existe, "
        "pero el provider esta apagado."
    )
    st.plotly_chart(provider_status_figure(snapshot["providers"]), use_container_width=True)
    st.dataframe(_safe_df(pd, snapshot["providers"]), use_container_width=True, hide_index=True)
    with st.expander("Como leer los estados"):
        st.markdown(
            """
            - `available`: listo para uso read-only.
            - `configured_but_disabled`: hay credencial local, pero el flag esta apagado.
            - `missing_api_key`: se quiere usar, pero falta credencial.
            - `disabled`: apagado por configuracion.

            La UI nunca muestra valores de claves ni ultimos caracteres.
            """
        )


def _render_universe(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Universo")
    universe = snapshot["universe"]
    st.write(
        "El universo de investigacion combina ETFs liquidos, watchlists de equity e indices con "
        "crypto majors spot. No es una recomendacion de inversion."
    )
    st.plotly_chart(universe_mix_figure(universe), use_container_width=True)
    st.dataframe(_safe_df(pd, universe["groups"]), use_container_width=True, hide_index=True)
    st.info(
        "ETFs liquidos ayudan a empezar con datos mas estables. Crypto spot es mas volatil y "
        "opera 24/7, por eso sus gaps se interpretan distinto."
    )


def _render_datasets(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Datasets")
    datasets = snapshot["datasets"]
    if not datasets:
        _empty_state(st, "No hay manifests locales.", _dataset_commands())
        return
    st.dataframe(
        _safe_df(pd, dataset_quality_rows(datasets)),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(_safe_df(pd, datasets), use_container_width=True, hide_index=True)
    with st.expander("Rutas locales"):
        st.write("Los datasets viven bajo `data/registry/` y estan ignorados por Git.")


def _render_quality(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Calidad de datos")
    datasets = snapshot["datasets"]
    if not datasets:
        _empty_state(st, "No hay datasets para revisar calidad.", _dataset_commands())
        return
    labels = [f"{row['dataset_id']} / {row['version']}" for row in datasets]
    selected = st.selectbox("Dataset", labels)
    dataset = datasets[labels.index(selected)]
    report = load_dataset_quality_report(dataset)
    if report is None:
        st.warning("Este dataset no tiene data_quality_report.json local.")
        return
    cols = st.columns(4)
    cols[0].metric("Symbols total", report.get("symbols_total", 0))
    cols[1].metric("Successful", len(report.get("symbols_successful", [])))
    cols[2].metric("Failed", len(report.get("symbols_failed", [])))
    cols[3].metric("Suitable demo", str(report.get("suitable_for_backtest_demo")))
    st.plotly_chart(quality_coverage_figure(report), use_container_width=True)
    st.plotly_chart(data_quality_warnings_figure(report), use_container_width=True)
    st.dataframe(
        _safe_df(pd, report.get("coverage_by_asset", [])),
        use_container_width=True,
        hide_index=True,
    )
    if report.get("date_gaps"):
        st.markdown("### Gaps")
        st.dataframe(_safe_df(pd, report["date_gaps"]), use_container_width=True, hide_index=True)
    with st.expander("Quality report JSON"):
        st.json(report)


def _render_backtests(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Backtests")
    reports = [row for row in snapshot["reports"] if row.get("report_type") == "backtest"]
    if not reports:
        _empty_state(st, "No hay backtest reports locales.", _backtest_commands())
        return
    report_row = _select_report(st, reports, "Backtest report")
    report = read_json_report(report_row["path"])
    metric_rows = backtest_metric_rows(report)
    cols = st.columns(5)
    for col, key in zip(
        cols,
        (
            "final_equity",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ),
        strict=False,
    ):
        row = next(item for item in metric_rows if item["metric"] == key)
        col.metric(row["label"], _format_metric(row["value"]))
    st.plotly_chart(backtest_metric_figure(report), use_container_width=True)
    st.dataframe(_safe_df(pd, metric_rows), use_container_width=True, hide_index=True)

    equity_series = report_series(report, "equity_curve")
    drawdown_series = report_series(report, "drawdown_curve")
    returns = report_series(report, "net_returns", "returns")
    if equity_series or drawdown_series or returns:
        st.plotly_chart(equity_curve_figure(equity_series), use_container_width=True)
        st.plotly_chart(drawdown_curve_figure(drawdown_series), use_container_width=True)
        st.plotly_chart(returns_histogram_figure(returns), use_container_width=True)
    else:
        st.info(
            "Este report contiene metricas agregadas, pero no series temporales. Las curvas "
            "requieren guardar equity_curve/net_returns en un pipeline posterior."
        )
    st.plotly_chart(
        transaction_cost_breakdown_figure(transaction_cost_components(report)),
        use_container_width=True,
    )
    st.plotly_chart(
        var_es_conceptual_figure(
            report.get("historical_var_95"),
            report.get("historical_expected_shortfall_95"),
        ),
        use_container_width=True,
    )
    with st.expander("Backtest report JSON"):
        st.json(report)


def _render_profile_comparison(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Comparacion de perfiles")
    reports = [row for row in snapshot["reports"] if row.get("report_type") == "profile_comparison"]
    if not reports:
        _empty_state(st, "No hay profile comparison report local.", _comparison_commands())
        return
    report_row = _select_report(st, reports, "Profile comparison")
    report = read_json_report(report_row["path"])
    st.write(
        "Conservative y aggressive son perfiles de evaluacion. Sus targets de 15% y 30% "
        "son umbrales, no garantias. Una estrategia agresiva puede no ganar mas."
    )
    st.plotly_chart(profile_comparison_figure(report), use_container_width=True)
    st.dataframe(
        _safe_df(pd, report.get("comparison_table", [])),
        use_container_width=True,
        hide_index=True,
    )
    if report.get("warnings"):
        st.warning(" | ".join(str(item) for item in report["warnings"]))
    with st.expander("Profile comparison JSON"):
        st.json(report)


def _render_formulas(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Formulas")
    st.write("Formulas usadas o preparadas para entender retornos, riesgo y backtests.")
    for formula in all_formulas():
        with st.expander(formula["name"], expanded=False):
            st.latex(formula["latex"])
            st.write(formula["plain_explanation"])
            st.info(formula["intuition"])
            st.markdown(f"**Inputs:** {', '.join(formula['inputs'])}")
            st.markdown(f"**Output:** {formula['outputs']}")
            st.markdown(f"**Se usa en:** {formula['used_in']}")
            st.warning(f"Mala interpretacion comun: {formula['common_misinterpretation']}")
            st.caption(formula["example_short"])


def _render_actions(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Acciones permitidas")
    catalog = action_catalog()
    st.markdown("### Permitidas")
    st.dataframe(
        _safe_df(pd, catalog["allowed_actions"]),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### Prohibidas")
    st.dataframe(
        _safe_df(pd, catalog["prohibited_actions"]),
        use_container_width=True,
        hide_index=True,
    )
    st.error("La UI no ejecuta trading, paper trading, ordenes, brokers ni endpoints privados.")


def _render_risks(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Riesgos y limites")
    st.warning("No es asesoramiento financiero. Backtest no implica resultados futuros.")
    st.info("Datos demo, si existen, se marcan como DEMO_SYNTHETIC_NOT_REAL_DATA.")
    st.dataframe(_safe_df(pd, all_explainers()), use_container_width=True, hide_index=True)
    with st.expander("Mensajes clave"):
        st.markdown(
            """
            - Graficos sin contexto pueden inducir conclusiones incorrectas.
            - Datos reales o generados viven bajo rutas ignoradas y no deben commitearse.
            - Providers gratuitos pueden tener limites, licencias, gaps y revisiones.
            - Trials se registran para reducir overfitting, no para eliminarlo por completo.
            """
        )


def _concept_cards(st) -> None:  # noqa: ANN001
    explainers = all_explainers()[:6]
    cols = st.columns(3)
    for index, explainer in enumerate(explainers):
        with cols[index % 3]:
            st.markdown(f"### {explainer['title']}")
            st.write(explainer["summary"])
            st.caption(explainer["why_it_matters"])


def _kpi_row(st, items: list[tuple[str, str]]) -> None:  # noqa: ANN001
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items, strict=True):
        col.metric(label, value)


def _select_report(st, reports: list[dict[str, Any]], label: str) -> dict[str, Any]:  # noqa: ANN001
    labels = [f"{row['report_type']} | {row['name']}" for row in reports]
    selected = st.selectbox(label, labels)
    return reports[labels.index(selected)]


def _empty_state(st, message: str, commands: list[str]) -> None:  # noqa: ANN001
    st.warning(message)
    st.write("Puedes generar artefactos locales con comandos explicitos desde terminal:")
    for command in commands:
        st.code(command, language="powershell")


def _dataset_commands() -> list[str]:
    return [
        (
            "py -3 -m quant_platform.cli download-real-data --config "
            "configs/universe_etfs_crypto_daily.yaml --start 2024-01-01 --end 2024-03-31 "
            "--limit-equity 3 --limit-crypto 2 --dataset-id real_daily_demo --version v1"
        )
    ]


def _backtest_commands() -> list[str]:
    return [
        "py -3 -m quant_platform.cli run-backtest-demo --dataset-id real_daily_demo --version v1",
    ]


def _comparison_commands() -> list[str]:
    return [
        (
            "py -3 -m quant_platform.cli compare-profiles --dataset-id real_daily_demo "
            "--version v1 --config configs/universe_etfs_crypto_daily.yaml"
        )
    ]


def _terminal_commands() -> list[str]:
    return [
        (
            "py -3 -m quant_platform.cli build-quant-terminal-report --config "
            "configs/quant_terminal_10_stocks.yaml"
        ),
        (
            "py -3 -m quant_platform.cli build-quant-terminal-report --config "
            "configs/quant_terminal_10_stocks.yaml --offline-synthetic"
        ),
    ]


def _academic_report_commands(asset: str = "AAPL") -> list[str]:
    return [
        (
            "py -3 -m quant_platform.cli generate-stock-academic-report "
            f"--asset {asset} --terminal-report "
            "reports/generated/quant_terminal/10stocks_10y_report.json "
            "--output-dir reports/generated/academic_stock_reports "
            "--format md --format html --include-figures --overwrite"
        ),
        (
            "py -3 -m quant_platform.cli generate-all-stock-academic-reports "
            "--terminal-report reports/generated/quant_terminal/10stocks_10y_report.json "
            "--output-dir reports/generated/academic_stock_reports "
            "--format md --format html --include-figures --overwrite"
        ),
    ]


def _flatten_var_rows(var_payload: object) -> list[dict[str, object]]:
    if not isinstance(var_payload, dict):
        return []
    rows = []
    for model, payload in var_payload.items():
        if not isinstance(payload, dict):
            continue
        if model in {"alpha", "loss_sign_convention"}:
            continue
        rows.append(
            {
                "model": model,
                "var": payload.get("var"),
                "expected_shortfall": payload.get("expected_shortfall"),
                "model_status": payload.get("model_status"),
            }
        )
    return rows


def _safe_df(pd, rows: object):  # noqa: ANN001, ANN202
    text = json.loads(json.dumps(rows, default=str))
    if isinstance(text, dict):
        text = [text]
    normalized = []
    for row in text if isinstance(text, list) else []:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                key: ", ".join(str(item) for item in value)
                if isinstance(value, list)
                else json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in row.items()
            }
        )
    return pd.DataFrame(normalized)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: object) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def _num(value: object) -> str:
    try:
        return f"{float(value):,.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_metric(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        clean = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(clean) < 1:
        return f"{clean:.2%}"
    return f"{clean:,.2f}"


def _money(value: object, currency: str) -> str:
    try:
        return f"{float(value):,.0f} {currency}"
    except (TypeError, ValueError):
        return f"{value} {currency}"


def _is_demo_artifact(value: object) -> bool:
    return "DEMO_SYNTHETIC_NOT_REAL_DATA" in json.dumps(value, default=str)


def _inject_style(st) -> None:  # noqa: ANN001
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(214, 179, 90, 0.18), transparent 28rem),
                radial-gradient(circle at 85% 15%, rgba(61, 214, 198, 0.14), transparent 30rem),
                linear-gradient(135deg, #091014 0%, #111820 48%, #15110b 100%);
            color: #e8ecef;
        }
        .hero {
            display: flex;
            justify-content: space-between;
            gap: 2rem;
            padding: 2rem;
            border: 1px solid rgba(214, 179, 90, 0.28);
            border-radius: 28px;
            background: rgba(8, 13, 18, 0.78);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #d6b35a;
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }
        .hero h1 { margin: 0; font-size: 2.4rem; }
        .hero-copy { max-width: 52rem; color: #c8d0d8; font-size: 1.05rem; }
        .hero-badge {
            min-width: 170px;
            align-self: center;
            padding: 1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #d6b35a, #3dd6c6);
            color: #071014;
            font-weight: 800;
            text-align: center;
        }
        .ticker-note {
            border: 1px solid rgba(61, 214, 198, 0.30);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            background: rgba(7, 16, 20, 0.72);
            color: #d9f6f2;
            font-size: 0.98rem;
        }
        .paper-card {
            border: 1px solid rgba(214, 179, 90, 0.35);
            border-radius: 20px;
            padding: 1rem 1.2rem;
            background: rgba(12, 18, 24, 0.82);
            margin: 0.75rem 0;
        }
        [data-testid="stMetric"] {
            background: rgba(12, 18, 24, 0.78);
            border: 1px solid rgba(214, 179, 90, 0.28);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1117 0%, #111820 100%);
        }
        .stDataFrame { border-radius: 18px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
