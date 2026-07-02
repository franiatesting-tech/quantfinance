"""Reproducible study for the seven core quant algorithms."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import html
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.reporting.pdf_export import export_html_report_to_pdf
from quant_platform.research.state_of_art.garch_volatility import fit_garch_11
from quant_platform.research.state_of_art.kalman_filter import kalman_dynamic_regression
from quant_platform.research.state_of_art.ornstein_uhlenbeck import (
    pairs_zscore_backtest,
    screen_ou_pairs,
)

RESEARCH_ONLY_DISCLAIMER = (
    "This is not investment advice. This is a research-only quantitative signal based on "
    "historical data, assumptions and model limitations."
)

ALGORITHMS = [
    ("01_black_scholes", "Black-Scholes option pricing"),
    ("02_monte_carlo", "Monte Carlo simulation"),
    ("03_markowitz", "Markowitz mean-variance optimization"),
    ("04_pairs_trading_ou", "Pairs trading with Ornstein-Uhlenbeck mean reversion"),
    ("05_kalman_filter", "Kalman filter dynamic state estimation"),
    ("06_garch", "GARCH volatility modeling"),
    ("07_ml_alpha", "Machine learning for alpha with walk-forward validation"),
]


class SevenAlgorithmsStudyError(ValueError):
    """Raised when the seven-algorithm study cannot be built safely."""


def build_seven_algorithm_study(
    terminal_report: dict[str, Any],
    *,
    bibliography_pack_dir: str | Path | None = None,
    strict_real_data: bool = True,
    max_table_rows: int = 30,
) -> dict[str, Any]:
    """Build JSON-safe tables for the seven core algorithms from a terminal report."""

    if terminal_report.get("report_type") != "professional_quant_terminal":
        raise SevenAlgorithmsStudyError("terminal_report must be a professional quant terminal JSON.")
    data = _mapping(terminal_report.get("data"))
    data_mode = str(data.get("mode", "unknown"))
    if strict_real_data and not data_mode.startswith("provider_"):
        raise SevenAlgorithmsStudyError(f"strict real-data study requires provider data, got {data_mode!r}.")
    prices = _price_frame_from_terminal_report(terminal_report)
    returns = _return_frame_from_terminal_report(terminal_report)
    if prices.empty or returns.empty:
        raise SevenAlgorithmsStudyError("terminal report does not contain usable price/return series.")
    representative_asset = "AAPL" if "AAPL" in prices.columns else str(prices.columns[0])
    pair_rows = screen_ou_pairs(prices, max_pairs=max(10, max_table_rows))
    best_pair = pair_rows[0] if pair_rows else None
    pair_backtest = pairs_zscore_backtest(prices, best_pair) if best_pair else {}
    kalman = _kalman_payload(prices, best_pair)
    garch_rows, garch_series = _garch_payload(returns)
    bibliography_rows = _bibliography_rows(bibliography_pack_dir)
    tables = {
        "data_quality_gate": _data_quality_gate(terminal_report, prices),
        "algorithm_implementation_map": _algorithm_implementation_map(),
        "black_scholes_replication": _black_scholes_table(terminal_report),
        "monte_carlo_replication": _monte_carlo_table(terminal_report),
        "markowitz_replication": _markowitz_table(terminal_report),
        "ou_pair_screen": pair_rows[:max_table_rows],
        "ou_pair_backtest": [_without_series(pair_backtest)] if pair_backtest else [],
        "kalman_dynamic_hedge": [_without_series(kalman)] if kalman else [],
        "garch_volatility": garch_rows[:max_table_rows],
        "ml_alpha_validation": _ml_alpha_table(terminal_report),
        "broker_readiness_decisions": _broker_readiness_table(data_mode, terminal_report, pair_rows, garch_rows),
        "bibliography_trace": bibliography_rows,
        "manual_data_requirements": _manual_data_requirements(),
    }
    warnings = _unique_strings(list(terminal_report.get("warnings", [])) + list(data.get("warnings", [])))
    return {
        "report_type": "seven_quant_algorithms_research_study",
        "report_version": "seven_algorithms_v1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "not_investment_advice": True,
        "disclaimer": RESEARCH_ONLY_DISCLAIMER,
        "source_terminal_report": {
            "report_type": terminal_report.get("report_type"),
            "data_mode": data_mode,
            "warnings": warnings,
            "aligned_observations": data.get("aligned_observations"),
            "start_timestamp": data.get("start_timestamp"),
            "end_timestamp": data.get("end_timestamp"),
        },
        "representative_asset": representative_asset,
        "representative_pair": best_pair,
        "series": {
            "ou_pair_backtest": pair_backtest.get("series", []) if pair_backtest else [],
            "kalman_dynamic_hedge": kalman.get("state_rows", []) if kalman else [],
            "garch_volatility": garch_series,
            "monte_carlo_fan": _representative_mc_fan(terminal_report, representative_asset),
            "black_scholes_payoff": _representative_payoff(terminal_report, representative_asset),
            "markowitz_frontier": _markowitz_frontier_rows(terminal_report, max_rows=500),
            "ml_predictions": _representative_ml_predictions(terminal_report, representative_asset),
        },
        "tables": tables,
    }


def write_seven_algorithm_study(
    terminal_report: dict[str, Any],
    *,
    output_dir: str | Path,
    bibliography_pack_dir: str | Path | None = None,
    strict_real_data: bool = True,
    max_table_rows: int = 30,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write study JSON, tables, figures, Markdown, HTML, optional PDF and metadata."""

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise SevenAlgorithmsStudyError(f"Output directory already contains files: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_generated_directory(output_path)
    tables_dir = output_path / "tables"
    figures_dir = output_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    study = build_seven_algorithm_study(
        terminal_report,
        bibliography_pack_dir=bibliography_pack_dir,
        strict_real_data=strict_real_data,
        max_table_rows=max_table_rows,
    )
    json_path = output_path / "seven_quant_algorithms_study.json"
    md_path = output_path / "seven_quant_algorithms_paper.md"
    html_path = output_path / "seven_quant_algorithms_paper.html"
    pdf_path = output_path / "seven_quant_algorithms_paper.pdf"
    table_paths = _write_tables(study["tables"], tables_dir)
    figure_paths = _write_figures(study, figures_dir)
    json_path.write_text(json.dumps(study, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(study, table_paths, figure_paths), encoding="utf-8")
    html_path.write_text(render_html(study, table_paths, figure_paths), encoding="utf-8")
    pdf_export = export_html_report_to_pdf(html_path, pdf_path)
    outputs = {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}
    if pdf_export.get("success") and pdf_export.get("pdf_path"):
        outputs["pdf"] = str(pdf_export["pdf_path"])
    else:
        pdf_export["status"] = "PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER"
    manifest = _manifest(study, outputs, table_paths, figure_paths)
    manifest_path = output_path / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "report_type": "seven_quant_algorithms_study_metadata",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "not_investment_advice": True,
        "outputs": {**outputs, "tables": table_paths, "figures": figure_paths},
        "pdf_export": pdf_export,
        "reproducibility_manifest": str(manifest_path),
        "warnings": study["source_terminal_report"].get("warnings", []),
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def render_markdown(
    study: dict[str, Any],
    table_paths: dict[str, str] | None = None,
    figure_paths: dict[str, str] | None = None,
) -> str:
    """Render a professional Markdown paper for the seven algorithms."""

    tables = _mapping(study.get("tables"))
    lines = [
        "# Replication Study: The Seven Algorithms Every Quant Should Master",
        "",
        f"**{RESEARCH_ONLY_DISCLAIMER}**",
        "",
        "## Executive Summary",
        "",
        "This paper implements the seven algorithms from the supplied guide using the platform's real-data terminal report. The output is a research replication and broker-readiness assessment, not a broker instruction.",
        "",
    ]
    for title, table_names in _paper_sections():
        lines.extend([f"## {title}", "", _section_text(title, study), ""])
        for table_name in table_names:
            lines.extend([f"### Table: {table_name}", "", _markdown_table(tables.get(table_name, [])), ""])
        figure_path = (figure_paths or {}).get(_figure_key_for_title(title))
        if figure_path:
            lines.extend([f"Figure: `{figure_path}`", ""])
    if table_paths:
        lines.extend(["## Reproducibility", ""])
        lines.extend(f"- `{name}`: `{path}`" for name, path in sorted(table_paths.items()))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_html(
    study: dict[str, Any],
    table_paths: dict[str, str] | None = None,
    figure_paths: dict[str, str] | None = None,
) -> str:
    """Render journal-style HTML suitable for PDF export."""

    tables = _mapping(study.get("tables"))
    parts = []
    for title, table_names in _paper_sections():
        parts.append(f"<h2>{html.escape(title)}</h2>")
        parts.append(f"<p>{html.escape(_section_text(title, study))}</p>")
        for table_name in table_names:
            parts.append(f"<h3>Table: {html.escape(table_name)}</h3>")
            parts.append(_html_table(tables.get(table_name, [])))
        figure_path = (figure_paths or {}).get(_figure_key_for_title(title))
        if figure_path:
            parts.append(f"<h3>Figure: {html.escape(title)}</h3>")
            parts.append(_embedded_html(Path(figure_path)))
    if table_paths:
        parts.append("<h2>Reproducibility</h2>")
        parts.append(_path_list(table_paths))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Seven Quant Algorithms Replication Study</title>
  <style>
    @page {{ size: A4; margin: 16mm 14mm 18mm; }}
    body {{ margin:0; background:#101827; color:#172033; font-family: Georgia, 'Times New Roman', serif; line-height:1.52; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 34px 26px 72px; background:#fffdf7; }}
    h1 {{ font-family: Aptos, Segoe UI, sans-serif; font-size: 34px; letter-spacing:-0.03em; border-bottom:4px solid #14b8a6; padding-bottom:12px; }}
    h2 {{ font-family: Aptos, Segoe UI, sans-serif; font-size: 22px; margin-top:32px; break-after: avoid; }}
    h2:not(:first-child) {{ break-before: page; }}
    h3 {{ font-family: Aptos, Segoe UI, sans-serif; font-size: 15px; color:#115e59; }}
    p {{ text-align: justify; }}
    table {{ border-collapse: collapse; width:100%; margin: 12px 0 18px; font-size: 10.5px; break-inside: avoid; }}
    th, td {{ border:1px solid #cbd5e1; padding:6px 7px; vertical-align: top; }}
    th {{ background:#ccfbf1; font-family: Aptos, Segoe UI, sans-serif; }}
    .disclaimer {{ border:1px solid #9f1239; background:#fff1f2; color:#7f1d1d; padding:10px 12px; border-radius:8px; font-weight:700; }}
    .figure-box {{ border:1px solid #94a3b8; background:#ffffff; padding:10px; margin:14px 0 18px; break-inside: avoid; overflow:hidden; }}
    .small {{ font-size: 12px; color:#475569; }}
  </style>
</head>
<body><main>
<h1>Replication Study: The Seven Algorithms Every Quant Should Master</h1>
<div class="disclaimer">{html.escape(RESEARCH_ONLY_DISCLAIMER)}</div>
{''.join(parts)}
</main></body></html>"""


def _data_quality_gate(report: dict[str, Any], prices: pd.DataFrame) -> list[dict[str, Any]]:
    data = _mapping(report.get("data"))
    warnings = _unique_strings(list(report.get("warnings", [])) + list(data.get("warnings", [])))
    return [
        {
            "data_mode": data.get("mode"),
            "provider_data": str(data.get("mode", "")).startswith("provider_"),
            "aligned_observations": int(data.get("aligned_observations", len(prices))),
            "asset_count": int(len(prices.columns)),
            "start_timestamp": data.get("start_timestamp"),
            "end_timestamp": data.get("end_timestamp"),
            "warning_count": len(warnings),
            "warnings": "; ".join(str(item) for item in warnings),
            "study_quality_status": "REAL_DATA_WITH_WARNINGS" if warnings else "REAL_DATA_NO_WARNINGS",
        }
    ]


def _algorithm_implementation_map() -> list[dict[str, str]]:
    return [
        _algo_row("01_black_scholes", "Closed-form BSM plus Greeks and payoff scenarios", "Replicated with terminal option analytics", "Requires real option chain calibration before live option trading"),
        _algo_row("02_monte_carlo", "Parametric normal, GBM and bootstrap simulations", "Replicated with historical daily provider returns", "Needs model-risk limits and stress scenarios before broker use"),
        _algo_row("03_markowitz", "Long-only grid efficient frontier", "Replicated with historical returns/covariance", "Needs robust optimizer, constraints and execution model before broker use"),
        _algo_row("04_pairs_trading_ou", "OLS spread plus OU AR(1) mean-reversion proxy", "Implemented from provider close prices", "Needs borrow, shorting, locate, slippage and regime controls before broker use"),
        _algo_row("05_kalman_filter", "Dynamic hedge-ratio state estimation", "Implemented for selected OU pair", "Needs intraday execution and hedge rebalance controls before broker use"),
        _algo_row("06_garch", "Gaussian GARCH(1,1) volatility forecast", "Implemented with constrained ML estimation", "Needs distributional backtesting and calibrated risk limits before broker use"),
        _algo_row("07_ml_alpha", "Walk-forward ML with leakage controls", "Replicated from terminal ML diagnostics", "Broker promotion blocked unless strict OOS edge survives costs and multiple testing"),
    ]


def _black_scholes_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for asset, stock in _mapping(report.get("stocks")).items():
        options = _mapping(stock.get("options_theoretical_analytics"))
        greeks = _mapping(options.get("call_greeks"))
        rows.append(
            {
                "asset_id": asset,
                "spot": options.get("spot"),
                "strike": options.get("strike"),
                "volatility": options.get("volatility"),
                "black_scholes_call": options.get("black_scholes_call"),
                "black_scholes_put": options.get("black_scholes_put"),
                "call_delta": greeks.get("delta"),
                "call_gamma": greeks.get("gamma"),
                "call_vega": greeks.get("vega"),
                "put_call_parity_gap": options.get("put_call_parity_gap"),
                "status": options.get("model_status"),
            }
        )
    return rows


def _monte_carlo_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for asset, stock in _mapping(report.get("stocks")).items():
        mc = _mapping(stock.get("monte_carlo"))
        for model in ("parametric_normal", "historical_bootstrap", "block_bootstrap", "gbm_baseline"):
            payload = _mapping(mc.get(model))
            if not payload:
                continue
            rows.append(
                {
                    "asset_id": asset,
                    "model": model,
                    "path_count": payload.get("path_count"),
                    "horizon_days": payload.get("horizon_days"),
                    "terminal_mean": payload.get("terminal_mean"),
                    "terminal_p05": payload.get("terminal_p05"),
                    "terminal_p95": payload.get("terminal_p95"),
                    "probability_of_loss": payload.get("probability_of_loss"),
                    "status": payload.get("model_status"),
                }
            )
    return rows


def _markowitz_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    optimization = _mapping(report.get("optimization"))
    rows = []
    for name in ("min_variance", "max_sharpe"):
        payload = _mapping(optimization.get(name))
        weights = _mapping(payload.get("weights"))
        rows.append(
            {
                "portfolio": name,
                "annualized_return": payload.get("annualized_return"),
                "annualized_volatility": payload.get("annualized_volatility"),
                "sharpe_ratio": payload.get("sharpe_ratio"),
                "largest_weight_asset": max(weights, key=weights.get) if weights else None,
                "largest_weight": max(weights.values()) if weights else None,
                "nonzero_positions": sum(1 for value in weights.values() if float(value) > 0.0),
                "method": optimization.get("method"),
                "grid_step": optimization.get("grid_step"),
            }
        )
    return rows


def _kalman_payload(prices: pd.DataFrame, best_pair: dict[str, Any] | None) -> dict[str, Any]:
    if not best_pair:
        return {}
    y = np.log(prices[str(best_pair["asset_y"])]).dropna()
    x = np.log(prices[str(best_pair["asset_x"])]).dropna()
    return kalman_dynamic_regression(y, x)


def _garch_payload(returns: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    series_rows = []
    for asset in returns.columns:
        try:
            result = fit_garch_11(returns[asset])
        except Exception as exc:  # noqa: BLE001 - row captures model failure without aborting study.
            rows.append({"asset_id": asset, "status": "GARCH_FAILED", "error": str(exc)})
            continue
        rows.append(
            {
                "asset_id": asset,
                "observations": result["observations"],
                "alpha": result["alpha"],
                "beta": result["beta"],
                "persistence": result["persistence"],
                "half_life_days": result["half_life_days"],
                "forecast_volatility_annual": result["forecast_volatility_annual"],
                "status": result["model_status"],
            }
        )
        if not series_rows:
            series_rows = [{"asset_id": asset, **row} for row in result.get("volatility_series", [])]
    return rows, series_rows


def _ml_alpha_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for asset, payload in _mapping(report.get("ml_forecasting")).items():
        model = _mapping(payload)
        rows.append(
            {
                "asset_id": asset,
                "status": model.get("status"),
                "test_observations": model.get("test_observations"),
                "oos_r_squared": model.get("oos_r_squared"),
                "directional_accuracy": model.get("directional_accuracy"),
                "baseline_directional_accuracy": model.get("baseline_directional_accuracy"),
                "directional_accuracy_edge_vs_naive": model.get("directional_accuracy_edge_vs_naive"),
                "information_coefficient": model.get("information_coefficient"),
                "strategy_sharpe": model.get("strategy_sharpe"),
                "status_reason": model.get("status_reason"),
            }
        )
    return rows


def _broker_readiness_table(
    data_mode: str,
    report: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    garch_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    warnings = _unique_strings(
        list(report.get("warnings", [])) + list(_mapping(report.get("data")).get("warnings", []))
    )
    common_blockers = []
    if not data_mode.startswith("provider_"):
        common_blockers.append("real_provider_data_required")
    if warnings:
        common_blockers.append("provider_warnings_require_review")
    rows = []
    for algorithm, label in ALGORITHMS:
        blockers = list(common_blockers)
        if algorithm == "04_pairs_trading_ou" and not pair_rows:
            blockers.append("no_mean_reverting_pair_selected")
        if algorithm == "06_garch" and not any(row.get("status") == "GARCH_11_GAUSSIAN_RESEARCH_ESTIMATE" for row in garch_rows):
            blockers.append("garch_estimation_failed")
        if algorithm == "07_ml_alpha":
            blockers.append("strict_oos_cost_adjusted_edge_required")
            blockers.append("multiple_testing_control_required")
        blockers.append("broker_execution_controls_not_implemented")
        rows.append(
            {
                "algorithm": algorithm,
                "label": label,
                "research_status": "REPLICATED_FOR_RESEARCH" if len(blockers) <= 2 else "REPLICATED_WITH_BLOCKERS",
                "broker_action_allowed": False,
                "broker_decision": "BLOCKED_FOR_BROKER_PROMOTION",
                "blockers": "; ".join(dict.fromkeys(blockers)),
            }
        )
    return rows


def _bibliography_rows(pack_dir: str | Path | None) -> list[dict[str, Any]]:
    if pack_dir is None:
        return []
    root = Path(pack_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return []
    rows = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest if isinstance(manifest, list) else []:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", ""))
        pdf_paths = list((root / "papers").rglob(filename))
        rows.append(
            {
                "filename": filename,
                "title": item.get("title"),
                "authors": item.get("authors"),
                "year": item.get("year"),
                "source": item.get("source"),
                "topic_folder": str(pdf_paths[0].parent.name) if pdf_paths else "MISSING_OR_MANUAL_DOWNLOAD",
                "downloaded_pdf_exists": bool(pdf_paths),
                "algorithm_relevance": _bibliography_relevance(filename),
                "landing_url": item.get("landing_url"),
            }
        )
    return rows


def _manual_data_requirements() -> list[dict[str, str]]:
    return [
        {"requirement": "Options", "needed_for": "Real Black-Scholes broker decisions", "details": "Use live option chains, bid/ask, dividends, borrow rates, corporate actions and calibrated implied-vol surfaces."},
        {"requirement": "Intraday/LOB data", "needed_for": "Pairs/Kalman/LOB papers", "details": "The downloaded LOB papers require order-book snapshots or order-flow imbalance data; daily OHLCV is insufficient for broker-grade replication."},
        {"requirement": "Execution and borrow", "needed_for": "Broker pairs trading", "details": "Short availability, locate cost, financing, commissions, spread, slippage and market impact must be broker-calibrated."},
        {"requirement": "Risk-free/factor data", "needed_for": "Portfolio and ML promotion", "details": "A reliable risk-free curve and factor returns are required before alpha/risk-adjusted claims can be promoted."},
        {"requirement": "Production controls", "needed_for": "Any broker integration", "details": "Kill switches, order throttles, exposure limits, pre-trade checks and human approval remain unimplemented by design."},
    ]


def _price_frame_from_terminal_report(report: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for asset, stock in _mapping(report.get("stocks")).items():
        rows = [row for row in stock.get("price_series", []) if isinstance(row, dict)]
        frame = pd.DataFrame(rows)
        if frame.empty or "timestamp" not in frame or "price" not in frame:
            continue
        series = pd.Series(frame["price"].astype(float).to_numpy(), index=pd.to_datetime(frame["timestamp"]), name=str(asset))
        frames.append(series)
    return pd.concat(frames, axis=1).dropna(how="any") if frames else pd.DataFrame()


def _return_frame_from_terminal_report(report: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for asset, stock in _mapping(report.get("stocks")).items():
        rows = [row for row in stock.get("simple_returns", []) if isinstance(row, dict)]
        frame = pd.DataFrame(rows)
        if frame.empty or "timestamp" not in frame or "return" not in frame:
            continue
        series = pd.Series(frame["return"].astype(float).to_numpy(), index=pd.to_datetime(frame["timestamp"]), name=str(asset))
        frames.append(series)
    return pd.concat(frames, axis=1).dropna(how="any") if frames else pd.DataFrame()


def _write_tables(tables: dict[str, Any], tables_dir: Path) -> dict[str, str]:
    paths = {}
    for index, (name, rows) in enumerate(tables.items(), start=1):
        path = tables_dir / f"{index:02d}_{name}.csv"
        pd.DataFrame(rows if isinstance(rows, list) else []).to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def _write_figures(study: dict[str, Any], figures_dir: Path) -> dict[str, str]:
    go = _plotly_go()
    series = _mapping(study.get("series"))
    paths = {}
    figure_builders = {
        "black_scholes_payoff": _figure_black_scholes(go, series.get("black_scholes_payoff", [])),
        "monte_carlo_fan": _figure_monte_carlo(go, series.get("monte_carlo_fan", [])),
        "markowitz_frontier": _figure_markowitz(go, series.get("markowitz_frontier", [])),
        "ou_pair_zscore": _figure_ou(go, series.get("ou_pair_backtest", [])),
        "kalman_beta": _figure_kalman(go, series.get("kalman_dynamic_hedge", [])),
        "garch_volatility": _figure_garch(go, series.get("garch_volatility", [])),
        "ml_predictions": _figure_ml(go, series.get("ml_predictions", [])),
    }
    for index, (name, figure) in enumerate(figure_builders.items(), start=1):
        path = figures_dir / f"{index:02d}_{name}.html"
        figure.write_html(str(path), include_plotlyjs="cdn", full_html=True)
        paths[name] = str(path)
    return paths


def _figure_black_scholes(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    figure.add_trace(go.Scatter(x=[r.get("underlying_price") for r in data], y=[r.get("payoff") for r in data], name="Payoff", mode="lines"))
    figure.update_layout(template="plotly_white", title="Black-Scholes payoff profile", xaxis_title="Underlying price", yaxis_title="Payoff")
    return figure


def _figure_monte_carlo(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    for key in ("p5", "p25", "p50", "p75", "p95"):
        figure.add_trace(go.Scatter(x=[r.get("step") for r in data], y=[r.get(key) for r in data], name=key.upper(), mode="lines"))
    figure.update_layout(template="plotly_white", title="Monte Carlo terminal wealth fan", xaxis_title="Day", yaxis_title="Wealth")
    return figure


def _figure_markowitz(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    figure.add_trace(go.Scatter(x=[r.get("annualized_volatility") for r in data], y=[r.get("annualized_return") for r in data], mode="markers", marker={"color": [r.get("sharpe_ratio") for r in data], "colorscale": "Viridis", "showscale": True}, name="Frontier grid"))
    figure.update_layout(template="plotly_white", title="Markowitz efficient frontier grid", xaxis_title="Annualized volatility", yaxis_title="Annualized return")
    return figure


def _figure_ou(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    figure.add_trace(go.Scatter(x=[r.get("timestamp") for r in data], y=[r.get("zscore") for r in data], name="Spread z-score", mode="lines"))
    for level in (-2, 0, 2):
        figure.add_hline(y=level, line_dash="dash", line_color="#64748b")
    figure.update_layout(template="plotly_white", title="OU pairs-trading z-score", xaxis_title="Date", yaxis_title="Z-score")
    return figure


def _figure_kalman(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    figure.add_trace(go.Scatter(x=[r.get("timestamp") for r in data], y=[r.get("beta") for r in data], name="Kalman beta", mode="lines"))
    figure.update_layout(template="plotly_white", title="Kalman dynamic hedge ratio", xaxis_title="Date", yaxis_title="Beta")
    return figure


def _figure_garch(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    figure.add_trace(go.Scatter(x=[r.get("timestamp") for r in data], y=[r.get("conditional_volatility_annual") for r in data], name="Annualized conditional volatility", mode="lines"))
    figure.update_layout(template="plotly_white", title="GARCH conditional volatility", xaxis_title="Date", yaxis_title="Annual vol")
    return figure


def _figure_ml(go, rows: object):  # noqa: ANN001, ANN201
    figure = go.Figure()
    data = [row for row in rows if isinstance(row, dict)]
    figure.add_trace(go.Scatter(x=[r.get("timestamp") for r in data], y=[r.get("actual_return") for r in data], name="Actual", mode="lines"))
    figure.add_trace(go.Scatter(x=[r.get("timestamp") for r in data], y=[r.get("model_prediction") for r in data], name="Predicted", mode="lines"))
    figure.update_layout(template="plotly_white", title="ML alpha walk-forward prediction", xaxis_title="Date", yaxis_title="Return")
    return figure


def _plotly_go():  # noqa: ANN201
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        raise SevenAlgorithmsStudyError("plotly is required for figure generation.") from exc
    return go


def _representative_mc_fan(report: dict[str, Any], asset: str) -> list[dict[str, Any]]:
    stock = _mapping(_mapping(report.get("stocks")).get(asset))
    return list(_mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal")).get("fan_chart", []))


def _representative_payoff(report: dict[str, Any], asset: str) -> list[dict[str, Any]]:
    stock = _mapping(_mapping(report.get("stocks")).get(asset))
    return list(_mapping(stock.get("options_theoretical_analytics")).get("payoff_profile", []))


def _representative_ml_predictions(report: dict[str, Any], asset: str) -> list[dict[str, Any]]:
    return list(_mapping(_mapping(report.get("ml_forecasting")).get(asset)).get("prediction_rows", []))


def _markowitz_frontier_rows(report: dict[str, Any], *, max_rows: int) -> list[dict[str, Any]]:
    frontier = list(_mapping(report.get("optimization")).get("frontier", []))
    if len(frontier) <= max_rows:
        return frontier
    positions = np.linspace(0, len(frontier) - 1, max_rows, dtype=int)
    return [frontier[int(pos)] for pos in positions]


def _paper_sections() -> list[tuple[str, list[str]]]:
    return [
        ("Data Quality and Scope", ["data_quality_gate", "manual_data_requirements"]),
        ("Implementation Map", ["algorithm_implementation_map"]),
        ("1. Black-Scholes", ["black_scholes_replication"]),
        ("2. Monte Carlo", ["monte_carlo_replication"]),
        ("3. Markowitz", ["markowitz_replication"]),
        ("4. Pairs Trading OU", ["ou_pair_screen", "ou_pair_backtest"]),
        ("5. Kalman Filter", ["kalman_dynamic_hedge"]),
        ("6. GARCH", ["garch_volatility"]),
        ("7. Machine Learning Alpha", ["ml_alpha_validation"]),
        ("Broker Readiness", ["broker_readiness_decisions"]),
        ("Bibliography Trace", ["bibliography_trace"]),
    ]


def _section_text(title: str, study: dict[str, Any]) -> str:
    if title == "Data Quality and Scope":
        source = _mapping(study.get("source_terminal_report"))
        return f"The study uses {source.get('data_mode')} with {source.get('aligned_observations')} aligned observations. Provider warnings are treated as blockers for broker promotion."
    if title == "Broker Readiness":
        return "Every algorithm is evaluated for broker implications, but real broker actions remain blocked because execution controls, live market data calibration and human approval are intentionally absent."
    if title == "Bibliography Trace":
        return "The downloaded paper pack is mapped to algorithm relevance. Missing SSRN files are marked as manual-download requirements."
    return "This section replicates the algorithm with historical provider data, explains the quantitative evidence, and records why the output is research-only rather than an executable broker instruction."


def _figure_key_for_title(title: str) -> str:
    mapping = {
        "1. Black-Scholes": "black_scholes_payoff",
        "2. Monte Carlo": "monte_carlo_fan",
        "3. Markowitz": "markowitz_frontier",
        "4. Pairs Trading OU": "ou_pair_zscore",
        "5. Kalman Filter": "kalman_beta",
        "6. GARCH": "garch_volatility",
        "7. Machine Learning Alpha": "ml_predictions",
    }
    return mapping.get(title, "")


def _manifest(
    study: dict[str, Any],
    outputs: dict[str, str],
    tables: dict[str, str],
    figures: dict[str, str],
) -> dict[str, Any]:
    files = {**outputs, **{f"table:{k}": v for k, v in tables.items()}, **{f"figure:{k}": v for k, v in figures.items()}}
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "report_type": study.get("report_type"),
        "research_only": True,
        "files": {key: {"path": path, "sha256": _sha256(Path(path))} for key, path in files.items() if Path(path).exists()},
    }


def _clear_generated_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _unique_strings(values: list[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _without_series(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"series", "state_rows", "volatility_series"}}


def _algo_row(algorithm: str, method: str, status: str, broker_gap: str) -> dict[str, str]:
    return {"algorithm": algorithm, "method": method, "replication_status": status, "broker_gap": broker_gap}


def _bibliography_relevance(filename: str) -> str:
    prefix = filename[:2]
    if prefix in {"01", "02", "03", "04"}:
        return "Black-Scholes / Monte Carlo / hedging extensions"
    if prefix in {"05", "06", "10"}:
        return "GARCH / ML alpha / forecasting"
    if prefix in {"07", "08", "09"}:
        return "Markowitz / portfolio optimization extensions"
    if prefix in {"11", "12", "13", "14"}:
        return "Pairs trading / Kalman / microstructure extensions"
    if prefix in {"15", "16", "17"}:
        return "Broker surveillance / fraud-risk extensions"
    if prefix in {"18", "19", "20", "21", "22"}:
        return "Monte Carlo / synthetic market simulation extensions"
    return "General quant finance"


def _markdown_table(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "_No rows available._"
    keys = list(dict.fromkeys(key for row in rows if isinstance(row, dict) for key in row))
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append("| " + " | ".join(_cell(row.get(key)) for key in keys) + " |")
    return "\n".join(lines)


def _html_table(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "<p class='small'>No rows available.</p>"
    keys = list(dict.fromkeys(key for row in rows if isinstance(row, dict) for key in row))
    head = "".join(f"<th>{html.escape(str(key))}</th>" for key in keys)
    body = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        body.append("<tr>" + "".join(f"<td>{html.escape(_cell(row.get(key)))}</td>" for key in keys) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)[:300]
    return str(value if value is not None else "")


def _embedded_html(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return f"<p class='small'>Figure file unavailable: {html.escape(str(path))}</p>"
    return f"<div class='figure-box'>{text}</div>"


def _path_list(paths: dict[str, str]) -> str:
    return "<ul>" + "".join(f"<li><code>{html.escape(k)}</code>: <code>{html.escape(v)}</code></li>" for k, v in sorted(paths.items())) + "</ul>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
