"""Plotly figure generation for academic stock reports."""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

FIGURE_FILENAMES = {
    "price_history": "price_history.html",
    "volume": "volume.html",
    "simple_returns": "simple_returns.html",
    "log_returns": "log_returns.html",
    "cumulative_returns": "cumulative_returns.html",
    "drawdown": "drawdown.html",
    "rolling_volatility": "rolling_volatility.html",
    "rolling_sharpe": "rolling_sharpe.html",
    "rolling_beta": "rolling_beta.html",
    "returns_distribution": "returns_distribution.html",
    "var_comparison": "var_comparison.html",
    "monte_carlo_paths": "monte_carlo_paths.html",
    "monte_carlo_percentiles": "monte_carlo_percentiles.html",
    "monte_carlo_terminal_distribution": "monte_carlo_terminal_distribution.html",
    "backtest_equity_curves": "backtest_equity_curves.html",
    "options_payoff": "options_payoff.html",
    "greeks": "greeks.html",
}


def build_academic_stock_figures(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build all Plotly figures for a stock academic report."""

    stock = _stock(report_model)
    return {
        "price_history": _line_figure(stock, "price_series", "price", "Price History", "Price"),
        "volume": _bar_figure(stock, "volume_series", "volume", "Volume", "Volume"),
        "simple_returns": _line_figure(
            stock, "simple_returns", "return", "Simple Daily Returns", "Return"
        ),
        "log_returns": _line_figure(
            stock, "log_returns", "log_return", "Log Returns", "Log return"
        ),
        "cumulative_returns": _line_figure(
            stock,
            "cumulative_returns",
            "cumulative_return",
            "Cumulative Returns",
            "Cumulative return",
        ),
        "drawdown": _line_figure(stock, "drawdown_series", "drawdown", "Drawdown", "Drawdown"),
        "rolling_volatility": _line_figure(
            stock,
            "rolling_volatility",
            "rolling_volatility",
            "Rolling Volatility",
            "Annualized volatility",
        ),
        "rolling_sharpe": _line_figure(
            stock, "rolling_sharpe", "rolling_sharpe", "Rolling Sharpe", "Sharpe"
        ),
        "rolling_beta": _line_figure(stock, "rolling_beta", "rolling_beta", "Rolling Beta", "Beta"),
        "returns_distribution": _histogram_figure(
            stock, "simple_returns", "return", "Returns Distribution", "Return"
        ),
        "var_comparison": _var_comparison_figure(stock),
        "monte_carlo_paths": _monte_carlo_paths_figure(stock),
        "monte_carlo_percentiles": _monte_carlo_percentiles_figure(stock),
        "monte_carlo_terminal_distribution": _monte_carlo_terminal_figure(stock),
        "backtest_equity_curves": _backtest_equity_figure(stock),
        "options_payoff": _options_payoff_figure(stock),
        "greeks": _greeks_figure(stock),
    }


def write_academic_stock_figures(
    report_model: dict[str, Any],
    figures_dir: str | Path,
) -> dict[str, str]:
    """Write academic stock figures as standalone Plotly HTML files."""

    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = build_academic_stock_figures(report_model)
    paths = {}
    for name, figure in figures.items():
        path = output_dir / FIGURE_FILENAMES[name]
        figure.write_html(path, include_plotlyjs="cdn", full_html=True)
        paths[name] = str(path)
    return paths


def _line_figure(
    stock: dict[str, Any],
    series_key: str,
    value_key: str,
    title: str,
    yaxis_title: str,
):
    go = _plotly_go()
    rows = _rows(stock.get(series_key))
    figure = go.Figure()
    if rows:
        figure.add_trace(
            go.Scatter(
                x=[row.get("timestamp") for row in rows],
                y=[row.get(value_key) for row in rows],
                mode="lines",
                name=value_key,
                hovertemplate="Date=%{x}<br>Value=%{y:.6f}<extra></extra>",
            )
        )
    _apply_layout(figure, stock, title, "Date", yaxis_title, _model_note(stock))
    return figure


def _bar_figure(
    stock: dict[str, Any],
    series_key: str,
    value_key: str,
    title: str,
    yaxis_title: str,
):
    go = _plotly_go()
    rows = _rows(stock.get(series_key))
    figure = go.Figure()
    if rows:
        figure.add_trace(
            go.Bar(
                x=[row.get("timestamp") for row in rows],
                y=[row.get(value_key) for row in rows],
                name=value_key,
                marker_color="#d6b35a",
            )
        )
    _apply_layout(figure, stock, title, "Date", yaxis_title, _model_note(stock))
    return figure


def _histogram_figure(
    stock: dict[str, Any], series_key: str, value_key: str, title: str, xaxis: str
):
    go = _plotly_go()
    values = [row.get(value_key) for row in _rows(stock.get(series_key))]
    figure = go.Figure()
    if values:
        figure.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#3dd6c6"))
    _apply_layout(
        figure, stock, title, xaxis, "Count", "Historical distribution; tails are sample-dependent."
    )
    return figure


def _var_comparison_figure(stock: dict[str, Any]):
    go = _plotly_go()
    var_payload = _mapping(stock.get("var"))
    rows = []
    for model in ("historical", "parametric_normal", "monte_carlo"):
        payload = _mapping(var_payload.get(model))
        rows.append({"model": model, "metric": "VaR", "value": payload.get("var")})
        rows.append({"model": model, "metric": "ES", "value": payload.get("expected_shortfall")})
    figure = go.Figure()
    for metric, color in (("VaR", "#d6b35a"), ("ES", "#d66a4a")):
        metric_rows = [row for row in rows if row["metric"] == metric]
        figure.add_trace(
            go.Bar(
                name=metric,
                x=[row["model"] for row in metric_rows],
                y=[row["value"] for row in metric_rows],
                marker_color=color,
            )
        )
    figure.update_layout(barmode="group")
    _apply_layout(
        figure,
        stock,
        "VaR / ES Comparison",
        "Model",
        "Positive loss",
        "Positive-loss convention, alpha=95%.",
    )
    return figure


def _monte_carlo_paths_figure(stock: dict[str, Any]):
    go = _plotly_go()
    paths = _rows(_normal_mc(stock).get("paths_sample"))
    figure = go.Figure()
    for path_id in sorted({row.get("path_id") for row in paths})[:25]:
        path_rows = [row for row in paths if row.get("path_id") == path_id]
        figure.add_trace(
            go.Scatter(
                x=[row.get("step") for row in path_rows],
                y=[row.get("value") for row in path_rows],
                mode="lines",
                name=f"path {path_id}",
                opacity=0.35,
                showlegend=False,
            )
        )
    _apply_layout(
        figure,
        stock,
        "Monte Carlo Sample Paths",
        "Step",
        "Simulated wealth",
        "Parametric normal paths; not a prediction.",
    )
    return figure


def _monte_carlo_percentiles_figure(stock: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(_normal_mc(stock).get("fan_chart"))
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
    _apply_layout(
        figure,
        stock,
        "Monte Carlo Percentiles",
        "Step",
        "Simulated wealth",
        "Fan chart from simulated paths.",
    )
    return figure


def _monte_carlo_terminal_figure(stock: dict[str, Any]):
    go = _plotly_go()
    values = _normal_mc(stock).get("terminal_distribution", [])
    figure = go.Figure()
    if isinstance(values, list):
        figure.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#d6b35a"))
    _apply_layout(
        figure,
        stock,
        "Monte Carlo Terminal Distribution",
        "Terminal return",
        "Count",
        "Distribution depends on model assumptions and seed.",
    )
    return figure


def _backtest_equity_figure(stock: dict[str, Any]):
    go = _plotly_go()
    backtests = _mapping(stock.get("backtesting_results"))
    figure = go.Figure()
    for name, payload in backtests.items():
        rows = _rows(_mapping(payload).get("equity_curve"))
        if rows:
            figure.add_trace(
                go.Scatter(
                    name=str(name),
                    x=[row.get("timestamp") for row in rows],
                    y=[row.get("equity") for row in rows],
                    mode="lines",
                )
            )
    _apply_layout(
        figure,
        stock,
        "Backtest Equity Curves",
        "Date",
        "Fictitious capital",
        "Backtests are historical simulations with no real execution.",
    )
    return figure


def _options_payoff_figure(stock: dict[str, Any]):
    go = _plotly_go()
    options = _mapping(stock.get("options_theoretical_analytics"))
    rows = _rows(options.get("payoff_profile"))
    figure = go.Figure()
    if rows:
        figure.add_trace(
            go.Scatter(
                x=[row.get("underlying_price") for row in rows],
                y=[row.get("payoff") for row in rows],
                mode="lines",
                name="call payoff",
            )
        )
    _apply_layout(
        figure,
        stock,
        "Options Payoff",
        "Underlying price",
        "Payoff",
        "PARAMETRIC_EDUCATIONAL_MODEL; no option chain used.",
    )
    return figure


def _greeks_figure(stock: dict[str, Any]):
    go = _plotly_go()
    greeks = _mapping(_mapping(stock.get("options_theoretical_analytics")).get("call_greeks"))
    keys = ["delta", "gamma", "vega", "theta_annual", "rho"]
    figure = go.Figure(
        data=[go.Bar(x=keys, y=[greeks.get(key) for key in keys], marker_color="#3dd6c6")]
    )
    _apply_layout(
        figure,
        stock,
        "Black-Scholes Greeks",
        "Greek",
        "Value",
        "Greeks are model sensitivities under BSM assumptions.",
    )
    return figure


def _apply_layout(
    figure, stock: dict[str, Any], title: str, xaxis: str, yaxis: str, note: str
) -> None:  # noqa: ANN001
    data_used = _mapping(stock.get("data_used"))
    figure.update_layout(
        title=f"{stock.get('asset_id', 'Asset')} - {title}",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#e8ecef"},
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        margin={"l": 50, "r": 30, "t": 70, "b": 85},
    )
    figure.add_annotation(
        text=(
            f"Source={data_used.get('provider', 'unknown')} | "
            f"Frequency={data_used.get('frequency', '1d')} | {note}"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.22,
        showarrow=False,
        align="left",
        font={"size": 11, "color": "#c8d0d8"},
    )


def _plotly_go():  # noqa: ANN202
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional UI/report extra.
        raise RuntimeError("Install the ui extra to generate academic figures.") from exc
    return go


def _stock(report_model: dict[str, Any]) -> dict[str, Any]:
    stock = report_model.get("stock", {})
    return stock if isinstance(stock, dict) else {}


def _rows(value: object) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normal_mc(stock: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal"))


def _model_note(stock: dict[str, Any]) -> str:
    data_used = _mapping(stock.get("data_used"))
    mode = data_used.get("data_mode", "unknown")
    if mode in {"synthetic_fallback", "offline_synthetic"}:
        return "DEMO_SYNTHETIC_NOT_REAL_DATA."
    return "Historical provider data; research-only."
