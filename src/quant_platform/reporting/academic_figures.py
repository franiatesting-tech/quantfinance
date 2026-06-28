"""Plotly figure generation for academic stock reports with value annotations."""

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

_SECTION_LETTERS = {
    "price_history": "A", "volume": "B", "drawdown": "C",
    "simple_returns": "D", "log_returns": "E", "cumulative_returns": "F",
    "returns_distribution": "G", "rolling_volatility": "H", "rolling_sharpe": "I",
    "rolling_beta": "J", "var_comparison": "K",
    "monte_carlo_paths": "L", "monte_carlo_percentiles": "M",
    "monte_carlo_terminal_distribution": "N",
    "backtest_equity_curves": "O", "options_payoff": "P", "greeks": "Q",
}


def build_academic_stock_figures(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build all Plotly figures for a stock academic report."""

    stock = _stock(report_model)
    metrics = stock.get("metrics", {})
    return {
        "price_history": _price_figure(stock, metrics),
        "volume": _bar_figure(stock, "volume_series", "volume",
                              "Volume Profile", "Volume (shares)",
                              "Daily trading volume in shares"),
        "simple_returns": _line_figure(stock, "simple_returns", "return",
                                       "Simple Daily Returns", "Daily return",
                                       "Fluctuation del precio diario en %"),
        "log_returns": _line_figure(stock, "log_returns", "log_return",
                                    "Log Returns", "Log return",
                                    "Retorno logaritmico ln(P_t/P_{t-1})"),
        "cumulative_returns": _cumret_figure(stock, metrics),
        "drawdown": _drawdown_figure(stock, metrics),
        "rolling_volatility": _rolling_vol_figure(stock, metrics),
        "rolling_sharpe": _rolling_sharpe_figure(stock, metrics),
        "rolling_beta": _line_figure(stock, "rolling_beta", "rolling_beta",
                                     "Rolling Beta (vs Benchmark)", "Beta",
                                     "Beta movil de 63 sesiones contra el benchmark"),
        "returns_distribution": _histogram_figure(
            stock, "simple_returns", "return",
            "Returns Distribution", "Daily return"
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


# ---------------------------------------------------------------------------
# Individual figure builders (with value annotations)
# ---------------------------------------------------------------------------

def _price_figure(stock: dict[str, Any], metrics: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("price_series"))
    figure = go.Figure()
    if rows:
        prices = [row.get("price") for row in rows]
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=prices, mode="lines", name="Close",
            hovertemplate="Date=%{x}<br>Price=%{y:.2f}<extra></extra>",
            line=dict(color="#3dd6c6", width=2),
        ))
    _add_annotation(figure, f"Retorno acumulado: {_pct(metrics.get('final_cumulative_return'))} | "
                    f"Volatilidad anual: {_pct(metrics.get('annualized_volatility'))}")
    _apply_layout(figure, stock, "Precio historico (Price History)", "Date", "Price (USD)",
                  "Evolucion del precio de cierre ajustado. Muestra tendencia de largo plazo, "
                  "volatilidad y posibles puntos de entrada/salida.")
    return figure


def _cumret_figure(stock: dict[str, Any], metrics: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("cumulative_returns"))
    figure = go.Figure()
    if rows:
        cumret = [row.get("cumulative_return") for row in rows]
        start_val = 100.0
        equity = [start_val * (1 + v) for v in cumret]
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=equity, mode="lines", name="1 EUR invertido",
            hovertemplate="Date=%{x}<br>Valor=%{y:.2f} EUR<extra></extra>",
            line=dict(color="#f2d27a", width=2),
        ))
        figure.add_hline(y=start_val, line_dash="dash", line_color="#6b7b8b",
                         annotation_text="Capital inicial (100 EUR)")
    ret = metrics.get("annualized_return")
    cagr = metrics.get("cagr")
    _add_annotation(
        figure,
        f"Rentabilidad anualizada: {_pct(ret)} | CAGR: {_pct(cagr)} | "
        "Interpretacion: capital inicial de 100 EUR revalorizado segun retorno compuesto",
    )
    _apply_layout(figure, stock, "Crecimiento de 100 EUR (Cumulative Returns)", "Date",
                  "Valor de la inversion (EUR)",
                  "Muestra la evolucion de 100 EUR invertidos al inicio del periodo. "
                  "Facilita comparacion visual entre activos y con buy-and-hold.")
    return figure


def _drawdown_figure(stock: dict[str, Any], metrics: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("drawdown_series"))
    figure = go.Figure()
    if rows:
        dd = [row.get("drawdown") for row in rows]
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=dd, mode="lines", name="Drawdown",
            fill="tozeroy", line=dict(color="#d66a4a", width=2),
            hovertemplate="Date=%{x}<br>Drawdown=%{y:.2%}<extra></extra>",
        ))
    max_dd = metrics.get("max_drawdown")
    _add_annotation(figure, f"Maximo drawdown: {_pct(max_dd)} | "
                    "Mide la caida desde el maximo historico hasta el minimo posterior. "
                    "Drawdowns severos (>30%) indican riesgo de perdida significativa.")
    _apply_layout(figure, stock, "Drawdown (caida desde maximo)", "Date",
                  "Drawdown (%)",
                  "Drawdown = (Precio actual / Maximo historico) - 1. "
                  "Muestra el riesgo de caida real experimentado.")
    return figure


def _rolling_vol_figure(stock: dict[str, Any], metrics: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("rolling_volatility"))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=[row.get("rolling_volatility") for row in rows],
            mode="lines", name="Vol movil",
            line=dict(color="#3dd6c6", width=2),
            hovertemplate="Date=%{x}<br>Vol=%{y:.2%}<extra></extra>",
        ))
    ann_vol = metrics.get("annualized_volatility")
    _add_annotation(figure, f"Volatilidad anualizada media: {_pct(ann_vol)} | "
                    "Ventana de calculo: 63 sesiones. "
                    "Una volatilidad >30% anual se considera alta; <15% baja.")
    _apply_layout(figure, stock, "Volatilidad movil (Rolling Volatility)", "Date",
                  "Volatilidad anualizada (%)",
                  "Desviacion estandar de retornos diarios en ventana movil de 63 sesiones, "
                  "anualizada x sqrt(252). Identifica periodos de turbulencia y calma.")
    return figure


def _rolling_sharpe_figure(stock: dict[str, Any], metrics: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("rolling_sharpe"))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=[row.get("rolling_sharpe") for row in rows],
            mode="lines", name="Sharpe movil",
            line=dict(color="#f2d27a", width=2),
            hovertemplate="Date=%{x}<br>Sharpe=%{y:.2f}<extra></extra>",
        ))
        figure.add_hline(y=1.0, line_dash="dash", line_color="#6b7b8b",
                         annotation_text="Sharpe=1.0 (referencia)")
    sharpe = metrics.get("sharpe_ratio")
    _add_annotation(figure, f"Sharpe ratio medio: {_num(sharpe)} | "
                    "Sharpe > 1 indica retorno historico elevado por unidad de riesgo. "
                    "Valores negativos indican que el activo no compenso su riesgo.")
    _apply_layout(figure, stock, "Sharpe Ratio movil (Rolling Sharpe)", "Date",
                  "Sharpe ratio",
                  "Sharpe = (Rendimiento - Tasa libre de riesgo) / Volatilidad. "
                  "Ventana movil de 63 sesiones. Mide eficiencia riesgo-retorno.")
    return figure


def _line_figure(
    stock: dict[str, Any], series_key: str, value_key: str,
    title: str, yaxis_title: str, subtitle: str,
):
    go = _plotly_go()
    rows = _rows(stock.get(series_key))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=[row.get(value_key) for row in rows],
            mode="lines",
            name=value_key,
            hovertemplate="Date=%{x}<br>Value=%{y:.6f}<extra></extra>",
            line=dict(color="#3dd6c6", width=2),
        ))
    _add_annotation(figure, subtitle)
    _apply_layout(figure, stock, title, "Date", yaxis_title, subtitle)
    return figure


def _bar_figure(
    stock: dict[str, Any], series_key: str, value_key: str,
    title: str, yaxis_title: str, subtitle: str,
):
    go = _plotly_go()
    rows = _rows(stock.get(series_key))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Bar(
            x=[row.get("timestamp") for row in rows],
            y=[row.get(value_key) for row in rows],
            name=value_key,
            marker_color="#d6b35a",
        ))
    _add_annotation(figure, subtitle)
    _apply_layout(figure, stock, title, "Date", yaxis_title, subtitle)
    return figure


def _histogram_figure(
    stock: dict[str, Any], series_key: str, value_key: str, title: str, xaxis: str
):
    go = _plotly_go()
    values = [row.get(value_key) for row in _rows(stock.get(series_key))]
    figure = go.Figure()
    if values:
        filtered = [v for v in values if v is not None]
        mean_v = sum(filtered) / len(filtered) if filtered else 0
        figure.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#3dd6c6",
                                       name="retornos diarios"))
        figure.add_vline(x=mean_v, line_dash="dash", line_color="#d66a4a",
                         annotation_text=f"Media={mean_v:.4%}")
    metrics = stock.get("metrics", {})
    _add_annotation(figure,
                    f"Asimetria: {_num(metrics.get('skewness'))} | "
                    f"Curtosis: {_num(metrics.get('kurtosis'))} | "
                    f"Hit rate: {_pct(metrics.get('hit_rate'))} | "
                    "Distribucion histogramada de retornos diarios. "
                    "Colas gruesas (curtosis > 3) indican mayor probabilidad de extremos.")
    _apply_layout(figure, stock, title, xaxis, "Count",
                  "Histograma de frecuencias de retornos diarios. La linea roja marca la media.")
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
        metric_rows = [r for r in rows if r["metric"] == metric]
        figure.add_trace(go.Bar(
            name=metric, marker_color=color,
            x=[r["model"] for r in metric_rows],
            y=[r["value"] for r in metric_rows],
        ))
    figure.update_layout(barmode="group")
    historical = _mapping(var_payload.get("historical"))
    _add_annotation(figure,
                    f"VaR historico 95%: {_pct(historical.get('var'))} | "
                    f"ES historico 95%: {_pct(historical.get('expected_shortfall'))} | "
                    "VaR: perdida maxima esperada con 95% confianza en 1 dia. "
                    "ES: perdida media en el peor 5% de los dias.")
    _apply_layout(figure, stock, "Comparison: Value at Risk / Expected Shortfall",
                  "Modelo", "Perdida positiva (%)",
                  "Positive-loss convention, alpha=95%. "
                  "ES captura el riesgo de cola mejor que VaR.")
    return figure


def _monte_carlo_paths_figure(stock: dict[str, Any]):
    go = _plotly_go()
    paths = _rows(_normal_mc(stock).get("paths_sample"))
    figure = go.Figure()
    for path_id in sorted({row.get("path_id") for row in paths})[:25]:
        path_rows = [row for row in paths if row.get("path_id") == path_id]
        figure.add_trace(go.Scatter(
            x=[row.get("step") for row in path_rows],
            y=[row.get("value") for row in path_rows],
            mode="lines", name=f"path {path_id}", opacity=0.35, showlegend=False,
            line=dict(color="#3dd6c6", width=1),
        ))
    mc = _normal_mc(stock)
    loss_p = mc.get("probability_of_loss")
    _add_annotation(figure,
                    f"Probabilidad de perdida: {_pct(loss_p)} | "
                    "Cada linea es una trayectoria simulada bajo el modelo GBM. "
                    "Muestra la dispersion de resultados posibles.")
    _apply_layout(figure, stock, "Simulacion Monte Carlo: trayectorias posibles",
                  "Dia de simulacion (252 sesiones = 1 ano)", "Valor de la inversion (EUR)",
                  "Parametric normal paths. 25 trayectorias mostradas de 1000 simuladas. "
                  "No constituye prediccion.")
    return figure


def _monte_carlo_percentiles_figure(stock: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(_normal_mc(stock).get("fan_chart"))
    figure = go.Figure()
    colors = {
        "p5": "#d66a4a",
        "p25": "#d6b35a",
        "p50": "#f2d27a",
        "p75": "#3dd6c6",
        "p95": "#3dd6c6",
    }
    for percentile in ("p5", "p25", "p50", "p75", "p95"):
        figure.add_trace(go.Scatter(
            x=[row.get("step") for row in rows],
            y=[row.get(percentile) for row in rows],
            mode="lines", name=percentile.upper(),
            line=dict(
                color=colors.get(percentile, "#e8ecef"),
                width=2 if percentile == "p50" else 1,
            ),
        ))
    mc = _normal_mc(stock)
    _add_annotation(figure,
                    f"P5: {_pct(mc.get('terminal_p05'))} | "
                    f"P50: {_pct(mc.get('terminal_median'))} | "
                    f"P95: {_pct(mc.get('terminal_p95'))} | "
                    "El abanico muestra el rango de resultados posibles. "
                    "P50 es la mediana; P5-P95 es el intervalo de confianza del 90%.")
    _apply_layout(figure, stock, "Monte Carlo: percentiles de rentabilidad",
                  "Dia de simulacion", "Valor de la inversion (EUR)",
                  "Fan chart con percentiles 5, 25, 50, 75, 95. "
                  "La mediana (P50) es el resultado central esperado.")
    return figure


def _monte_carlo_terminal_figure(stock: dict[str, Any]):
    go = _plotly_go()
    values = _normal_mc(stock).get("terminal_distribution", [])
    figure = go.Figure()
    if isinstance(values, list):
        filtered = [v for v in values if v is not None]
        mean_v = sum(filtered) / len(filtered) if filtered else 0
        figure.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#d6b35a",
                                       name="retorno terminal"))
        figure.add_vline(x=mean_v, line_dash="dash", line_color="#d66a4a",
                         annotation_text=f"Media={mean_v:.2%}")
    mc = _normal_mc(stock)
    _add_annotation(figure,
                    f"Mediana: {_pct(mc.get('terminal_median'))} | "
                    f"Prob. perdida: {_pct(mc.get('probability_of_loss'))} | "
                    "Distribucion de rentabilidades al final del horizonte de simulacion. "
                    "Ayuda a visualizar la probabilidad de resultados negativos.")
    _apply_layout(figure, stock, "Monte Carlo: distribucion terminal",
                  "Rentabilidad terminal (%)", "Frecuencia",
                  "Histograma de 1000 simulaciones. La linea roja marca la media.")
    return figure


def _backtest_equity_figure(stock: dict[str, Any]):
    go = _plotly_go()
    backtests = _mapping(stock.get("backtesting_results"))
    figure = go.Figure()
    colors = ["#3dd6c6", "#f2d27a", "#d66a4a", "#6b7b8b", "#d6b35a"]
    for idx, (name, payload) in enumerate(backtests.items()):
        rows = _rows(_mapping(payload).get("equity_curve"))
        if rows:
            color = colors[idx % len(colors)]
            figure.add_trace(go.Scatter(
                name=str(name), line=dict(color=color, width=2),
                x=[row.get("timestamp") for row in rows],
                y=[row.get("equity") for row in rows],
                mode="lines",
            ))
    bh = _mapping(backtests.get("buy_and_hold"))
    _add_annotation(
        figure,
        f"Buy-and-Hold equity final: {_num(bh.get('metrics', {}).get('final_equity'))} | "
        "Backtest simula estrategias con capital ficticio. "
        "Compara rendimiento de distintas reglas de trading.",
    )
    _apply_layout(figure, stock, "Backtesting: curvas de capital",
                  "Date", "Capital ficticio (EUR)",
                  "Simulacion historica sin ejecucion real. "
                  "Buy-and-hold es la referencia pasiva.")
    return figure


def _options_payoff_figure(stock: dict[str, Any]):
    go = _plotly_go()
    options = _mapping(stock.get("options_theoretical_analytics"))
    rows = _rows(options.get("payoff_profile"))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("underlying_price") for row in rows],
            y=[row.get("payoff") for row in rows],
            mode="lines", name="Call payoff",
            line=dict(color="#f2d27a", width=2),
        ))
    _add_annotation(figure,
                    f"Precio call BSM: {_num(options.get('black_scholes_call'))} | "
                    f"Precio put BSM: {_num(options.get('black_scholes_put'))} | "
                    "Payoff teorico de opcion call ATM bajo Black-Scholes. "
                    "No representa precios de mercado reales.")
    _apply_layout(figure, stock, "Perfil de pago de opcion (Options Payoff)",
                  "Precio del subyacente (USD)", "Payoff (EUR)",
                  "PARAMETRIC_EDUCATIONAL_MODEL. Sin option chain real.")
    return figure


def _greeks_figure(stock: dict[str, Any]):
    go = _plotly_go()
    greeks = _mapping(_mapping(stock.get("options_theoretical_analytics")).get("call_greeks"))
    keys = ["delta", "gamma", "vega", "theta_annual", "rho"]
    values = [greeks.get(key) for key in keys]
    colors_list = ["#3dd6c6", "#f2d27a", "#d66a4a", "#d6b35a", "#6b7b8b"]
    figure = go.Figure(data=[go.Bar(x=keys, y=values, marker_color=colors_list)])
    _add_annotation(figure,
                    f"Delta={_num(greeks.get('delta'))} | "
                    f"Gamma={_num(greeks.get('gamma'))} | "
                    "Griegas BSM: Delta mide sensibilidad al precio; "
                    "Gamma mide curvatura; Vega sensibilidad a volatilidad.")
    _apply_layout(figure, stock, "Griegas de Black-Scholes (Greeks)",
                  "Griega", "Valor",
                  "Griegas son sensibilidades del modelo BSM. "
                  "Sin datos de mercado reales.")
    return figure


# ---------------------------------------------------------------------------
# Layout & helpers
# ---------------------------------------------------------------------------

def _add_annotation(figure, text: str) -> None:
    """Add a value annotation below the chart."""
    figure.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0, y=-0.30,
        showarrow=False,
        align="left",
        font={"size": 12, "color": "#c8d0d8"},
        bordercolor="#2c3a46",
        borderwidth=1,
        borderpad=6,
        bgcolor="#0d1520",
    )


def _apply_layout(
    figure, stock: dict[str, Any], title: str, xaxis: str, yaxis: str, note: str
) -> None:
    data_used = _mapping(stock.get("data_used"))
    fig_title = f"{data_used.get('ticker', stock.get('asset_id', 'Asset'))} - {title}"
    figure.update_layout(
        title=fig_title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#e8ecef"},
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        margin={"l": 60, "r": 40, "t": 80, "b": 120},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    figure.add_annotation(
        text=(
            f"Source={data_used.get('provider', 'unknown')} | "
            f"Freq={data_used.get('frequency', '1d')} | {_model_note(stock)}"
        ),
        xref="paper", yref="paper",
        x=0, y=-0.10,
        showarrow=False,
        align="left",
        font={"size": 10, "color": "#6b7b8b"},
    )


def _plotly_go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
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


def _pct(value: object) -> str:
    try:
        v = float(value)
        return f"{v:.2%}" if v == v else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def _num(value: object) -> str:
    try:
        v = float(value)
        return f"{v:,.4f}" if v == v else "N/A"
    except (TypeError, ValueError):
        return "N/A"
