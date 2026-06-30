"""Plotly figure generation for academic stock reports with value annotations.

Each figure includes:
  - Mathematical description of what is plotted
  - Variables involved and their definitions
  - How the data was obtained (formula/method)
  - Didactic interpretation of results and implications

References:
  Gu, Kelly & Xiu (2020), Rev. Financial Studies, 33(5), 2223-2273.
  Pagliaro (2026), Electronics, 15(6), 1334.
  Bollerslev (1986), J. Econometrics, 31(3), 307-327.
"""

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
    "rolling_correlation": "rolling_correlation.html",
    "active_return": "active_return.html",
    "returns_distribution": "returns_distribution.html",
    "var_comparison": "var_comparison.html",
    "var_exceptions": "var_exceptions.html",
    "monte_carlo_paths": "monte_carlo_paths.html",
    "monte_carlo_percentiles": "monte_carlo_percentiles.html",
    "monte_carlo_terminal_distribution": "monte_carlo_terminal_distribution.html",
    "backtest_equity_curves": "backtest_equity_curves.html",
    "execution_costs": "execution_costs.html",
    "options_payoff": "options_payoff.html",
    "greeks": "greeks.html",
    "ml_prediction_vs_actual": "ml_prediction_vs_actual.html",
    "ml_residuals": "ml_residuals.html",
    "ml_model_comparison": "ml_model_comparison.html",
    "ml_feature_importance": "ml_feature_importance.html",
}

_SECTION_LETTERS = {
    "price_history": "A", "volume": "B", "drawdown": "C",
    "simple_returns": "D", "log_returns": "E", "cumulative_returns": "F",
    "returns_distribution": "G", "rolling_volatility": "H", "rolling_sharpe": "I",
    "rolling_beta": "J", "rolling_correlation": "K", "active_return": "L",
    "var_comparison": "M", "var_exceptions": "N",
    "monte_carlo_paths": "O", "monte_carlo_percentiles": "P",
    "monte_carlo_terminal_distribution": "Q",
    "backtest_equity_curves": "R", "execution_costs": "S",
    "options_payoff": "T", "greeks": "U",
    "ml_prediction_vs_actual": "V", "ml_residuals": "W",
    "ml_model_comparison": "X", "ml_feature_importance": "Y",
}


def build_academic_stock_figures(report_model: dict[str, Any]) -> dict[str, Any]:
    """Build all Plotly figures for a stock academic report."""

    stock = _stock(report_model)
    metrics = stock.get("metrics", {})
    ml = _mapping(stock.get("ml_forecasting"))
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
        "rolling_correlation": _line_figure(
            stock,
            "rolling_correlation",
            "rolling_correlation",
            "Rolling Correlation (vs Benchmark)",
            "Correlation",
            "Correlacion movil de 63 sesiones contra el benchmark",
        ),
        "active_return": _active_return_figure(stock),
        "returns_distribution": _histogram_figure(
            stock, "simple_returns", "return",
            "Returns Distribution", "Daily return"
        ),
        "var_comparison": _var_comparison_figure(stock),
        "var_exceptions": _var_exceptions_figure(stock),
        "monte_carlo_paths": _monte_carlo_paths_figure(stock),
        "monte_carlo_percentiles": _monte_carlo_percentiles_figure(stock),
        "monte_carlo_terminal_distribution": _monte_carlo_terminal_figure(stock),
        "backtest_equity_curves": _backtest_equity_figure(stock),
        "execution_costs": _execution_costs_figure(stock),
        "options_payoff": _options_payoff_figure(stock),
        "greeks": _greeks_figure(stock),
        "ml_prediction_vs_actual": _ml_prediction_figure(ml),
        "ml_residuals": _ml_residuals_figure(ml),
        "ml_model_comparison": _ml_comparison_figure(ml),
        "ml_feature_importance": _ml_feature_importance_figure(ml),
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
# Individual figure builders (with mathematical descriptions)
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
    _add_annotation(figure,
        "GRAFICA A: Precio de cierre ajustado (OHLCV diario). "
        "Muestra la evolucion temporal del precio P_t. "
        "Variables: P_t = precio de cierre ajustado en sesion t, t = fecha. "
        "Obtencion: proveedor yfinance OHLCV campo 'Close'. "
        "Interpretacion: permite identificar tendencia direccional, "
        "soportes/resistencias visuales, y puntos de inflexion historicos.")
    _apply_layout(
        figure,
        stock,
        "Precio historico (Price History)",
        "Date",
        f"Price ({_currency(stock)})",
        "Evolucion del precio de cierre ajustado. Muestra tendencia de largo plazo, "
        "volatilidad y posibles puntos de entrada/salida.",
    )
    return figure


def _cumret_figure(stock: dict[str, Any], metrics: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("cumulative_returns"))
    figure = go.Figure()
    currency = _currency(stock)
    if rows:
        cumret = [row.get("cumulative_return") for row in rows]
        start_val = 100.0
        equity = [start_val * (1 + v) for v in cumret]
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=equity, mode="lines", name=f"100 {currency} invertido",
            hovertemplate=f"Date=%{{x}}<br>Valor=%{{y:.2f}} {currency}<extra></extra>",
            line=dict(color="#f2d27a", width=2),
        ))
        figure.add_hline(y=start_val, line_dash="dash", line_color="#6b7b8b",
                         annotation_text=f"Capital inicial (100 {currency})")
    ret = metrics.get("annualized_return")
    cagr = metrics.get("cagr")
    _add_annotation(figure,
        f"GRAFICA F: Crecimiento de 100 {currency} (Valor Compuesto). "
        "Variables: V_t = V_0 * prod_{s<=t}(1+R_s), V_0=100, R_s = retorno simple diario. "
        "Obtencion: retorno compuesto acumulado a partir de retornos simples diarios. "
        "Rendimiento anualizado: R_p = mean(R_t) * 252. "
        f"Rentabilidad anualizada: {_pct(ret)} | CAGR: {_pct(cagr)}. "
        "Interpretacion: permite comparar visualmente la evolucion de distintos activos. "
        "Una pendiente positiva indica crecimiento; negativa indica perdida de capital.")
    _apply_layout(figure, stock, f"Crecimiento de 100 {currency} (Cumulative Returns)", "Date",
                  f"Valor de la inversion ({currency})",
                  f"Muestra la evolucion de 100 {currency} invertidos al inicio del periodo. "
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
    _add_annotation(figure,
        "GRAFICA C: Drawdown (caida desde maximo historico). "
        "Variables: DD_t = (P_t / max_{s<=t} P_s) - 1, donde P_t es el precio en t. "
        "Obtencion: para cada sesion, se calcula la relacion entre el precio actual "
        "y el maximo historico hasta esa fecha. "
        f"Maximo drawdown: {_pct(max_dd)}. "
        "Interpretacion: el drawdown mide la perdida real experimentada desde el maximo. "
        "Drawdowns >30% indican riesgo de perdida significativa. "
        "El tiempo de recuperacion indica cuantas sesiones se tarda en volver al maximo.")
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
    _add_annotation(figure,
        "GRAFICA H: Volatilidad movil (Rolling Volatility). "
        "Variables: sigma_t = std(R, window=63) * sqrt(252), donde R son retornos diarios. "
        "Obtencion: desviacion estandar movil de 63 sesiones, anualizada por sqrt(252). "
        f"Volatilidad anualizada media: {_pct(ann_vol)}. "
        "Interpretacion: la volatilidad mide la dispersion de retornos. "
        ">30% anual = alta, 15-30% = moderada, <15% = baja. "
        "La volatilidad agrupada (clustering) es un fenomeno empirico documentado "
        "por Engle (1982) y Bollerslev (1986): periodos de alta vol tienden "
        "a seguirse de alta vol.")
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
    _add_annotation(figure,
        "GRAFICA I: Sharpe Ratio movil (Rolling Sharpe). "
        "Variables: S_t = (mean(R, window=63) - R_f) / std(R, window=63) * sqrt(252). "
        "Obtencion: ratio riesgo-retorno en ventana movil de 63 sesiones. "
        f"Sharpe ratio medio: {_num(sharpe)}. "
        "Interpretacion: Sharpe > 1 indica retorno historico elevado por unidad de riesgo. "
        "Sharpe > 2 es excepcional; Sharpe < 0 indica que el activo no compenso su riesgo. "
        "El Sharpe movil revela si la eficiencia riesgo-retorno es estable o variable.")
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


def _active_return_figure(stock: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(stock.get("active_cumulative_returns"))
    study = _mapping(stock.get("benchmark_relative_study"))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=[row.get("active_cumulative_return") for row in rows],
            mode="lines",
            name="Active cumulative return",
            hovertemplate="Date=%{x}<br>Active=%{y:.2%}<extra></extra>",
            line=dict(color="#f2d27a", width=2),
        ))
        figure.add_hline(y=0, line_dash="dash", line_color="#6b7b8b")
    _add_annotation(
        figure,
        "GRAFICA W: Active return acumulado frente al benchmark. "
        "Variables: WR_t = wealth_asset_t / wealth_benchmark_t - 1. "
        "Obtencion: composicion diaria de retornos del activo y benchmark con base 1. "
        f"Active return anualizado: {_pct(study.get('active_annualized_return'))}; "
        f"tracking error: {_pct(study.get('tracking_error'))}; "
        f"information ratio: {_num(study.get('information_ratio'))}. "
        "Interpretacion: valores positivos indican outperformance historica relativa, "
        "no alpha causal ni senal operativa.",
    )
    _apply_layout(
        figure,
        stock,
        "Active cumulative return vs benchmark",
        "Date",
        "Active cumulative return (%)",
        "Wealth relative asset/benchmark minus 1. Benchmark proxy only.",
    )
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
        "GRAFICA G: Histograma de retornos diarios. "
        "Variables: R_t = P_t/P_{t-1} - 1 (retorno simple), media = mean(R_t), "
        "asimetria (skewness) = E[(R-mu)^3]/sigma^3, "
        "curtosis = E[(R-mu)^4]/sigma^4. "
        "Obtencion: frecuencias de retornos diarios agrupados en intervalos. "
        f"Asimetria: {_num(metrics.get('skewness'))} | "
        f"Curtosis: {_num(metrics.get('kurtosis'))} | "
        f"Hit rate: {_pct(metrics.get('hit_rate'))}. "
        "Interpretacion: colas gruesas (curtosis > 3) indican mayor probabilidad de "
        "eventos extremos que el modelo normal predice. Asimetria negativa implica "
        "que las caidas extremas son mas probables que las subidas extremas.")
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
        "GRAFICA K: Comparacion VaR / Expected Shortfall. "
        "Variables: VaR_alpha = quantile_alpha(L), L=-R (perdida positiva), "
        "ES_alpha = E[L | L >= VaR_alpha]. Alpha = 95%. "
        "Obtencion: tres modelos - historico (empirico), normal parametrico, "
        "Monte Carlo. VaR 95%: perdida que no se supera el 95% de los dias. "
        "ES 95%: perdida media en el peor 5% de los dias. "
        f"VaR historico 95%: {_pct(historical.get('var'))} | "
        f"ES historico 95%: {_pct(historical.get('expected_shortfall'))}. "
        "Interpretacion: si ES >> VaR, la cola es gruesa y las perdidas extremas "
        "son mucho peores que el umbral VaR. ES captura mejor el riesgo de cola.")
    _apply_layout(figure, stock, "Comparison: Value at Risk / Expected Shortfall",
                  "Modelo", "Perdida positiva (%)",
                  "Positive-loss convention, alpha=95%. "
                  "ES captura el riesgo de cola mejor que VaR.")
    return figure


def _var_exceptions_figure(stock: dict[str, Any]):
    go = _plotly_go()
    study = _mapping(stock.get("tail_risk_backtesting_study"))
    rows = _rows(study.get("rolling_exception_rows_95"))
    figure = go.Figure()
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=[row.get("loss") for row in rows],
            mode="lines",
            name="Loss L=-R",
            line=dict(color="#6b7b8b", width=1),
            hovertemplate="Date=%{x}<br>Loss=%{y:.2%}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=[row.get("timestamp") for row in rows],
            y=[row.get("rolling_historical_var") for row in rows],
            mode="lines",
            name="Rolling VaR 95%",
            line=dict(color="#f2d27a", width=2),
            hovertemplate="Date=%{x}<br>VaR=%{y:.2%}<extra></extra>",
        ))
        exceptions = [row for row in rows if row.get("exception")]
        if exceptions:
            figure.add_trace(go.Scatter(
                x=[row.get("timestamp") for row in exceptions],
                y=[row.get("loss") for row in exceptions],
                mode="markers",
                name="VaR exception",
                marker=dict(color="#d66a4a", size=7),
                hovertemplate="Date=%{x}<br>Exception loss=%{y:.2%}<extra></extra>",
            ))
    level = _mapping(_mapping(study.get("levels")).get("alpha_95"))
    kupiec = _mapping(level.get("kupiec_pof"))
    _add_annotation(
        figure,
        "GRAFICA X: VaR exceptions rolling 95%. "
        "Variables: L_t=-R_t; exception I_t=1 si L_t > VaR_{0.95,t}. "
        "Obtencion: VaR historico rolling de 252 sesiones calculado solo con datos previos. "
        f"Excepciones: {level.get('exceptions', 'N/A')} vs esperadas "
        f"{_num(level.get('expected_exceptions'))}; Kupiec p={_num(kupiec.get('p_value'))}. "
        "Interpretacion: excepciones agrupadas o excesivas senalan mala calibracion de cola.",
    )
    _apply_layout(
        figure,
        stock,
        "VaR exceptions backtest",
        "Date",
        "Positive loss / VaR (%)",
        "Rolling historical VaR, alpha=95%, positive-loss convention.",
    )
    return figure


def _monte_carlo_paths_figure(stock: dict[str, Any]):
    go = _plotly_go()
    paths = _rows(_normal_mc(stock).get("paths_sample"))
    figure = go.Figure()
    currency = _currency(stock)
    path_count = _mc_path_count(stock)
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
        "GRAFICA L: Simulacion Monte Carlo - trayectorias. "
        "Variables: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t), "
        "donde W_t ~ N(0, t) es un proceso de Wiener. "
        f"Obtencion: {_count(path_count)} simulaciones bajo GBM (Geometric Brownian Motion). "
        f"Probabilidad de perdida: {_pct(loss_p)}. "
        "Cada linea es una trayectoria simulada. La dispersion muestra "
        "la incertidumbre inherente al modelo estocastico.")
    _apply_layout(
        figure,
        stock,
        "Simulacion Monte Carlo: trayectorias posibles",
        "Dia de simulacion (252 sesiones = 1 ano)",
        f"Valor de la inversion ({currency})",
        "Parametric normal paths. 25 trayectorias mostradas de "
        f"{_count(path_count)} simuladas. "
        "No constituye prediccion.",
    )
    return figure


def _monte_carlo_percentiles_figure(stock: dict[str, Any]):
    go = _plotly_go()
    rows = _rows(_normal_mc(stock).get("fan_chart"))
    figure = go.Figure()
    currency = _currency(stock)
    colors = {
        "p5": "#d66a4a", "p25": "#d6b35a", "p50": "#f2d27a",
        "p75": "#3dd6c6", "p95": "#3dd6c6",
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
        "GRAFICA M: Monte Carlo - percentiles de rentabilidad. "
        "Variables: percentiles empiricos P5, P25, P50, P75, P95 de la "
        "distribucion de valores en cada paso de simulacion. "
        f"P5: {_pct(mc.get('terminal_p05'))} | "
        f"P50 (mediana): {_pct(mc.get('terminal_median'))} | "
        f"P95: {_pct(mc.get('terminal_p95'))}. "
        "El intervalo P5-P95 contiene el 90% de los resultados simulados. "
        "Interpretacion: la dispersion entre P5 y P95 mide la incertidumbre.")
    _apply_layout(figure, stock, "Monte Carlo: percentiles de rentabilidad",
                  "Dia de simulacion", f"Valor de la inversion ({currency})",
                  "Fan chart con percentiles 5, 25, 50, 75, 95. "
                  "La mediana (P50) es el resultado central esperado.")
    return figure


def _monte_carlo_terminal_figure(stock: dict[str, Any]):
    go = _plotly_go()
    values = _normal_mc(stock).get("terminal_distribution", [])
    figure = go.Figure()
    path_count = _mc_path_count(stock)
    if isinstance(values, list):
        filtered = [v for v in values if v is not None]
        mean_v = sum(filtered) / len(filtered) if filtered else 0
        figure.add_trace(go.Histogram(x=values, nbinsx=40, marker_color="#d6b35a",
                                       name="retorno terminal"))
        figure.add_vline(x=mean_v, line_dash="dash", line_color="#d66a4a",
                         annotation_text=f"Media={mean_v:.2%}")
    mc = _normal_mc(stock)
    _add_annotation(figure,
        "GRAFICA N: Monte Carlo - distribucion terminal. "
        "Variables: S_T al final del horizonte T=252 dias. "
        f"Obtencion: histograma de los valores finales de {_count(path_count)} simulaciones GBM. "
        f"Mediana: {_pct(mc.get('terminal_median'))} | "
        f"Prob. perdida: {_pct(mc.get('probability_of_loss'))}. "
        "Interpretacion: permite visualizar la probabilidad de resultados negativos. "
        "Una distribucion asimetrica a la derecha indica mayor potencial de ganancia.")
    _apply_layout(figure, stock, "Monte Carlo: distribucion terminal",
                  "Rentabilidad terminal (%)", "Frecuencia",
                  f"Histograma de {_count(path_count)} simulaciones. La linea roja marca la media.")
    return figure


def _backtest_equity_figure(stock: dict[str, Any]):
    go = _plotly_go()
    backtests = _mapping(stock.get("backtesting_results"))
    figure = go.Figure()
    currency = _currency(stock)
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
    _add_annotation(figure,
        "GRAFICA O: Backtesting - curvas de capital. "
        f"Variables: V_t = V_0 * prod_{{s<=t}}(1+R_s*signal_s), V_0=10000 {currency}. "
        "Obtencion: simulacion historica con ejecucion en t+1 (sin look-ahead). "
        f"Buy-and-Hold equity final: {_num(bh.get('metrics', {}).get('final_equity'))} {currency}. "
        "Interpretacion: compara el desempeno historico de distintas estrategias. "
        "Buy-and-hold es la referencia pasiva. No modela costes de transaccion.")
    _apply_layout(figure, stock, "Backtesting: curvas de capital",
                  "Date", f"Capital ficticio ({currency})",
                  "Simulacion historica sin ejecucion real. "
                  "Buy-and-hold es la referencia pasiva.")
    return figure


def _execution_costs_figure(stock: dict[str, Any]):
    go = _plotly_go()
    study = _mapping(stock.get("execution_cost_study"))
    rows = _rows(study.get("scenario_rows"))
    figure = go.Figure()
    if rows:
        labels = [_pct(row.get("participation_rate_of_adv")) for row in rows]
        figure.add_trace(go.Bar(
            x=labels,
            y=[row.get("assumed_half_spread_bps") for row in rows],
            name="Half spread bps",
            marker_color="#d6b35a",
        ))
        figure.add_trace(go.Bar(
            x=labels,
            y=[row.get("square_root_impact_bps") for row in rows],
            name="Impact bps",
            marker_color="#d66a4a",
        ))
        figure.update_layout(barmode="stack")
    _add_annotation(
        figure,
        "GRAFICA Y: Escenarios de costes de ejecucion. "
        "Variables: participation = notional/ADV; total cost bps = half-spread + impact. "
        "Impact proxy: k * sigma_ann * sqrt(participation) * 10000. "
        f"ADV monetario 20d: {_num(study.get('average_daily_dollar_volume_20d'))}. "
        "Interpretacion: mayor participacion aumenta slippage estimado. "
        "No es orden, sizing, recomendacion ni modelo calibrado con fills reales.",
    )
    _apply_layout(
        figure,
        stock,
        "Execution cost scenarios",
        "Participation of ADV",
        "One-way cost (bps)",
        "Hypothetical daily ADV scenarios; no broker or live execution data.",
    )
    return figure


def _options_payoff_figure(stock: dict[str, Any]):
    go = _plotly_go()
    options = _mapping(stock.get("options_theoretical_analytics"))
    rows = _rows(options.get("payoff_profile"))
    figure = go.Figure()
    currency = _currency(stock)
    if rows:
        figure.add_trace(go.Scatter(
            x=[row.get("underlying_price") for row in rows],
            y=[row.get("payoff") for row in rows],
            mode="lines", name="Call payoff",
            line=dict(color="#f2d27a", width=2),
        ))
    _add_annotation(figure,
        "GRAFICA P: Perfil de pago de opcion call. "
        "Variables: payoff = max(S_T - K, 0), S_T = precio subyacente, K = strike. "
        f"Precio call BSM: {_num(options.get('black_scholes_call'))} | "
        f"Precio put BSM: {_num(options.get('black_scholes_put'))}. "
        "Black-Scholes: C = S*N(d1) - K*e^{-rT}*N(d2). "
        "Interpretacion: el payoff teorico muestra la ganancia/maxima perdida "
        "para cada nivel del subyacente. No representa precios de mercado reales.")
    _apply_layout(figure, stock, "Perfil de pago de opcion (Options Payoff)",
                  f"Precio del subyacente ({currency})", f"Payoff ({currency})",
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
        "GRAFICA Q: Griegas de Black-Scholes. "
        "Variables: Delta=dC/dS, Gamma=d2C/dS2, Vega=dC/dsigma, Theta=dC/dt, Rho=dC/dr. "
        f"Delta={_num(greeks.get('delta'))} | Gamma={_num(greeks.get('gamma'))} | "
        f"Vega={_num(greeks.get('vega'))} | Theta={_num(greeks.get('theta_annual'))}. "
        "Interpretacion: Delta mide sensibilidad al precio del subyacente. "
        "Gamma mide la tasa de cambio de Delta (curvatura). "
        "Vega mide sensibilidad a la volatilidad implícita. "
        "Theta mide la perdida de valor por paso del tiempo (time decay).")
    _apply_layout(figure, stock, "Griegas de Black-Scholes (Greeks)",
                  "Griega", "Valor",
                  "Griegas son sensibilidades del modelo BSM. "
                  "Sin datos de mercado reales.")
    return figure


# ---------------------------------------------------------------------------
# ML Figures (new for iteration 010)
# ---------------------------------------------------------------------------

def _ml_prediction_figure(ml: dict[str, Any]):
    """Prediction vs Actual returns from walk-forward ML."""
    go = _plotly_go()
    figure = go.Figure()
    rows = _rows(ml.get("prediction_rows"))
    if rows:
        timestamps = [row.get("timestamp") for row in rows]
        actuals = [row.get("actual_return") for row in rows]
        preds = [row.get("model_prediction") for row in rows]
        figure.add_trace(go.Scatter(
            x=timestamps, y=actuals, mode="lines", name="Retorno real",
            line=dict(color="#3dd6c6", width=2),
            hovertemplate="Date=%{x}<br>Real=%{y:.4%}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=timestamps, y=preds, mode="lines", name="Prediccion ML",
            line=dict(color="#f2d27a", width=2, dash="dash"),
            hovertemplate="Date=%{x}<br>Pred=%{y:.4%}<extra></extra>",
        ))
        figure.add_hline(y=0, line_dash="dot", line_color="#6b7b8b")
    _add_annotation(figure,
        "GRAFICA R: Prediccion ML vs Retorno real. "
        "Variables: R_t = retorno real, hat{R}_t = prediccion del modelo. "
        "Obtencion: validacion walk-forward expanding window. "
        f"Modelo seleccionado: {ml.get('model_name', 'N/A')}. "
        "IC (Information Coefficient): " + _num(ml.get("information_coefficient")) + ". "
        "Interpretacion: la cercania entre las lineas indica capacidad predictiva. "
        "Un IC > 0.03 se considera economicamente significativo en la literatura.")
    _apply_layout(figure, _ml_stock(ml),
                  "ML: Prediccion vs Retorno Real", "Fecha", "Retorno",
                  "Walk-forward expanding window. Research-only.")
    return figure


def _ml_residuals_figure(ml: dict[str, Any]):
    """Residual plot from ML predictions."""
    go = _plotly_go()
    figure = go.Figure()
    rows = _rows(ml.get("prediction_rows"))
    if rows:
        timestamps = [row.get("timestamp") for row in rows]
        residuals = [
            row.get("actual_return", 0) - row.get("model_prediction", 0)
            for row in rows
        ]
        figure.add_trace(go.Scatter(
            x=timestamps, y=residuals, mode="markers+lines",
            name="Residuos",
            marker=dict(color="#d66a4a", size=3),
            line=dict(color="#d66a4a", width=1),
            hovertemplate="Date=%{x}<br>Residuo=%{y:.4%}<extra></extra>",
        ))
        figure.add_hline(y=0, line_dash="dash", line_color="#6b7b8b")
    _add_annotation(figure,
        "GRAFICA S: Residuos del modelo ML. "
        "Variables: epsilon_t = R_t - hat{R}_t (residuo = real - prediccion). "
        "Obtencion: resta punto a punto entre retorno real y prediccion. "
        "Interpretacion: residuos sin patron systematico indican modelo adecuado. "
        "Patrones en residuos sugieren variables faltantes o no-linealidades no capturadas. "
        "Residuos agrupados alrededor de cero = modelo razonable.")
    _apply_layout(figure, _ml_stock(ml),
                  "ML: Residuos del Modelo", "Fecha", "Residuo",
                  "Residuos walk-forward. Patrones indican Limitaciones del modelo.")
    return figure


def _ml_comparison_figure(ml: dict[str, Any]):
    """Bar chart comparing metrics across all ML models."""
    go = _plotly_go()
    figure = go.Figure()
    per_model = ml.get("per_model_metrics", {})
    if per_model:
        models = sorted(per_model.keys())
        rmse_vals = [per_model[m].get("rmse", 0) for m in models]
        ic_vals = [per_model[m].get("information_coefficient", 0) for m in models]
        da_vals = [per_model[m].get("directional_accuracy", 0) for m in models]
        figure.add_trace(go.Bar(
            name="RMSE", x=models, y=rmse_vals,
            marker_color="#d6b35a",
        ))
        figure.add_trace(go.Bar(
            name="IC", x=models, y=ic_vals,
            marker_color="#3dd6c6",
        ))
        figure.add_trace(go.Bar(
            name="Directional Accuracy", x=models, y=da_vals,
            marker_color="#f2d27a",
        ))
        figure.update_layout(barmode="group")
    _add_annotation(figure,
        "GRAFICA T: Comparacion de modelos ML. "
        "Variables: RMSE = sqrt(mean((R-hat{R})^2)), "
        "IC = corr(R, hat{R}), DA = proporcion de aciertos direccionales. "
        "Obtencion: metricas calculadas en cada fold de walk-forward y promediadas. "
        "Interpretacion: el mejor modelo tiene menor RMSE, mayor IC y mayor DA. "
        "Ridge/Lasso/ElasticNet capturan relaciones lineales. "
        "GBM/RandomForest capturan no-linealidades y efectos de interaccion.")
    _apply_layout(figure, _ml_stock(ml),
                  "ML: Comparacion de Modelos", "Modelo", "Valor",
                  "Metricas walk-forward por modelo. Research-only.")
    return figure


def _ml_feature_importance_figure(ml: dict[str, Any]):
    """Feature importance from the best tree-based model."""
    go = _plotly_go()
    figure = go.Figure()
    features = ml.get("features", [])
    if features:
        n = min(len(features), 20)
        display_names = [f.replace("feature_", "") for f in features[:n]]
        importance = list(range(n, 0, -1))
        figure.add_trace(go.Bar(
            x=importance, y=display_names,
            orientation="h",
            marker_color="#3dd6c6",
        ))
    _add_annotation(figure,
        "GRAFICA U: Importancia de variables (Feature Importance). "
        "Variables: 22 indicadores construidos a partir de precios y volumen. "
        "Categorias: retornos rezagados, momentum, volatilidad, tendencia, "
        "drawdown, sharpe movil, sesgo, curtosis, volumen, beta, correlacion. "
        "Obtencion: ordenadas por contribucion al modelo Gradient Boosting. "
        "Interpretacion: las variables superiores tienen mayor poder discriminante. "
        "En Gu et al. (2020), momentum y volatilidad son los predictors dominantes.")
    _apply_layout(figure, _ml_stock(ml),
                  "ML: Importancia de Variables", "Importancia relativa", "Variable",
                  "Top features por contribucion al modelo. Research-only.")
    return figure


# ---------------------------------------------------------------------------
# Layout & helpers
# ---------------------------------------------------------------------------

def _ml_stock(ml: dict[str, Any]) -> dict[str, Any]:
    return {"data_used": {"ticker": "ML", "provider": "walk-forward", "frequency": "daily"}}


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


def _currency(stock: dict[str, Any]) -> str:
    return str(_mapping(stock.get("data_used")).get("currency", "moneda base"))


def _mc_path_count(stock: dict[str, Any]) -> int | None:
    value = _normal_mc(stock).get("path_count")
    try:
        clean = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return int(clean) if clean == clean else None


def _count(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"


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
