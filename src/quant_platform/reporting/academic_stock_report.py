"""Academic stock report model, Markdown/HTML rendering, and file writing."""
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_platform.reporting.academic_conclusions import build_stock_conclusions
from quant_platform.reporting.academic_figures import (
    FIGURE_FILENAMES,
    write_academic_stock_figures,
)
from quant_platform.reporting.pdf_export import export_html_report_to_pdf
from quant_platform.research.bibliography import method_catalog

SECTION_TITLES = [
    "1. Portada",
    "2. Executive Summary",
    "3. Plain-English Summary",
    "4. Research Question",
    "5. Data & Provenance",
    "6. Data Quality Review",
    "7. Price Dynamics",
    "8. Return Construction",
    "9. Performance Metrics",
    "10. CAPM Metrics",
    "11. Value at Risk Analysis",
    "12. Monte Carlo Simulation",
    "13. ML Forecasting Assessment",
    "14. Backtesting Analysis",
    "15. Options Analytics",
    "16. Comparison Against Benchmark",
    "17. Quantitative Decision Signal",
    "18. Statistical Interpretation",
    "19. Model Performance Assessment",
    "20. Stock-Specific Conclusions",
    "21. Limitations",
    "22. Reproducibility",
    "23. Mathematical Appendix",
    "24. Bibliography & Method Traceability",
]

FORMULAS = [
    ("Simple return", "R_t = P_t / P_{t-1} - 1"),
    ("Log return", "r_t = ln(P_t / P_{t-1})"),
    ("Continuous compounding", "P_t = P_0 * exp(sum r_t)"),
    ("Annualized volatility", "sigma_ann = std(R_t) * sqrt(252)"),
    ("Sharpe", "S = mean(R_t - R_f) / std(R_t - R_f) * sqrt(252)"),
    ("Sortino", "Sortino = mean(R_t - R_f) / downside_deviation * sqrt(252)"),
    ("Max drawdown", "DD_t = V_t / max_{s<=t}(V_s) - 1"),
    ("Beta", "Beta_i = Cov(R_i, R_m) / Var(R_m)"),
    ("Treynor", "Treynor_i = (R_i - R_f) / Beta_i"),
    ("Jensen alpha", "Alpha_i = R_i - [R_f + Beta_i * (R_m - R_f)]"),
    ("Historical VaR", "VaR_alpha(L) = quantile_alpha(L), L=-R_t"),
    ("Expected Shortfall", "ES_alpha = E[L | L >= VaR_alpha]"),
    ("GBM", "S_t = S_0 exp((mu - 0.5 sigma^2)t + sigma W_t)"),
    ("Black-Scholes call", "C = S exp(-qT) N(d1) - K exp(-rT) N(d2)"),
    ("Put-call parity", "C - P = S exp(-qT) - K exp(-rT)"),
]


class AcademicStockReportError(ValueError):
    """Raised when academic stock report inputs are invalid."""


def build_stock_academic_report_model(full_report: dict[str, Any], asset_id: str) -> dict[str, Any]:
    """Build a JSON-safe model for one stock academic report."""

    stocks = full_report.get("stocks", {})
    if not isinstance(stocks, dict) or asset_id not in stocks:
        raise AcademicStockReportError(
            f"Asset {asset_id} is not present in terminal report stocks."
        )
    stock = stocks[asset_id]
    if not isinstance(stock, dict):
        raise AcademicStockReportError(f"Stock payload for {asset_id} must be an object.")
    conclusions = build_stock_conclusions(stock, asset_id=asset_id)
    model = {
        "asset_id": asset_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "terminal_metadata": full_report.get("metadata", {}),
        "data_provenance": full_report.get("data_provenance", full_report.get("data", {})),
        "universe": full_report.get("universe", {}),
        "stock": stock,
        "metrics": stock.get("metrics", {}),
        "conclusions": conclusions,
        "bibliography": full_report.get("bibliography", method_catalog()),
        "methods_review": _methods_review_summary(),
        "figure_links": {
            name: f"figures/{filename}" for name, filename in FIGURE_FILENAMES.items()
        },
        "warning": "Research-only. This is not investment advice.",
    }
    return model


def render_stock_report_markdown(report_model: dict[str, Any]) -> str:
    """Render one stock academic report as Markdown."""

    asset_id = str(report_model["asset_id"])
    stock = _mapping(report_model.get("stock"))
    metrics = _mapping(report_model.get("metrics"))
    data_used = _mapping(stock.get("data_used"))
    conclusions = _mapping(report_model.get("conclusions"))
    lines = [
        f"# Academic Quant Research Report - {asset_id}",
        "",
        "**Research-only. This is not investment advice.**",
        "",
        "## 1. Portada",
        "",
        _key_value_table(
            {
                "Stock": asset_id,
                "Period": f"{data_used.get('start_timestamp')} to {data_used.get('end_timestamp')}",
                "Frequency": data_used.get("frequency"),
                "Currency": data_used.get("currency"),
                "Provider": data_used.get("provider"),
                "Data mode": data_used.get("data_mode"),
                "Benchmark": data_used.get("benchmark"),
                "Risk-free proxy": data_used.get("risk_free_proxy"),
                "Risk-free rate": _pct(data_used.get("risk_free_rate_annual")),
                "Generated at": report_model.get("generated_at"),
                "Report version": _mapping(report_model.get("terminal_metadata")).get(
                    "report_version"
                ),
            }
        ),
        "",
        "## 2. Executive Summary",
        "",
        _executive_summary(asset_id, metrics, conclusions),
        "",
        "## 3. Plain-English Summary",
        "",
        _plain_summary(asset_id, conclusions),
        "",
    ]
    for title in SECTION_TITLES[3:]:
        lines.extend(_render_section(title, report_model))
    return "\n".join(lines).strip() + "\n"


def render_stock_report_html(report_model: dict[str, Any]) -> str:
    """Render one stock academic report as standalone HTML with inline figures."""

    markdown = render_stock_report_markdown(report_model)
    body = _markdown_to_basic_html(markdown)
    figure_paths = report_model.get("figure_paths", {})
    if figure_paths:
        body = _embed_figures_inline(body, figure_paths, str(report_model.get("asset_id", "")))
    asset_id = html.escape(str(report_model["asset_id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{asset_id} Academic Quant Research Report</title>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    body {{ background: #081018; color: #e8ecef; font-family: Georgia, 'Times New Roman', serif; margin: 0; line-height: 1.6; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 48px 28px 80px; }}
    h1, h2, h3 {{ color: #f2d27a; font-family: Aptos, Segoe UI, sans-serif; }}
    a {{ color: #3dd6c6; }}
    code, pre {{ background: #121c25; color: #f2f0df; padding: 2px 6px; border-radius: 6px; font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
    th, td {{ border: 1px solid #2c3a46; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #142231; color: #f2d27a; }}
    .warning {{ border: 1px solid #d66a4a; padding: 14px; border-radius: 12px; background: rgba(214,106,74,0.12); }}
    .figure-box {{ background: #0d1520; border: 1px solid #2c3a46; border-radius: 12px; padding: 12px; margin: 18px 0; }}
    .figure-box .plotly-graph-div {{ height: 450px !important; }}
    .mjx-chtml {{ font-size: 110% !important; }}
  </style>
</head>
<body><main>{body}</main></body>
</html>
"""


def write_stock_academic_report(
    report_model: dict[str, Any],
    output_dir: str | Path,
    formats: tuple[str, ...] = ("md", "html"),
    include_figures: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write Markdown/HTML academic report, figures, and metadata for one stock."""

    asset_id = str(report_model["asset_id"])
    stock_dir = Path(output_dir) / asset_id
    if stock_dir.exists() and not overwrite:
        existing = list(stock_dir.glob("*"))
        if existing:
            raise AcademicStockReportError(f"Output directory already contains files: {stock_dir}")
    stock_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, str] = {}
    if include_figures:
        figure_paths = write_academic_stock_figures(report_model, stock_dir / "figures")
        report_model["figure_paths"] = figure_paths
    outputs: dict[str, str] = {}
    format_set = {fmt.lower() for fmt in formats}
    if "md" in format_set:
        markdown_path = stock_dir / f"{asset_id}_academic_report.md"
        markdown_path.write_text(render_stock_report_markdown(report_model), encoding="utf-8")
        outputs["md"] = str(markdown_path)
    html_path = stock_dir / f"{asset_id}_academic_report.html"
    if "html" in format_set or "pdf" in format_set:
        html_path.write_text(render_stock_report_html(report_model), encoding="utf-8")
        outputs["html"] = str(html_path)
    pdf_export: dict[str, Any] | None = None
    if "pdf" in format_set:
        pdf_path = stock_dir / f"{asset_id}_academic_report.pdf"
        pdf_export = export_html_report_to_pdf(html_path, pdf_path)
        if pdf_export.get("success") and pdf_export.get("pdf_path"):
            outputs["pdf"] = str(pdf_export["pdf_path"])
    metadata = {
        "report_type": "academic_stock_report_metadata",
        "asset_id": asset_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "outputs": outputs,
        "pdf_export": pdf_export,
        "figures": figure_paths,
        "formats": sorted(format_set),
        "research_only": True,
        "not_investment_advice": True,
        "summary": _executive_summary(
            asset_id,
            _mapping(report_model.get("metrics")),
            _mapping(report_model.get("conclusions")),
        ),
    }
    metadata_path = stock_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def write_all_stock_academic_reports(
    full_report: dict[str, Any],
    output_dir: str | Path,
    formats: tuple[str, ...] = ("md", "html"),
    include_figures: bool = True,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write academic reports for every stock in a terminal report."""

    stocks = full_report.get("stocks", {})
    if not isinstance(stocks, dict) or not stocks:
        raise AcademicStockReportError("Terminal report has no stocks payload.")
    results = []
    for asset_id in sorted(stocks):
        model = build_stock_academic_report_model(full_report, str(asset_id))
        results.append(
            write_stock_academic_report(
                model,
                output_dir=output_dir,
                formats=formats,
                include_figures=include_figures,
                overwrite=overwrite,
            )
        )
    return results


def _render_section(title: str, report_model: dict[str, Any]) -> list[str]:
    stock = _mapping(report_model.get("stock"))
    metrics = _mapping(report_model.get("metrics"))
    data_used = _mapping(stock.get("data_used"))
    figures = _mapping(report_model.get("figure_links"))
    conclusions = _mapping(report_model.get("conclusions"))
    section_key = title.split(". ", 1)[-1]
    plain, technical, formulas, figure_keys, limitations = _section_content(section_key)
    lines = [f"## {title}", "", "### Explicacion divulgativa", "", plain, ""]
    lines.extend(["### Explicacion tecnica", "", technical, ""])
    if formulas:
        lines.extend(["### Formulas", ""])
        for name, formula in formulas:
            lines.append(f"- **{name}:** $$ {formula} $$")
        lines.append("")
    lines.extend(["### Parametros y datos usados", ""])
    lines.append(
        _key_value_table(
            {
                "Asset": report_model.get("asset_id"),
                "Provider": data_used.get("provider"),
                "Frequency": data_used.get("frequency"),
                "Observations": data_used.get("observations"),
                "Benchmark": data_used.get("benchmark"),
                "Risk-free rate": _pct(data_used.get("risk_free_rate_annual")),
            }
        )
    )
    lines.append("")
    if section_key == "Data & Provenance":
        lines.extend([_data_table(stock), ""])
    if section_key in {"Performance Metrics", "CAPM Metrics", "Quantitative Decision Signal"}:
        lines.extend([_metrics_table(metrics), ""])
    if section_key == "Bibliography & Method Traceability":
        lines.extend([_bibliography_table(report_model), ""])
    if section_key == "Mathematical Appendix":
        lines.extend([_formula_table(), ""])
    if figure_keys:
        lines.extend(["### Graficas asociadas", ""])
        for key in figure_keys:
            if key in figures:
                lines.append(f"- [{key}]({figures[key]})")
        lines.append("")
    lines.extend(["### Conclusiones", ""])
    lines.append(_section_conclusion(section_key, conclusions))
    lines.extend(["", "### Limitaciones", "", limitations, ""])
    return lines


def _section_content(section: str) -> tuple[str, str, list[tuple[str, str]], list[str], str]:
    common_limit = (
        "Estas mediciones describen la muestra historica y dependen de proveedor, ventana, "
        "frecuencia, limpieza, supuestos y estabilidad estadistica."
    )
    content = {
        "Research Question": (
            "La pregunta es que se puede aprender historicamente de este stock sin convertirlo en consejo operativo.",
            "Se evalua comportamiento de precio, retorno, riesgo, benchmark sensitivity, simulaciones y modelos parametrizados.",
            [],
            [],
            common_limit,
        ),
        "Data & Provenance": (
            "Esta seccion dice de donde salen los datos y que periodo cubren.",
            "El dataset usa OHLCV diario, proveedor registrado, benchmark, moneda y proxy/configuracion de tasa libre de riesgo.",
            [],
            [],
            "Datos gratuitos pueden tener revisiones, gaps, licencias y sesgos de supervivencia.",
        ),
        "Data Quality Review": (
            "Antes de interpretar graficas se revisa si hay precios faltantes, duplicados o volumen anomalo.",
            "Los checks resumen missing values, duplicados y observaciones no positivas bajo convenciones OHLCV.",
            [],
            [],
            "No reemplaza auditoria profesional de corporate actions, delistings o licencias.",
        ),
        "Price Dynamics": (
            "GRAFICA A (Price History): evolucion del precio de cierre ajustado. Muestra la tendencia de largo plazo "
            "y permite identificar visualmente periodos alcistas, bajistas y de lateralizacion. "
            "GRAFICA B (Volume): volumen de negociacion diario. Picos de volumen suelen coincidir con eventos "
            "significativos (resultados, noticias macro, cambios de tendencia). "
            "GRAFICA C (Drawdown): caida porcentual desde el maximo historico. Identifica visualmente las "
            "correcciones y el riesgo de caida real experimentado. El drawdown maximo es la peor caida registrada.",
            "El precio de cierre se obtiene de OHLCV diario. Drawdown = (P_t / max_{s<=t} P_s) - 1. "
            "Volumen en acciones negociadas. Estas tres series combinadas permiten evaluar: "
            "(1) tendencia direccional, (2) liquidez y actividad, (3) riesgo de caida real.",
            [("Max drawdown", "DD_t = V_t / max_{s<=t}(V_s) - 1")],
            ["price_history", "volume", "drawdown"],
            common_limit,
        ),
        "Return Construction": (
            "GRAFICA D (Simple Returns): retorno diario simple, muestra magnitud de movimientos. "
            "GRAFICA E (Log Returns): retorno logaritmico, usado en modelos continuos. "
            "GRAFICA F (Cumulative Returns): crecimiento de 100 EUR, muestra el poder del interes compuesto. "
            "GRAFICA G (Returns Distribution): histograma de retornos diarios con media marcada. "
            "Interpretacion economica: la prima de riesgo historica se refleja en la pendiente de F. "
            "Cuanto mas pronunciada la curva F, mayor rentabilidad compuesta.",
            "R_t = P_t/P_{t-1} - 1. r_t = ln(P_t/P_{t-1}). "
            "V_t = 100 * prod(1+R_s). Retorno anualizado = media(R_t) * 252. "
            "La relacion entre R_t y r_t: r_t = ln(1+R_t). Para R_t pequeno, son casi iguales.",
            FORMULAS[:3],
            ["simple_returns", "log_returns", "cumulative_returns", "returns_distribution"],
            common_limit,
        ),
        "Performance Metrics": (
            "GRAFICA F (Cumulative Returns): 100 EUR invertidos al inicio. "
            "GRAFICA I (Rolling Sharpe): eficiencia riesgo-retorno a lo largo del tiempo. "
            "TABLA DE METRICAS: Annualized Return, CAGR, Volatilidad, Sharpe, Sortino, Calmar, Hit Rate. "
            "Interpretacion economica: Sharpe > 1 indica que el retorno compensa el riesgo total. "
            "Sortino > 1 indica que compensa especificamente el riesgo de caida. "
            "Calmar relaciona el retorno anual con el peor drawdown: un Calmar alto es senal de solidez.",
            "R_p = media(R_t)*252. sigma_p = std(R_t)*sqrt(252). "
            "Sharpe = (R_p - R_f)/sigma_p. Sortino = (R_p - R_f)/downside_dev. "
            "Calmar = R_p/|max_drawdown|. Hit rate = count(R_t > 0)/N. "
            "Todas las metricas son historicas y ventana-dependentes.",
            FORMULAS[3:6],
            ["cumulative_returns", "rolling_sharpe"],
            common_limit,
        ),
        "Risk Metrics": (
            "GRAFICA C (Drawdown): caidas desde maximos historicos. "
            "GRAFICA H (Rolling Volatility): volatilidad movil 63 sesiones. "
            "GRAFICA G (Returns Distribution): histograma con asimetria y curtosis visibles. "
            "Interpretacion economica: un activo con drawdown >30% requiere alta tolerancia al riesgo. "
            "Skewness negativo significa que las caidas extremas son mas probables que las subidas extremas. "
            "Kurtosis > 3 implica colas mas gruesas que la normal: eventos extremos mas frecuentes.",
            "sigma_rolling = rolling_std(R_t, window=63) * sqrt(252). "
            "Skewness = E[(R-mu)^3]/sigma^3. Kurtosis = E[(R-mu)^4]/sigma^4. "
            "Max DD = min(P_t/max_{s<=t}P_s - 1). "
            "Estas metricas describen el riesgo real experimentado, no solo la volatilidad.",
            FORMULAS[3:7],
            ["drawdown", "rolling_volatility", "returns_distribution"],
            common_limit,
        ),
        "CAPM Metrics": (
            "GRAFICA J (Rolling Beta): estabilidad de la sensibilidad al mercado. "
            "TABLA CAPM: Beta, Treynor, Jensen Alpha. "
            "Interpretacion economica: Beta indica cuanto riesgo de mercado tiene el activo. "
            "Alpha positivo significa que el activo rindio mas de lo esperado por su riesgo sistematico. "
            "Si Beta es cercano a 0, el activo es casi independiente del mercado (defensivo/descCorrelacionado).",
            "Beta_i = Cov(R_i, R_m)/Var(R_m). Alpha_i = R_i - [R_f + Beta_i*(R_m - R_f)]. "
            "Treynor_i = (R_i - R_f)/Beta_i. CAPM: E[R_i] = R_f + Beta_i*(E[R_m]-R_f). "
            "Beta estima el riesgo sistematico; alpha mide el valor anadido (o destruido).",
            FORMULAS[7:10],
            ["rolling_beta"],
            "El benchmark no es el mercado completo y beta puede cambiar por ventana temporal.",
        ),
        "Value at Risk Analysis": (
            "GRAFICA K (VaR/ES Comparison): comparacion de modelos (historico, normal, MC). "
            "GRAFICA G (Returns Distribution): la cola izquierda muestra donde estan las perdidas. "
            "Interpretacion economica: VaR 95% = perdida que no se supera el 95% de los dias. "
            "ES 95% = perdida media en el peor 5% de los dias. "
            "Si ES >> VaR, la cola es gruesa: las perdidas extremas son mucho peores que el umbral.",
            "VaR_alpha(L) = quantile_alpha(L), L = -R (perdida positiva). "
            "ES_alpha = E[L | L >= VaR_alpha]. Alpha = 95%. "
            "Tres modelos: historico (empirico), normal parametrico, Monte Carlo.",
            FORMULAS[10:12],
            ["var_comparison", "returns_distribution"],
            "VaR no mide todo lo que ocurre mas alla del umbral y depende del modelo.",
        ),
        "Monte Carlo Simulation": (
            "GRAFICA L (MC Paths): 25 trayectorias simuladas de 1000. "
            "GRAFICA M (MC Percentiles): P5, P25, P50, P75, P95. "
            "GRAFICA N (MC Terminal Distribution): histograma de resultados finales. "
            "Interpretacion economica: la dispersion entre P5 y P95 mide la incertidumbre. "
            "P50 (mediana) es el escenario central. Probabilidad de perdida = % de simulaciones con resultado < 0. "
            "Un activo con P5 muy negativo pero P95 muy positivo tiene alta dispersion (alto riesgo).",
            "GBM: S_t = S_0 exp((mu - 0.5*sigma^2)t + sigma*W_t). "
            "1000 simulaciones, horizonte 252 dias. Parametros estimados de la serie historica. "
            "Percentiles empiricos de la distribucion terminal. Seed reproducible.",
            [("GBM", FORMULAS[12][1])],
            ["monte_carlo_paths", "monte_carlo_percentiles", "monte_carlo_terminal_distribution"],
            "Cambiar modelo, seed, horizonte o ventana puede cambiar las conclusiones simuladas.",
        ),
        "ML Forecasting Assessment": (
            "Esta seccion evalua si un modelo predictivo supera baselines simples en validacion walk-forward.",
            "Los targets permitidos son retorno proximo, retorno forward a 5 dias y direccion proxima. "
            "La validacion no mezcla futuro con pasado: cada test ocurre despues de su periodo de entrenamiento.",
            [],
            ["ml_prediction_vs_actual", "ml_residuals"],
            "Un modelo que no supera el baseline naive debe marcarse como diagnostico debil, no como prediccion fiable.",
        ),
        "Backtesting Analysis": (
            "GRAFICA O (Backtest Equity Curves): capital ficticio de cada estrategia. "
            "Interpretacion economica: buy-and-hold con 10.000 EUR iniciales. "
            "El equity final muestra la rentabilidad neta. "
            "El drawdown del backtest muestra el riesgo real de la estrategia. "
            "Comparacion entre estrategias indica que reglas funcionaron mejor en el pasado.",
            "Equity curve: V_t = V_0 * prod(1 + R_s * signal_s). "
            "Convencion: datos disponibles hasta t, ejecucion en t+1 (sin look-ahead). "
            "Estrategias: buy-and-hold (referencia pasiva), y otras segun config.",
            [("Equity curve", "V_t = V_0 prod_{s<=t}(1+R_s)")],
            ["backtest_equity_curves"],
            "No modela fills reales, liquidez intradia ni impacto de mercado profesional.",
        ),
        "Options Analytics": (
            "GRAFICA P (Options Payoff): perfil de pago de opcion call. "
            "GRAFICA Q (Greeks): sensibilidades BSM. "
            "Interpretacion economica: opciones permiten apalancar o cubrir posiciones. "
            "Delta indica cuantas unidades del subyacente replica una opcion. "
            "Gamma mide cuanto cambia Delta. Vega mide sensibilidad a la volatilidad. "
            "Theta mide perdida de valor por paso del tiempo.",
            "Black-Scholes: C = S*N(d1) - K*e^{-rT}*N(d2). P = C - S + K*e^{-rT}. "
            "Griegas: dC/dS, d2C/dS2, dC/dsigma, dC/dt, dC/dr. "
            "Strike ATM, madurez 30 dias, volatilidad historica, tipo libre de riesgo.",
            FORMULAS[13:15],
            ["options_payoff", "greeks"],
            "PARAMETRIC_EDUCATIONAL_MODEL: sin option chain, smile, dividendos reales ni microestructura.",
        ),
        "Comparison Against Benchmark": (
            "El benchmark sirve como referencia, no como verdad absoluta.",
            "La comparacion usa beta y alpha frente al benchmark configurado.",
            FORMULAS[7:10],
            ["rolling_beta"],
            "SPY u otro proxy no captura todo el conjunto de oportunidades de mercado.",
        ),
        "Statistical Interpretation": (
            "La estabilidad importa: una metrica puede cambiar si cambia la ventana.",
            "Se consideran tamano muestral, colas, asimetria, curtosis, outliers, overfitting y sesgo de supervivencia.",
            [],
            ["returns_distribution", "rolling_volatility"],
            common_limit,
        ),
        "Model Performance Assessment": (
            "Un modelo util debe explicar sus supuestos y errores posibles.",
            "Se evalua coherencia de metricas, VaR/ES, Monte Carlo y backtest bajo restricciones de research-only.",
            [],
            ["var_comparison", "monte_carlo_percentiles", "backtest_equity_curves"],
            common_limit,
        ),
        "Stock-Specific Conclusions": (
            "Esta seccion resume lo que se puede decir del stock sin convertirlo en instruccion operativa.",
            "Las conclusiones son reglas deterministicas basadas en metricas del report, no texto generativo externo.",
            [],
            [],
            "No debe interpretarse como asesoramiento financiero ni prediccion.",
        ),
        "Quantitative Decision Signal": (
            "Research-only quantitative decision signal resume metricas historicas, riesgo de cola, "
            "Monte Carlo, ML, backtests y calidad de datos en una salida no operativa.",
            "Valores permitidos: FAVORABLE, NEUTRAL, CAUTION, UNFAVORABLE, INSUFFICIENT_DATA. "
            "La senal describe evidencia historica bajo supuestos; no calcula posicion, acciones ni capital.",
            [],
            ["decision_score_decomposition"],
            "This is not investment advice. It is a quantitative research signal based on historical data, assumptions, and model limitations.",
        ),
        "Limitations": (
            "Todo modelo simplifica la realidad.",
            "Limitaciones: proveedor gratuito, sesgos de muestra, modelos parametrizados, no option chain, no curvas reales, no ejecucion real.",
            [],
            [],
            "Las limitaciones son parte del resultado, no notas menores.",
        ),
        "Reproducibility": (
            "El informe indica parametros para repetirlo localmente.",
            "Incluye config, seed, horizonte, provider/data mode, frecuencia, benchmark, report version y rutas de artefactos.",
            [],
            [],
            "La red o providers publicos pueden devolver datos distintos en otro momento.",
        ),
        "Mathematical Appendix": (
            "Aqui se agrupan las formulas para consulta.",
            "Variables, unidades, annualization, perdidas positivas y convenciones temporales quedan explicitas.",
            FORMULAS,
            [],
            "La validacion page-level de varias fuentes sigue pendiente cuando asi esta documentado.",
        ),
        "Bibliography & Method Traceability": (
            "Cada metodo debe estar trazado a una fuente o marcado como pendiente.",
            "La tabla consume `professional_quant_methods_review.md` y `research.bibliography.method_catalog` sin inventar paginas.",
            [],
            [],
            "PENDING_PAGE_LEVEL_VALIDATION significa que no se debe afirmar validacion por pagina.",
        ),
    }
    return content.get(section, ("Seccion informativa.", "Detalle tecnico.", [], [], common_limit))


def _section_conclusion(section: str, conclusions: dict[str, Any]) -> str:
    mapping = {
        "Performance Metrics": "Historical performance",
        "CAPM Metrics": "CAPM interpretation",
        "Value at Risk Analysis": "Tail-risk interpretation",
        "Monte Carlo Simulation": "Monte Carlo interpretation",
        "ML Forecasting Assessment": "ML forecasting interpretation",
        "Backtesting Analysis": "Backtesting interpretation",
        "Options Analytics": "Options interpretation",
        "Quantitative Decision Signal": "Decision signal",
        "Statistical Interpretation": "Risk profile",
        "Stock-Specific Conclusions": "Overall research conclusion",
    }
    key = mapping.get(section)
    if key and key in conclusions:
        return str(conclusions[key])
    return "La interpretacion debe leerse historicamente, bajo supuestos del modelo y sin extrapolar a decisiones operativas."


def _executive_summary(asset_id: str, metrics: dict[str, Any], conclusions: dict[str, Any]) -> str:
    return (
        f"Para {asset_id}, en el periodo analizado, el rendimiento acumulado fue "
        f"{_pct(metrics.get('final_cumulative_return'))}, la volatilidad anualizada fue "
        f"{_pct(metrics.get('annualized_volatility'))}, el maximo drawdown fue "
        f"{_pct(metrics.get('max_drawdown'))}, Sharpe fue {_num(metrics.get('sharpe_ratio'))}, "
        f"Sortino fue {_num(metrics.get('sortino_ratio'))}, beta fue "
        f"{_num(metrics.get('beta_to_benchmark'))} y Jensen alpha fue "
        f"{_pct(metrics.get('jensen_alpha'))}. {conclusions.get('Monte Carlo interpretation', '')} "
        f"{conclusions.get('Backtesting interpretation', '')} No implica prediccion ni asesoramiento."
    )


def _plain_summary(asset_id: str, conclusions: dict[str, Any]) -> str:
    return (
        f"Este documento intenta explicar {asset_id} como un informe de laboratorio: primero mira "
        "datos, luego transforma precios en retornos, despues mide riesgo y finalmente prueba modelos. "
        "La intuicion es similar a revisar el historial medico de un paciente: ayuda a entender el pasado, "
        "pero no promete el futuro. "
        f"{conclusions.get('Overall research conclusion', '')}"
    )


def _key_value_table(values: dict[str, Any]) -> str:
    rows = ["| Campo | Valor |", "| --- | --- |"]
    for key, value in values.items():
        rows.append(f"| {key} | {value} |")
    return "\n".join(rows)


def _data_table(stock: dict[str, Any]) -> str:
    return _key_value_table(
        {**_mapping(stock.get("data_used")), **_mapping(stock.get("data_quality"))}
    )


def _metrics_table(metrics: dict[str, Any]) -> str:
    selected = {
        "annualized_return": _pct(metrics.get("annualized_return")),
        "cagr": _pct(metrics.get("cagr")),
        "annualized_volatility": _pct(metrics.get("annualized_volatility")),
        "sharpe_ratio": _num(metrics.get("sharpe_ratio")),
        "sortino_ratio": _num(metrics.get("sortino_ratio")),
        "calmar_ratio": _num(metrics.get("calmar_ratio")),
        "hit_rate": _pct(metrics.get("hit_rate")),
        "skewness": _num(metrics.get("skewness")),
        "kurtosis": _num(metrics.get("kurtosis")),
        "beta_to_benchmark": _num(metrics.get("beta_to_benchmark")),
        "treynor_ratio": _num(metrics.get("treynor_ratio")),
        "jensen_alpha": _pct(metrics.get("jensen_alpha")),
    }
    return _key_value_table(selected)


def _formula_table() -> str:
    rows = ["| Formula | Expression |", "| --- | --- |"]
    for name, formula in FORMULAS:
        rows.append(f"| {name} | $$ {formula} $$ |")
    return "\n".join(rows)


def _bibliography_table(report_model: dict[str, Any]) -> str:
    rows = [
        "| Metodo | Formula | Fuente | Estado validacion | Implementacion | Limitacion |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report_model.get("bibliography", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            "| {method} | {formula} | {source} | {validation} | {implementation} | {limit} |".format(
                method=item.get("method", item.get("block", "N/A")),
                formula=item.get("formula", "See methods review"),
                source=item.get(
                    "local_source", "docs/research/professional_quant_methods_review.md"
                ),
                validation=item.get(
                    "page_level_validation", item.get("validation_status", "PENDING")
                ),
                implementation="quant_platform.research / quant_platform.reporting",
                limit="See documented limitations; no page numbers invented.",
            )
        )
    return "\n".join(rows)


def _methods_review_summary() -> dict[str, Any]:
    path = Path("docs/research/professional_quant_methods_review.md")
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "exists": True,
        "mentions_pending_page_level_validation": "PENDING_PAGE_LEVEL_VALIDATION" in text,
    }


def _inline_markdown(text: str) -> str:
    """Convert inline markdown syntax (bold, code, links, LaTeX math) to HTML."""
    s = html.escape(text)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    s = re.sub(r'\$\$(.+?)\$\$', r'\\[\1\\]', s)
    return s


def _markdown_to_basic_html(markdown: str) -> str:
    html_lines: list[str] = []
    in_table = False
    in_list = False
    lines = markdown.splitlines()
    for line in lines:
        stripped = line.strip()
        if line.startswith("| ") and line.endswith(" |") and "---" not in line:
            cells = [_inline_markdown(cell.strip()) for cell in stripped.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                html_lines.append(
                    "<table><tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"
                )
                in_table = True
            else:
                html_lines.append(
                    "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"
                )
            continue
        if in_table:
            html_lines.append("</table>")
            in_table = False
        if stripped.startswith("- "):
            content = _inline_markdown(stripped[2:])
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{content}</li>")
            continue
        if stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{_inline_markdown(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{_inline_markdown(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{_inline_markdown(stripped[4:])}</h3>")
        elif not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{_inline_markdown(line)}</p>")
    if in_table:
        html_lines.append("</table>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def _embed_figures_inline(
    body: str, figure_paths: dict[str, str], asset_id: str
) -> str:
    """Replace figure text links with inline embedded Plotly figures."""
    for name, fpath in figure_paths.items():
        fpath_obj = Path(fpath)
        if not fpath_obj.exists():
            continue
        raw = fpath_obj.read_text(encoding="utf-8")
        div_match = re.search(
            r'(<div\s+id="[^"]*"\s+class="plotly-graph-div"[^>]*>)\s*</div>',
            raw,
        )
        script_match = re.search(
            r'(<script>\s*window\.PLOTLYENV.*?Plotly\.newPlot\(\s*"[^"]*".*?</script>)',
            raw,
            re.DOTALL,
        )
        if not div_match or not script_match:
            continue
        plotly_div = div_match.group(1) + "</div>"
        plotly_script = script_match.group(1)
        figure_html = (
            f'<div class="figure-box">'
            f'<h4 style="color:#f2d27a;margin:0 0 8px;">{html.escape(name.replace("_", " ").title())}</h4>'
            f'{plotly_div}{plotly_script}</div>'
        )
        filename = fpath_obj.name
        body = re.sub(
            rf'<li><a\s+href="[^"]*{re.escape(filename)}">[^<]*</a></li>',
            lambda _m: figure_html,
            body,
        )
    return body


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    try:
        clean = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if clean != clean:
        return None
    return clean


def _pct(value: object) -> str:
    clean = _float(value)
    return "N/A" if clean is None else f"{clean:.2%}"


def _num(value: object) -> str:
    clean = _float(value)
    return "N/A" if clean is None else f"{clean:,.4f}"
