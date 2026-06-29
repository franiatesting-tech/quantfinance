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
    ("Black-Scholes d1", "d1 = [ln(S/K) + (r - q + 0.5 sigma^2)T] / [sigma sqrt(T)]"),
    ("Black-Scholes d2", "d2 = d1 - sigma sqrt(T)"),
    ("Black-Scholes call", "C = S exp(-qT) N(d1) - K exp(-rT) N(d2)"),
    ("Black-Scholes put", "P = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)"),
    ("Put-call parity", "C - P = S exp(-qT) - K exp(-rT)"),
    ("Delta", "Delta_call = exp(-qT) N(d1)"),
    ("Gamma", "Gamma = exp(-qT) phi(d1) / [S sigma sqrt(T)]"),
    ("Vega", "Vega = S exp(-qT) phi(d1) sqrt(T)"),
    ("Ridge Regression", "beta_hat = argmin{||Y - X*beta||^2 + lambda*||beta||^2}"),
    ("Lasso Regression", "beta_hat = argmin{||Y - X*beta||^2 + lambda*||beta||_1}"),
    ("ElasticNet", "beta_hat = argmin{||Y - X*beta||^2 + l1*||beta||_1 + l2*||beta||^2}"),
    ("GARCH(1,1)", "sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1}"),
    ("OOS R-squared", "R^2_OOS = 1 - sum(R_t - hat{R}_t)^2 / sum(R_t - mean(R))^2"),
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
        body = _embed_figures_inline(body, figure_paths, report_model)
    asset_id = html.escape(str(report_model["asset_id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{asset_id} Academic Quant Research Report</title>
  <style>
    @page {{ size: A4; margin: 18mm 16mm 20mm; }}
    body {{ background: #f7f2e8; color: #111820; font-family: Georgia, 'Times New Roman', serif; margin: 0; line-height: 1.55; }}
    main {{ max-width: 1060px; margin: 0 auto; padding: 38px 26px 72px; }}
    h1 {{ color: #101820; font-family: Aptos, Segoe UI, sans-serif; font-size: 34px; letter-spacing: -0.03em; border-bottom: 4px solid #b58b38; padding-bottom: 12px; }}
    h2 {{ color: #101820; font-family: Aptos, Segoe UI, sans-serif; font-size: 23px; margin-top: 34px; break-after: avoid; }}
    h3 {{ color: #5a3d0c; font-family: Aptos, Segoe UI, sans-serif; font-size: 16px; margin-top: 18px; break-after: avoid; }}
    h2:nth-of-type(n+4) {{ break-before: page; }}
    p {{ text-align: justify; }}
    a {{ color: #0f766e; }}
    code, pre {{ background: #efe4cf; color: #111820; padding: 2px 6px; border-radius: 6px; font-family: 'Cascadia Code', 'Consolas', monospace; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 12px; break-inside: avoid; }}
    th, td {{ border: 1px solid #c9b894; padding: 7px 9px; vertical-align: top; }}
    th {{ background: #efe4cf; color: #111820; font-family: Aptos, Segoe UI, sans-serif; }}
    .math-block {{ display: block; margin: 8px 0 10px; padding: 10px 12px; border-left: 4px solid #b58b38; background: #fffaf0; color: #111820; font-family: 'Cambria Math', 'Times New Roman', serif; font-size: 17px; letter-spacing: 0.01em; break-inside: avoid; }}
    .math-inline {{ font-family: 'Cambria Math', 'Times New Roman', serif; color: #111820; }}
    .figure-box {{ background: #fffaf0; border: 1px solid #c9b894; border-radius: 10px; padding: 12px 14px; margin: 16px 0 20px; break-inside: avoid; }}
    .figure-title {{ color: #101820; margin: 0 0 6px; font-family: Aptos, Segoe UI, sans-serif; font-size: 15px; }}
    .figure-caption {{ color: #4a5563; font-size: 12px; margin: 8px 0 0; }}
    .static-chart {{ width: 100%; height: 260px; background: #fffdf8; border-radius: 8px; }}
    .research-action {{ border: 1px solid #b58b38; background: #fff6df; padding: 12px 14px; border-radius: 10px; margin: 14px 0; break-inside: avoid; }}
    .disclaimer {{ border: 1px solid #9f1239; color: #7f1d1d; background: #fff1f2; padding: 10px 12px; border-radius: 10px; font-weight: 700; }}
  </style>
</head>
<body><main><div class="disclaimer">Research-only. This is not investment advice. No broker, no order routing, no position sizing.</div>{body}</main></body>
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
    if section_key == "ML Forecasting Assessment":
        lines.extend([_ml_table(stock), ""])
    if section_key == "Options Analytics":
        lines.extend([_options_table(stock), ""])
    if section_key in {"Quantitative Decision Signal", "Stock-Specific Conclusions"}:
        lines.extend([_research_action_table(conclusions), ""])
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
            "GRAFICA R (Prediction vs Actual): retorno real vs prediccion ML. "
            "GRAFICA S (Residuals): residuos del modelo (real - prediccion). "
            "GRAFICA T (Model Comparison): RMSE, IC y DA por modelo. "
            "GRAFICA U (Feature Importance): importancia de variables. "
            "Metodos: Ridge (L2), Lasso (L1), ElasticNet (L1+L2), Gradient Boosting, Random Forest. "
            "Validacion: walk-forward expanding window con purge gap = horizon_days. "
            "Referencia: Gu, Kelly & Xiu (2020), Rev. Financial Studies, 33(5), 2223-2273. "
            "Interpretacion: si el modelo no supera al baseline naive en RMSE y directional accuracy, "
            "la evidencia predictiva es debil. IC > 0.03 se considera significativo.",
            "Ridge: argmin{||Y-Xbeta||^2 + lambda*||beta||^2}. "
            "Lasso: argmin{||Y-Xbeta||^2 + lambda*||beta||_1}. "
            "ElasticNet: argmin{||Y-Xbeta||^2 + lambda1*||beta||_1 + lambda2*||beta||^2}. "
            "GBM: L(phi) = sum l(y_i, hat{y}_i) + sum Omega(f_k), Omega(f) = gamma*T + 0.5*lambda*||w||^2. "
            "Walk-forward: train=[t_0, t_0+W], test=[t_0+W+purge, t_0+W+purge+test_size]. "
            "OOS R^2 = 1 - sum(R-hat{R})^2 / sum(R-mean(R))^2. "
            "IC = corr(R, hat{R}). DA = prop(sign(hat{R}) = sign(R)). "
            "GARCH(1,1): sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1}. "
            "Bollerslev (1986), J. Econometrics, 31(3), 307-327.",
            [
                ("Ridge", "argmin{||Y-Xbeta||^2 + lambda*||beta||^2}"),
                ("Lasso", "argmin{||Y-Xbeta||^2 + lambda*||beta||_1}"),
                ("ElasticNet", "argmin{||Y-Xbeta||^2 + l1*||beta||_1 + l2*||beta||^2}"),
                ("GBM", "L(phi) = sum l(y_i, hat{y}_i) + sum Omega(f_k)"),
                ("GARCH(1,1)", "sigma^2_t = omega + alpha*eps^2_{t-1} + beta*sigma^2_{t-1}"),
                ("OOS R^2", "1 - sum(R-hat{R})^2 / sum(R-mean(R))^2"),
            ],
            ["ml_prediction_vs_actual", "ml_residuals", "ml_model_comparison", "ml_feature_importance"],
            "Un modelo que no supera al baseline naive debe marcarse como diagnostico debil, no como prediccion fiable. "
            "El status STATUS_OUTPERFORMS requiere RMSE menor Y directional accuracy >= 2% superior al naive.",
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
            FORMULAS[13:20],
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
            "Aqui se agrupan todas las formulas del informe, incluyendo modelos ML y GARCH. "
            "Cada formula incluye definicion de variables, unidades y convenciones.",
            "Variables, unidades, annualization, perdidas positivas y convenciones temporales quedan explicitas. "
            "Modelos ML: Ridge, Lasso, ElasticNet, GARCH(1,1), OOS R^2. "
            "Referencias: Gu et al. (2020), Bollerslev (1986), Engle (1982).",
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


def _ml_table(stock: dict[str, Any]) -> str:
    ml = _mapping(stock.get("ml_forecasting"))
    garch = _mapping(ml.get("garch_forecast"))
    selected = {
        "validation": ml.get("validation"),
        "models_evaluated": ", ".join(str(item) for item in ml.get("models_evaluated", [])),
        "status": ml.get("status"),
        "RMSE": _num(ml.get("rmse")),
        "MAE": _num(ml.get("mae")),
        "OOS R^2": _num(ml.get("oos_r_squared")),
        "Information coefficient": _num(ml.get("information_coefficient")),
        "Directional accuracy": _pct(ml.get("directional_accuracy")),
        "Naive directional accuracy": _pct(ml.get("baseline_directional_accuracy")),
        "GARCH annual volatility": _pct(garch.get("conditional_volatility_annual")),
        "GARCH persistence": _num(garch.get("persistence")),
    }
    return _key_value_table(selected)


def _options_table(stock: dict[str, Any]) -> str:
    options = _mapping(stock.get("options_theoretical_analytics"))
    call_diag = _mapping(options.get("call_diagnostics"))
    selected = {
        "spot": _num(options.get("spot")),
        "strike ATM": _num(options.get("strike")),
        "historical volatility": _pct(options.get("volatility")),
        "BSM call": _num(options.get("black_scholes_call")),
        "BSM put": _num(options.get("black_scholes_put")),
        "CRR call": _num(options.get("binomial_call")),
        "CRR put": _num(options.get("binomial_put")),
        "put-call parity gap": _num(options.get("put_call_parity_gap")),
        "call intrinsic value": _num(call_diag.get("intrinsic_value")),
        "call time value": _num(call_diag.get("time_value")),
        "call breakeven": _num(call_diag.get("breakeven_at_maturity")),
        "risk-neutral exercise probability": _pct(
            call_diag.get("risk_neutral_exercise_probability")
        ),
    }
    return _key_value_table(selected)


def _research_action_table(conclusions: dict[str, Any]) -> str:
    return _key_value_table(
        {
            "Classification": conclusions.get("Research action classification", "N/A"),
            "Parameters": conclusions.get("Research action parameters", "N/A"),
            "Limits": conclusions.get("Research action limits", "N/A"),
            "Compliance": "Research-only; no operational direction, no sizing, no personalized advice.",
        }
    )


def _formula_table() -> str:
    rows = ["| Formula | Expression |", "| --- | --- |"]
    for name, formula in FORMULAS:
        rows.append(f"| {name} | $$ {formula.replace('|', ' given ')} $$ |")
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
    math_blocks: list[str] = []

    def stash_math(match: re.Match[str]) -> str:
        math_blocks.append(_math_html(match.group(1)))
        return f"@@MATH_BLOCK_{len(math_blocks) - 1}@@"

    raw = re.sub(r'\$\$(.+?)\$\$', stash_math, text)
    s = html.escape(raw)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', s)
    for index, block in enumerate(math_blocks):
        s = s.replace(f"@@MATH_BLOCK_{index}@@", block)
    return s


def _math_html(formula: str) -> str:
    """Render a compact LaTeX-like formula as static HTML for WeasyPrint."""

    s = html.escape(str(formula))
    replacements = {
        "sigma": "σ",
        "alpha": "α",
        "beta": "β",
        "lambda": "λ",
        "omega": "ω",
        "eps": "ε",
        "mu": "μ",
        "phi": "φ",
        "sqrt": "√",
        "sum": "∑",
        "prod": "∏",
        "argmin": "arg min",
        "mean": "mean",
        "std": "std",
        "quantile": "quantile",
        "exp": "exp",
        "ln": "ln",
        " &gt;= ": " ≥ ",
        " &lt;= ": " ≤ ",
        "&gt;=": "≥",
        "&lt;=": "≤",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r'([A-Za-zΑ-ω]+)_\{([^}]+)\}', r'\1<sub>\2</sub>', s)
    s = re.sub(r'([A-Za-zΑ-ω]+)_([A-Za-z0-9]+)', r'\1<sub>\2</sub>', s)
    s = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', s)
    s = re.sub(r'\^([A-Za-z0-9]+)', r'<sup>\1</sup>', s)
    s = s.replace("*", " · ")
    return f'<span class="math-block">{s}</span>'


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
    body: str, figure_paths: dict[str, str], report_model: dict[str, Any]
) -> str:
    """Replace figure text links with PDF-safe static SVG figures."""

    stock = _mapping(report_model.get("stock"))
    for name, fpath in figure_paths.items():
        fpath_obj = Path(fpath)
        filename = fpath_obj.name
        figure_html = _static_figure_html(name, stock, filename)
        body = re.sub(
            rf'<li><a\s+href="[^"]*{re.escape(filename)}">[^<]*</a></li>',
            lambda _m: figure_html,
            body,
        )
    return body


def _static_figure_html(name: str, stock: dict[str, Any], filename: str) -> str:
    title = name.replace("_", " ").title()
    svg = _static_svg_for(name, stock)
    caption = _figure_caption(name)
    return (
        '<div class="figure-box">'
        f'<h4 class="figure-title">{html.escape(title)}</h4>'
        f'{svg}'
        f'<p class="figure-caption">{html.escape(caption)} '
        f'Archivo interactivo HTML: figures/{html.escape(filename)}.</p>'
        '</div>'
    )


def _static_svg_for(name: str, stock: dict[str, Any]) -> str:
    line_map = {
        "price_history": ("price_series", "price"),
        "volume": ("volume_series", "volume"),
        "simple_returns": ("simple_returns", "return"),
        "log_returns": ("log_returns", "log_return"),
        "cumulative_returns": ("cumulative_returns", "cumulative_return"),
        "drawdown": ("drawdown_series", "drawdown"),
        "rolling_volatility": ("rolling_volatility", "rolling_volatility"),
        "rolling_sharpe": ("rolling_sharpe", "rolling_sharpe"),
        "rolling_beta": ("rolling_beta", "rolling_beta"),
    }
    if name in line_map:
        series_key, value_key = line_map[name]
        rows = _rows(stock.get(series_key))
        return _line_svg([(name, [row.get(value_key) for row in rows])])
    if name == "returns_distribution":
        values = [row.get("return") for row in _rows(stock.get("simple_returns"))]
        return _hist_svg(values)
    if name == "var_comparison":
        var_payload = _mapping(stock.get("var"))
        bars = []
        for model in ("historical", "parametric_normal", "monte_carlo"):
            payload = _mapping(var_payload.get(model))
            bars.append((f"{model} VaR", payload.get("var")))
            bars.append((f"{model} ES", payload.get("expected_shortfall")))
        return _bar_svg(bars)
    if name == "monte_carlo_paths":
        paths = _rows(_mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal")).get("paths_sample"))
        groups = []
        for path_id in sorted({row.get("path_id") for row in paths})[:8]:
            rows = [row for row in paths if row.get("path_id") == path_id]
            groups.append((f"path {path_id}", [row.get("value") for row in rows]))
        return _line_svg(groups)
    if name == "monte_carlo_percentiles":
        rows = _rows(_mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal")).get("fan_chart"))
        groups = [(p.upper(), [row.get(p) for row in rows]) for p in ("p5", "p25", "p50", "p75", "p95")]
        return _line_svg(groups)
    if name == "monte_carlo_terminal_distribution":
        values = _mapping(_mapping(stock.get("monte_carlo")).get("parametric_normal")).get("terminal_distribution", [])
        return _hist_svg(values if isinstance(values, list) else [])
    if name == "backtest_equity_curves":
        groups = []
        for label, payload in _mapping(stock.get("backtesting_results")).items():
            rows = _rows(_mapping(payload).get("equity_curve"))
            groups.append((str(label), [row.get("equity") for row in rows]))
        return _line_svg(groups)
    if name == "options_payoff":
        options = _mapping(stock.get("options_theoretical_analytics"))
        call_rows = _rows(options.get("payoff_profile"))
        protective_rows = _rows(options.get("protective_put_payoff"))
        return _line_svg(
            [
                ("call payoff", [row.get("payoff") for row in call_rows]),
                ("protective put", [row.get("net_payoff") for row in protective_rows]),
            ]
        )
    if name == "greeks":
        greeks = _mapping(_mapping(stock.get("options_theoretical_analytics")).get("call_greeks"))
        return _bar_svg([(key, greeks.get(key)) for key in ["delta", "gamma", "vega", "theta_annual", "rho"]])
    if name == "ml_prediction_vs_actual":
        rows = _rows(_mapping(stock.get("ml_forecasting")).get("prediction_rows"))
        return _line_svg(
            [
                ("actual", [row.get("actual_return") for row in rows]),
                ("predicted", [row.get("model_prediction") for row in rows]),
            ]
        )
    if name == "ml_residuals":
        rows = _rows(_mapping(stock.get("ml_forecasting")).get("prediction_rows"))
        residuals = [
            _float(row.get("actual_return")) - _float(row.get("model_prediction"))
            for row in rows
            if _float(row.get("actual_return")) is not None
            and _float(row.get("model_prediction")) is not None
        ]
        return _line_svg([("residual", residuals)])
    if name == "ml_model_comparison":
        metrics = _mapping(_mapping(stock.get("ml_forecasting")).get("per_model_metrics"))
        bars = [(label, _mapping(payload).get("directional_accuracy")) for label, payload in metrics.items()]
        return _bar_svg(bars)
    if name == "ml_feature_importance":
        features = list(_mapping(stock.get("ml_forecasting")).get("features", []))[:12]
        return _bar_svg([(str(feature).replace("feature_", ""), len(features) - idx) for idx, feature in enumerate(features)])
    return _placeholder_svg("No static data available")


def _line_svg(groups: list[tuple[str, list[Any]]]) -> str:
    colors = ["#0f766e", "#b45309", "#9f1239", "#1d4ed8", "#4d7c0f", "#7c3aed"]
    clean_groups = []
    all_values = []
    for label, values in groups:
        clean = [_float(value) for value in values]
        clean = [value for value in clean if value is not None]
        if clean:
            clean_groups.append((label, clean))
            all_values.extend(clean)
    if not all_values:
        return _placeholder_svg("No data")
    width, height, pad = 900, 260, 34
    lo, hi = min(all_values), max(all_values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    parts = [_svg_shell(width, height)]
    for idx, (label, values) in enumerate(clean_groups):
        color = colors[idx % len(colors)]
        denom = max(len(values) - 1, 1)
        points = []
        for pos, value in enumerate(values):
            x = pad + (width - 2 * pad) * pos / denom
            y = height - pad - (height - 2 * pad) * (value - lo) / (hi - lo)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}" />')
        parts.append(f'<text x="{pad + idx * 135}" y="20" fill="{color}" font-size="12">{html.escape(str(label))}</text>')
    parts.append(_svg_axes(width, height, pad, lo, hi))
    parts.append("</svg>")
    return "".join(parts)


def _bar_svg(bars: list[tuple[str, Any]]) -> str:
    clean = [(label, _float(value)) for label, value in bars]
    clean = [(label, value) for label, value in clean if value is not None]
    if not clean:
        return _placeholder_svg("No data")
    width, height, pad = 900, 260, 34
    values = [value for _, value in clean]
    lo = min(0.0, min(values))
    hi = max(values)
    if lo == hi:
        hi = lo + 1.0
    bar_width = max(8, (width - 2 * pad) / len(clean) * 0.68)
    parts = [_svg_shell(width, height)]
    zero_y = height - pad - (height - 2 * pad) * (0 - lo) / (hi - lo)
    for idx, (label, value) in enumerate(clean):
        x = pad + idx * (width - 2 * pad) / len(clean) + 4
        y = height - pad - (height - 2 * pad) * (value - lo) / (hi - lo)
        top = min(y, zero_y)
        bar_h = max(abs(zero_y - y), 1.0)
        parts.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="#0f766e" opacity="0.85" />')
        parts.append(f'<text transform="translate({x:.1f},{height - 9}) rotate(-35)" fill="#374151" font-size="9">{html.escape(str(label)[:20])}</text>')
    parts.append(_svg_axes(width, height, pad, lo, hi))
    parts.append("</svg>")
    return "".join(parts)


def _hist_svg(values: list[Any]) -> str:
    clean = [_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return _placeholder_svg("No data")
    bins = 24
    lo, hi = min(clean), max(clean)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    counts = [0] * bins
    for value in clean:
        pos = min(int((value - lo) / (hi - lo) * bins), bins - 1)
        counts[pos] += 1
    return _bar_svg([(str(idx + 1), count) for idx, count in enumerate(counts)])


def _placeholder_svg(text: str) -> str:
    return (
        '<svg class="static-chart" viewBox="0 0 900 260" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="0" y="0" width="900" height="260" fill="#fffdf8" />'
        f'<text x="450" y="132" text-anchor="middle" fill="#6b7280" font-size="18">{html.escape(text)}</text>'
        '</svg>'
    )


def _svg_shell(width: int, height: int) -> str:
    return (
        f'<svg class="static-chart" viewBox="0 0 {width} {height}" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf8" />'
    )


def _svg_axes(width: int, height: int, pad: int, lo: float, hi: float) -> str:
    return (
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#9ca3af" stroke-width="1" />'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#9ca3af" stroke-width="1" />'
        f'<text x="{pad}" y="{pad-8}" fill="#6b7280" font-size="10">max {hi:.4g}</text>'
        f'<text x="{pad}" y="{height-pad+16}" fill="#6b7280" font-size="10">min {lo:.4g}</text>'
    )


def _figure_caption(name: str) -> str:
    captions = {
        "price_history": "Precio de cierre ajustado: muestra tendencia, volatilidad y cambios de regimen.",
        "volume": "Volumen diario: aproxima liquidez y actividad de mercado.",
        "simple_returns": "Retornos simples diarios R_t = P_t/P_{t-1}-1.",
        "log_returns": "Retornos logaritmicos r_t = ln(P_t/P_{t-1}), usados en modelos continuos.",
        "cumulative_returns": "Crecimiento compuesto de capital: traduce retornos diarios a riqueza acumulada.",
        "drawdown": "Caida desde maximos: mide perdida historica real antes de recuperacion.",
        "rolling_volatility": "Volatilidad movil anualizada: identifica periodos de turbulencia y calma.",
        "rolling_sharpe": "Sharpe movil: eficiencia historica riesgo-retorno por ventana.",
        "rolling_beta": "Beta movil frente al benchmark: sensibilidad al mercado y estabilidad del riesgo sistematico.",
        "returns_distribution": "Distribucion empirica de retornos: asimetria y colas gruesas.",
        "var_comparison": "VaR y Expected Shortfall: umbral y perdida media de cola.",
        "monte_carlo_paths": "Trayectorias simuladas: escenarios posibles bajo supuestos del modelo.",
        "monte_carlo_percentiles": "Fan chart: percentiles P5-P95 de la simulacion.",
        "monte_carlo_terminal_distribution": "Distribucion terminal: resultados simulados al final del horizonte.",
        "backtest_equity_curves": "Curvas de capital ficticio: backtest historico sin ejecucion real.",
        "options_payoff": "Payoff teorico de opciones: convexidad y proteccion conceptual.",
        "greeks": "Griegas BSM: sensibilidad a precio, volatilidad, tiempo y tipo.",
        "ml_prediction_vs_actual": "Prediccion walk-forward vs retorno realizado.",
        "ml_residuals": "Residuos ML: error de prediccion real menos estimado.",
        "ml_model_comparison": "Comparacion de modelos ML segun metricas out-of-sample.",
        "ml_feature_importance": "Importancia de variables: factores usados por el modelo predictivo.",
    }
    return captions.get(name, "Figura academica generada para el informe.")


def _rows(value: object) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


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
