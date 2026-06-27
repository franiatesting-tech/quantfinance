"""Academic stock report model, Markdown/HTML rendering, and file writing."""
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_platform.reporting.academic_conclusions import build_stock_conclusions
from quant_platform.reporting.academic_figures import (
    FIGURE_FILENAMES,
    write_academic_stock_figures,
)
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
    "10. Risk Metrics",
    "11. CAPM-Based Metrics",
    "12. Value at Risk Analysis",
    "13. Monte Carlo Simulation",
    "14. Backtesting Analysis",
    "15. Options Analytics",
    "16. Comparison Against Benchmark",
    "17. Statistical Interpretation",
    "18. Model Performance Assessment",
    "19. Stock-Specific Conclusions",
    "20. Limitations",
    "21. Reproducibility",
    "22. Mathematical Appendix",
    "23. Bibliography & Method Traceability",
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
    ("Historical VaR", "VaR_alpha(L) = quantile_alpha(L), L=max(-R, 0)"),
    ("Expected Shortfall", "ES_alpha = E[L | L >= VaR_alpha]"),
    ("GBM", "S_t = S_0 exp((mu - 0.5 sigma^2)t + sigma W_t)"),
    ("Black-Scholes call", "C = S N(d1) - K exp(-rT) N(d2)"),
    ("Put-call parity", "C - P = S - K exp(-rT)"),
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
    conclusions = build_stock_conclusions(stock)
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
    """Render one stock academic report as standalone HTML."""

    markdown = render_stock_report_markdown(report_model)
    body = _markdown_to_basic_html(markdown)
    asset_id = html.escape(str(report_model["asset_id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{asset_id} Academic Quant Research Report</title>
  <style>
    body {{ background: #081018; color: #e8ecef; font-family: Georgia, 'Times New Roman', serif; margin: 0; line-height: 1.6; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 48px 28px 80px; }}
    h1, h2, h3 {{ color: #f2d27a; font-family: Aptos, Segoe UI, sans-serif; }}
    a {{ color: #3dd6c6; }}
    code, pre {{ background: #121c25; color: #f2f0df; padding: 2px 6px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
    th, td {{ border: 1px solid #2c3a46; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #142231; color: #f2d27a; }}
    .warning {{ border: 1px solid #d66a4a; padding: 14px; border-radius: 12px; background: rgba(214,106,74,0.12); }}
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
    if "html" in format_set:
        html_path = stock_dir / f"{asset_id}_academic_report.html"
        html_path.write_text(render_stock_report_html(report_model), encoding="utf-8")
        outputs["html"] = str(html_path)
    metadata = {
        "report_type": "academic_stock_report_metadata",
        "asset_id": asset_id,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "outputs": outputs,
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
            lines.append(f"- **{name}:** `{formula}`")
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
    if section_key in {"Performance Metrics", "Risk Metrics", "CAPM-Based Metrics"}:
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
            "El precio muestra la evolucion historica; drawdown muestra caidas desde maximos previos.",
            "La dinamica de precio se observa junto a volumen y drawdown para separar tendencia, actividad y riesgo de caida.",
            [("Max drawdown", "DD_t = V_t / max_{s<=t}(V_s) - 1")],
            ["price_history", "volume", "drawdown"],
            common_limit,
        ),
        "Return Construction": (
            "Los retornos convierten precios en cambios comparables dia a dia.",
            "Se calculan retornos simples, log-retornos y retorno acumulado con composicion continua via Euler.",
            FORMULAS[:3],
            ["simple_returns", "log_returns", "returns_distribution", "cumulative_returns"],
            common_limit,
        ),
        "Performance Metrics": (
            "Las metricas resumen rendimiento y consistencia historica.",
            "Se anualizan retornos y volatilidad con 252 sesiones y se reportan ratios ajustados por riesgo.",
            FORMULAS[3:6],
            ["cumulative_returns", "rolling_sharpe"],
            common_limit,
        ),
        "Risk Metrics": (
            "Riesgo no es solo volatilidad; tambien importan drawdown, colas y asimetria.",
            "Se revisan drawdown, skewness, kurtosis, volatilidad rolling y distribucion de retornos.",
            FORMULAS[3:7],
            ["drawdown", "rolling_volatility", "returns_distribution"],
            common_limit,
        ),
        "CAPM-Based Metrics": (
            "CAPM compara el stock con un benchmark de mercado aproximado.",
            "Beta, Treynor y Jensen alpha se calculan contra el benchmark y tasa libre de riesgo usada.",
            FORMULAS[7:10],
            ["rolling_beta"],
            "El benchmark no es el mercado completo y beta puede cambiar por ventana temporal.",
        ),
        "Value at Risk Analysis": (
            "VaR estima un umbral de perdida; ES estima la perdida media en la cola mala.",
            "Se comparan VaR/ES historico, normal parametrico y Monte Carlo con perdidas positivas.",
            FORMULAS[10:12],
            ["var_comparison", "returns_distribution"],
            "VaR no mide todo lo que ocurre mas alla del umbral y depende del modelo.",
        ),
        "Monte Carlo Simulation": (
            "Monte Carlo crea escenarios posibles bajo reglas explicitas, no predicciones.",
            "Se usan bootstrap historico, block bootstrap, normal parametrico y GBM baseline con seed reproducible.",
            [("GBM", FORMULAS[12][1])],
            ["monte_carlo_paths", "monte_carlo_percentiles", "monte_carlo_terminal_distribution"],
            "Cambiar modelo, seed, horizonte o ventana puede cambiar las conclusiones simuladas.",
        ),
        "Backtesting Analysis": (
            "El backtest simula reglas historicas con capital ficticio.",
            "La convencion usa datos disponibles hasta t y ejecucion t+1 para evitar look-ahead.",
            [("Equity curve", "V_t = V_0 prod_{s<=t}(1+R_s)")],
            ["backtest_equity_curves"],
            "No modela fills reales, liquidez intradia ni impacto de mercado profesional.",
        ),
        "Options Analytics": (
            "Las opciones se valoran teoricamente para explicar sensibilidades, no para cotizar mercado real.",
            "Black-Scholes, CRR, paridad put-call y Greeks usan spot, strike ATM, volatilidad historica, maturity y tasa.",
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
        "Risk Metrics": "Risk profile",
        "CAPM-Based Metrics": "CAPM interpretation",
        "Value at Risk Analysis": "Tail-risk interpretation",
        "Monte Carlo Simulation": "Monte Carlo interpretation",
        "Backtesting Analysis": "Backtesting interpretation",
        "Options Analytics": "Options interpretation",
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
        rows.append(f"| {name} | `{formula}` |")
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


def _markdown_to_basic_html(markdown: str) -> str:
    html_lines = []
    in_table = False
    for line in markdown.splitlines():
        if line.startswith("| ") and line.endswith(" |"):
            if "---" in line:
                continue
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
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
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            html_lines.append(f"<p>{html.escape(line)}</p>")
        elif not line.strip():
            html_lines.append("")
        else:
            html_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        html_lines.append("</table>")
    return "\n".join(html_lines)


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
