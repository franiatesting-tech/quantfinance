"""LaTeX PDF report generation for academic stock reports."""
# ruff: noqa: E501

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_platform.reporting.academic_stock_report import (
    SECTION_TITLES,
    _mapping,
    _num,
    _pct,
    _section_conclusion,
    _section_content,
)

LATEX_ENGINE = "pdflatex"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "\\": "\\textbackslash{}",
        "{": "\\{",
        "}": "\\}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _latex_bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def _latex_code(text: str) -> str:
    return f"\\texttt{{{text}}}"


def _sanitize_latex(text: str) -> str:
    return _escape_latex(text)


def _render_markdown_to_latex_inline(text: str) -> str:
    s = str(text)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("{", "\\{").replace("}", "\\}")
    s = s.replace("&", "\\&").replace("%", "\\%")
    s = s.replace("$", "\\$").replace("#", "\\#").replace("_", "\\_")
    s = s.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    s = s.replace("**", "").replace("`", "\\texttt{")
    s = s.replace("$$", "\\[")
    s = s.replace("]]", "\\]")
    return s


def _key_value_latex(values: dict[str, Any]) -> str:
    rows = ["\\begin{tabular}{ll}", "\\toprule"]
    for key, value in values.items():
        rows.append(f"  {_escape_latex(str(key))} & {_escape_latex(str(value))} \\\\")
    rows.append("\\bottomrule\\end{tabular}")
    return "\n".join(rows)


def build_latex_report(report_model: dict[str, Any], pdf_figures: dict[str, str]) -> str:
    """Generate a complete LaTeX document string for one stock academic report."""
    asset_id = str(report_model["asset_id"])
    metrics = _mapping(report_model.get("metrics"))
    conclusions = _mapping(report_model.get("conclusions"))
    stock = _mapping(report_model.get("stock"))

    figure_relpath: dict[str, str] = {}
    for name, fpath in pdf_figures.items():
        p = Path(fpath)
        figure_relpath[name] = f"figures/{p.name}"

    tb = "\\textbullet{}"
    preamble = (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{fancyhdr}\n"
        "\\usepackage{setspace}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{caption}\n"
        "\n"
        "\\definecolor{themegold}{RGB}{242,210,122}\n"
        "\\definecolor{themedark}{RGB}{8,16,24}\n"
        "\n"
        "\\hypersetup{\n"
        "  colorlinks=true,\n"
        "  linkcolor=blue,\n"
        "  urlcolor=blue,\n"
        "}\n"
        "\n"
        f"\\title{{Academic Quant Research Report \\\\ \\small{{Research-Only {tb} Not Investment Advice}}}}\n"
        f"\\author{{Quant Platform {tb} {_escape_latex(asset_id)}}}\n"
        f"\\date{{\\small{{Generated: {_escape_latex(str(report_model.get('generated_at', '')))}}}}}\n"
        "\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\thispagestyle{empty}\n"
        "\\vfill\n"
        "\\begin{center}\n"
        f"\\textcolor{{gray}}{{\\small{{DEMO {tb} SYNTHETIC DATA {tb} NOT REAL MARKET DATA}}}}\n"
        "\\end{center}\n"
        "\\newpage\n"
    )

    sections = []

    sections.append("\\section*{Executive Summary}")
    sections.append(_executive_summary_latex(asset_id, metrics, conclusions, stock))
    sections.append("")

    sections.append("\\section*{Plain-English Summary}")
    sections.append(_plain_summary_latex(asset_id, conclusions))
    sections.append("")

    for title in SECTION_TITLES[4:]:
        sections.extend(_render_latex_section(title, report_model, figure_relpath))

    sections.append("\\end{document}")
    return preamble + "\n".join(sections)


def _executive_summary_latex(
    asset_id: str,
    metrics: dict[str, Any],
    conclusions: dict[str, Any],
    stock: dict[str, Any],
) -> str:
    cum_ret = _pct(metrics.get("final_cumulative_return"))
    ann_vol = _pct(metrics.get("annualized_volatility"))
    max_dd = _pct(metrics.get("max_drawdown"))
    sharpe = _num(metrics.get("sharpe_ratio"))
    sortino = _num(metrics.get("sortino_ratio"))
    beta = _num(metrics.get("beta_to_benchmark"))
    alpha = _pct(metrics.get("jensen_alpha"))
    ml = _mapping(stock.get("ml_forecasting"))
    audit = _mapping(stock.get("predictive_reliability_audit"))
    _accent = "\\'{}"
    per = f"per{_accent}odo"
    max_txt = f"m{_accent}aximo"
    pred = f"predicci{_accent}on"
    text = (
        f"Historical asset profile: para {_escape_latex(asset_id)}, en el {per} analizado, "
        "el rendimiento acumulado fue "
        f"{_escape_latex(str(cum_ret))}, la volatilidad anualizada fue "
        f"{_escape_latex(str(ann_vol))}, el {max_txt} drawdown fue "
        f"{_escape_latex(str(max_dd))}, Sharpe fue {_escape_latex(str(sharpe))}, "
        f"Sortino fue {_escape_latex(str(sortino))}, beta fue "
        f"{_escape_latex(str(beta))} y Jensen alpha fue "
        f"{_escape_latex(str(alpha))}. Estas metricas describen buy-and-hold/riesgo "
        "historico, no alpha del modelo. Predictive model audit: "
        f"rating={_escape_latex(str(audit.get('rating', 'N/A')))}; "
        f"ML status={_escape_latex(str(ml.get('status', 'N/A')))}; "
        f"OOS R2={_escape_latex(_num(ml.get('oos_r_squared')))}; "
        f"IC={_escape_latex(_num(ml.get('information_coefficient')))}; "
        f"DA={_escape_latex(_pct(ml.get('directional_accuracy')))}. "
        f"{_escape_latex(str(audit.get('interpretation', '')))} "
        f"No implica {pred} ni asesoramiento."
    )
    return text


def _plain_summary_latex(asset_id: str, conclusions: dict[str, Any]) -> str:
    ocr = conclusions.get("Overall research conclusion", "")
    text = (
        f"Este documento intenta explicar {_escape_latex(asset_id)} como un informe de laboratorio: "
        "primero mira datos, luego transforma precios en retornos, despu\\'es mide riesgo y finalmente "
        "prueba modelos. La intuici\\'on es similar a revisar el historial m\\'edico de un paciente: "
        "ayuda a entender el pasado, pero no promete el futuro. "
        f"{_escape_latex(str(ocr))}"
    )
    return text


def _render_latex_section(
    title: str,
    report_model: dict[str, Any],
    figure_paths: dict[str, str],
) -> list[str]:
    stock = _mapping(report_model.get("stock"))
    metrics = _mapping(report_model.get("metrics"))
    data_used = _mapping(stock.get("data_used"))
    conclusions = _mapping(report_model.get("conclusions"))
    section_key = title.split(". ", 1)[-1]
    plain, technical, formulas, figure_keys, limitations = _section_content(section_key)

    lines = [f"\\section*{{{_escape_latex(title)}}}"]

    lines.append("\\subsection*{Explicaci\\'on divulgativa}")
    lines.append(_escape_latex(plain))
    lines.append("")

    lines.append("\\subsection*{Explicaci\\'on t\\'ecnica}")
    lines.append(_escape_latex(technical))
    lines.append("")

    if formulas:
        lines.append("\\subsection*{F\\'ormulas}")
        for name, formula in formulas:
            name_safe = _escape_latex(name)
            formula_latex = _convert_math(formula)
            lines.append(f"\\textbf{{{name_safe}:}} \\[{formula_latex}\\]")
        lines.append("")

    lines.append("\\subsection*{Par\\'ametros y datos usados}")
    lines.append(
        _key_value_latex(
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

    if section_key in {"Performance Metrics", "Risk Metrics", "CAPM-Based Metrics"}:
        lines.append(_metrics_latex_table(metrics))
        lines.append("")

    if section_key == "Data & Provenance":
        dq = _key_value_latex(
            {**_mapping(stock.get("data_used")), **_mapping(stock.get("data_quality"))}
        )
        lines.append(dq)
        lines.append("")

    if section_key == "Mathematical Appendix":
        lines.append(_formula_table_latex())
        lines.append("")

    if figure_keys:
        lines.append("\\subsection*{Gr\\'aficas asociadas}")
        for key in figure_keys:
            if key in figure_paths:
                pdf_file = figure_paths[key]
                caption = _escape_latex(key.replace("_", " ").title())
                lines.append(
                    f"\\begin{{figure}}[htbp]\\centering"
                    f"\\includegraphics[width=\\textwidth]{{{pdf_file}}}"
                    f"\\caption{{{caption}}}\\end{{figure}}"
                )
        lines.append("")

    lines.append("\\subsection*{Conclusiones}")
    lines.append(_escape_latex(_section_conclusion(section_key, conclusions)))
    lines.append("")

    lines.append("\\subsection*{Limitaciones}")
    lines.append(_escape_latex(limitations))
    lines.append("")

    return lines


def _convert_math(formula: str) -> str:
    """Convert Markdown-style formula notation to pure LaTeX math.

    More specific replacements must come before less specific ones.
    """
    s = str(formula)
    s = s.replace("max_{s<=t}", "\\max_{s \\leq t}")
    s = s.replace("prod_{s<=t}", "\\prod_{s \\leq t}")
    s = s.replace("exp(-rT)", "e^{-rT}")
    s = s.replace("N(d1)", "\\mathcal{N}(d_1)")
    s = s.replace("N(d2)", "\\mathcal{N}(d_2)")
    s = s.replace("downside_deviation", "\\operatorname{downside\\_deviation}")
    s = s.replace("sigma_ann", "\\sigma_{\\text{ann}}")
    s = s.replace("ES_alpha", "\\operatorname{ES}_{\\alpha}")
    s = s.replace("VaR_alpha", "\\operatorname{VaR}_{\\alpha}")
    s = s.replace("Alpha_i", "\\alpha_i")
    s = s.replace("Beta_i", "\\beta_i")
    s = s.replace("P_{t-1}", "P_{t-1}")
    s = s.replace("R_f", "R_f")
    s = s.replace("R_m", "R_m")
    s = s.replace("R_i", "R_i")
    s = s.replace("R_t", "R_t")
    s = s.replace("S_0", "S_0")
    s = s.replace("S_t", "S_t")
    s = s.replace("P_0", "P_0")
    s = s.replace("P_t", "P_t")
    s = s.replace("W_t", "W_t")
    s = s.replace("V_t", "V_t")
    s = s.replace("V_0", "V_0")
    s = s.replace("Beta", "\\beta")
    s = s.replace("*", "\\cdot ")
    s = s.replace("ln(", "\\ln(")
    s = s.replace("sqrt(", "\\sqrt{")
    s = s.replace("std(", "\\operatorname{std}(")
    s = s.replace("Cov(", "\\operatorname{Cov}(")
    s = s.replace("Var(", "\\operatorname{Var}(")
    s = s.replace("quantile", "\\operatorname{quantile}")
    s = s.replace("mean(", "\\operatorname{mean}(")
    s = s.replace("N(", "\\mathcal{N}(")
    s = s.replace("exp(", "\\exp(")
    s = s.replace(">= ", "\\geq ")
    s = s.replace("E[", "\\mathbb{E}[")
    s = s.replace(" | L", " \\mid L")
    if s.endswith(")") and "\\sqrt{" in s and s.count("{") > s.count("}"):
        s += "}"
    return s


def _metrics_latex_table(metrics: dict[str, Any]) -> str:
    selected = {
        "Annualized Return": _pct(metrics.get("annualized_return")),
        "CAGR": _pct(metrics.get("cagr")),
        "Annualized Volatility": _pct(metrics.get("annualized_volatility")),
        "Sharpe Ratio": _num(metrics.get("sharpe_ratio")),
        "Sortino Ratio": _num(metrics.get("sortino_ratio")),
        "Calmar Ratio": _num(metrics.get("calmar_ratio")),
        "Hit Rate": _pct(metrics.get("hit_rate")),
        "Skewness": _num(metrics.get("skewness")),
        "Kurtosis": _num(metrics.get("kurtosis")),
        "Beta to Benchmark": _num(metrics.get("beta_to_benchmark")),
        "Treynor Ratio": _num(metrics.get("treynor_ratio")),
        "Jensen Alpha": _pct(metrics.get("jensen_alpha")),
    }
    return _key_value_latex(selected)


def _formula_table_latex() -> str:
    from quant_platform.reporting.academic_stock_report import FORMULAS
    rows = ["\\begin{longtable}{ll}", "\\toprule"]
    for name, formula in FORMULAS:
        name_safe = _escape_latex(name)
        formula_latex = _convert_math(formula)
        rows.append(f"  {name_safe} & $\\displaystyle {formula_latex}$ \\\\")
    rows.append("\\bottomrule\\end{longtable}")
    return "\n".join(rows)


def compile_latex_to_pdf(
    tex_path: str | Path,
    engine: str = LATEX_ENGINE,
    runs: int = 2,
) -> dict[str, Any]:
    """Compile a .tex file to PDF using the given LaTeX engine.

    Multiple runs are needed for cross-references and ToC.
    Returns a dict with 'pdf_path', 'success', and 'log'.
    """
    tex_path = Path(tex_path)
    work_dir = tex_path.parent
    stem = tex_path.stem

    log_lines = []
    for i in range(runs):
        try:
            result = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", str(tex_path)],
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=120,
            )
            log_lines.append(f"Run {i+1}: returncode={result.returncode}")
            log_lines.append(result.stdout[-2000:] if result.stdout else "")
            log_lines.append(result.stderr[-2000:] if result.stderr else "")
            if result.returncode != 0:
                pdf_candidate = work_dir / f"{stem}.pdf"
                if pdf_candidate.exists():
                    log_lines.append("PDF exists despite errors - likely recoverable")
                    return {
                        "pdf_path": str(pdf_candidate),
                        "success": True,
                        "log": "\n".join(log_lines),
                        "warnings": True,
                    }
                return {
                    "pdf_path": None,
                    "success": False,
                    "log": "\n".join(log_lines),
                }
        except subprocess.TimeoutExpired:
            log_lines.append(f"Run {i+1}: timed out")
            return {
                "pdf_path": None,
                "success": False,
                "log": "\n".join(log_lines),
            }

    pdf_path = work_dir / f"{stem}.pdf"
    if pdf_path.exists():
        return {
            "pdf_path": str(pdf_path),
            "success": True,
            "log": "\n".join(log_lines),
        }
    return {
        "pdf_path": None,
        "success": False,
        "log": "\n".join(log_lines),
    }


def write_stock_latex_report(
    report_model: dict[str, Any],
    output_dir: str | Path,
    overwrite: bool = False,
    compile_pdf: bool = True,
    include_figures: bool = True,
) -> dict[str, Any]:
    """Generate and optionally compile a LaTeX academic report for one stock.

    Returns metadata with tex_path, pdf_path (if compiled), and figure paths.
    """
    asset_id = str(report_model["asset_id"])
    stock_dir = Path(output_dir) / asset_id
    if stock_dir.exists() and not overwrite:
        existing = list(stock_dir.glob("*"))
        if existing:
            raise FileExistsError(
                f"Output directory already contains files: {stock_dir}. "
                "Use --overwrite to replace."
            )
    stock_dir.mkdir(parents=True, exist_ok=True)

    pdf_figures: dict[str, str] = {}
    if include_figures:
        figures_dir = stock_dir / "figures"
        from quant_platform.reporting.latex_figures import export_stock_figures_to_pdf
        pdf_figures = export_stock_figures_to_pdf(report_model, figures_dir)

    tex_content = build_latex_report(report_model, pdf_figures)
    tex_path = stock_dir / f"{asset_id}_academic_report.tex"
    tex_path.write_text(tex_content, encoding="utf-8")

    result: dict[str, Any] = {
        "asset_id": asset_id,
        "tex_path": str(tex_path),
        "figures": pdf_figures,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }

    if compile_pdf:
        compile_result = compile_latex_to_pdf(tex_path)
        result["compile"] = compile_result
        result["pdf_path"] = compile_result.get("pdf_path")

    return result


def write_all_stock_latex_reports(
    full_report: dict[str, Any],
    output_dir: str | Path,
    overwrite: bool = False,
    compile_pdf: bool = True,
    include_figures: bool = True,
) -> list[dict[str, Any]]:
    """Generate LaTeX reports for every stock in a terminal report."""
    stocks = full_report.get("stocks", {})
    if not isinstance(stocks, dict) or not stocks:
        raise ValueError("Terminal report has no stocks payload.")
    results = []
    for asset_id in sorted(stocks):
        from quant_platform.reporting.academic_stock_report import build_stock_academic_report_model
        model = build_stock_academic_report_model(full_report, str(asset_id))
        results.append(
            write_stock_latex_report(
                model,
                output_dir=output_dir,
                overwrite=overwrite,
                compile_pdf=compile_pdf,
                include_figures=include_figures,
            )
        )
    return results
