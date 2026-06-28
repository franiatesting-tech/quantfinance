"""Export Plotly academic figures to PDF for LaTeX inclusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.io as pio

from quant_platform.reporting.academic_figures import (
    FIGURE_FILENAMES,
    build_academic_stock_figures,
)

PDF_FIGURE_WIDTH = 700
PDF_FIGURE_HEIGHT = 450


def export_stock_figures_to_pdf(
    report_model: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Export all academic stock figures as PDF files for LaTeX inclusion.

    Returns a dict mapping figure key (e.g. 'price_history') to the PDF file path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = build_academic_stock_figures(report_model)
    pdf_paths: dict[str, str] = {}

    for name, fig in figures.items():
        if fig is None:
            continue
        pdf_name = FIGURE_FILENAMES[name].replace(".html", ".pdf")
        pdf_file = output_path / pdf_name
        try:
            pio.write_image(
                fig,
                str(pdf_file),
                format="pdf",
                width=PDF_FIGURE_WIDTH,
                height=PDF_FIGURE_HEIGHT,
            )
            pdf_paths[name] = str(pdf_file)
        except Exception as exc:
            pdf_paths[name] = f"ERROR: {exc}"

    return pdf_paths
