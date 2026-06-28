"""PDF export helpers for academic stock reports."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

PDF_UNAVAILABLE = "PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER"


def export_html_report_to_pdf(html_path: str | Path, pdf_path: str | Path) -> dict[str, Any]:
    """Export a standalone HTML report to PDF when a renderer is available."""

    source = Path(html_path)
    target = Path(pdf_path)
    if not source.exists():
        raise FileNotFoundError(f"HTML report not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    weasyprint_result = _try_weasyprint(source, target)
    if weasyprint_result["success"]:
        return weasyprint_result
    pandoc_result = _try_pandoc(source, target)
    if pandoc_result["success"]:
        return pandoc_result
    return {
        "success": False,
        "status": PDF_UNAVAILABLE,
        "pdf_path": None,
        "html_printable_path": str(source),
        "attempts": [weasyprint_result, pandoc_result],
    }


def _try_weasyprint(source: Path, target: Path) -> dict[str, Any]:
    # Suppress WeasyPrint logging and stdout/stderr to avoid polluting CLI output
    weasyprint_logger = logging.getLogger("weasyprint")
    previous_level = weasyprint_logger.level
    weasyprint_logger.setLevel(logging.CRITICAL)
    import io
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        from weasyprint import HTML  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - optional renderer discovery.
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        weasyprint_logger.setLevel(previous_level)
        return {"success": False, "renderer": "weasyprint", "error": str(exc)}
    try:
        HTML(filename=str(source)).write_pdf(str(target))
    except Exception as exc:  # noqa: BLE001 - renderer failure should not kill report generation.
        return {"success": False, "renderer": "weasyprint", "error": str(exc)}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        weasyprint_logger.setLevel(previous_level)
    return {
        "success": True,
        "status": "PDF_EXPORTED",
        "renderer": "weasyprint",
        "pdf_path": str(target),
    }


def _try_pandoc(source: Path, target: Path) -> dict[str, Any]:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        return {"success": False, "renderer": "pandoc", "error": "pandoc not found"}
    try:
        result = subprocess.run(
            [pandoc, str(source), "-o", str(target)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - renderer failure should not kill report generation.
        return {"success": False, "renderer": "pandoc", "error": str(exc)}
    if result.returncode != 0:
        return {
            "success": False,
            "renderer": "pandoc",
            "error": (result.stderr or result.stdout)[-500:],
        }
    return {
        "success": True,
        "status": "PDF_EXPORTED",
        "renderer": "pandoc",
        "pdf_path": str(target),
    }
