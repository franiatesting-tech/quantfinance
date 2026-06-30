# Final Institutional Package User Guide

This guide explains how to regenerate and inspect the final research-only institutional quant package.

## Safety Scope

- The package is research-only.
- It does not generate broker orders.
- It does not provide investment advice.
- It rejects synthetic data by default for the final package.
- It records provider warnings and model blockers in the final paper.

Mandatory disclaimer:

```text
This is not investment advice. This is a research-only quantitative signal based on historical data, assumptions and model limitations.
```

## Setup

Use Python 3.13 in this Windows workspace:

```powershell
py -3.13 -m pip install -e ".[dev,ui,ml,pdf]"
py -3.13 -m playwright install chromium
```

The PDF renderer tries WeasyPrint first and falls back to Playwright/Pandoc when available. If PDF export fails, the Markdown and HTML paper remain generated and the metadata records the renderer failure.

## Generate The Final Package

```powershell
py -3.13 -m quant_platform.cli build-final-institutional-package --config configs/quant_terminal_10_stocks.yaml --provider yfinance --output-dir reports/generated/final_package --max-table-rows 20
```

Expected top-level outputs:

- `reports/generated/final_package/metadata.json`
- `reports/generated/final_package/terminal_report/10stocks_10y_report.json`
- `reports/generated/final_package/institutional_study/institutional_quant_study.json`
- `reports/generated/final_package/final_paper/final_institutional_quant_finance_paper.md`
- `reports/generated/final_package/final_paper/final_institutional_quant_finance_paper.html`
- `reports/generated/final_package/final_paper/final_institutional_quant_finance_paper.pdf`
- `reports/generated/final_package/reproducibility_manifest.json`

Generated artifacts are intentionally local and ignored by Git.

## Inspect In UI

```powershell
py -3.13 -m quant_platform.cli launch-ui --host localhost --port 8501
```

The Streamlit UI opens as an Institutional Quant Research Terminal. It reads local JSON/CSV/HTML/PDF artifacts and does not run downloads or trading operations automatically.

## Current Generated Warnings

The latest generated package used `provider_yfinance` and produced a PDF. It also recorded:

- `PROVIDER_PARTIAL_SYMBOL_FAILURES`
- `RISK_FREE_PROXY_UNAVAILABLE_USING_ZERO_RATE`

These are material limitations and remain visible in the final paper and decision gates.

## Email Delivery

The CLI does not send email or store SMTP credentials. To email the PDF, attach this local file manually:

```text
reports/generated/final_package/final_paper/final_institutional_quant_finance_paper.pdf
```
