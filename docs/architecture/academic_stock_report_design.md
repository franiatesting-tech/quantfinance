# Academic Stock Report Design

Iteration 010 adds a document-generation layer on top of the professional quant terminal report. It is research-only and consumes local JSON; it does not download data, call providers, or execute trading.

## Flow

```text
3stocks_10y_report.json
  -> build_stock_academic_report_model(full_report, asset_id)
  -> deterministic conclusions
  -> Plotly academic figures
  -> Markdown report
  -> HTML report
  -> optional PDF report
  -> metadata.json
  -> UI Stock Research Terminal inspection
```

## Inputs

- Base report: `reports/generated/quant_terminal/3stocks_10y_report.json`.
- Required stock payload: `stocks.<asset_id>`.
- Required fields: data provenance, OHLCV summary, price/volume/return/drawdown/rolling series, performance metrics, CAPM metrics, VaR/ES, Monte Carlo, backtesting, options analytics, warnings, and bibliography references.

## Outputs

For each asset:

```text
reports/generated/academic_stock_reports/<asset>/
├── <asset>_academic_report.md
├── <asset>_academic_report.html
├── <asset>_academic_report.pdf (if renderer is available)
├── metadata.json
└── figures/*.html
```

Generated outputs are ignored by Git.

## Document Structure

Each report has 24 sections:

1. Portada.
2. Executive Summary.
3. Plain-English Summary.
4. Research Question.
5. Data & Provenance.
6. Data Quality Review.
7. Price Dynamics.
8. Return Construction.
9. Performance Metrics.
10. CAPM Metrics.
11. Value at Risk Analysis.
12. Monte Carlo Simulation.
13. ML Forecasting Assessment.
14. Backtesting Analysis.
15. Options Analytics.
16. Comparison Against Benchmark.
17. Quantitative Decision Signal.
18. Statistical Interpretation.
19. Model Performance Assessment.
20. Stock-Specific Conclusions.
21. Limitations.
22. Reproducibility.
23. Mathematical Appendix.
24. Bibliography & Method Traceability.

Each section includes a plain-language explanation, technical explanation, parameters/data used, formulas where relevant, figures where relevant, conclusions, and limitations.

## Figures

Generated Plotly HTML figures include price history, volume, simple returns, log returns, cumulative returns, drawdown, rolling volatility, rolling Sharpe, rolling beta, returns distribution, VaR comparison, Monte Carlo paths, Monte Carlo percentiles, Monte Carlo terminal distribution, backtest equity curves, options payoff, and Greeks.

Each figure includes title, axes, source/model annotation, parameters or model note, and a research-only warning where applicable.

## Conclusions

Conclusions are deterministic rules in `academic_conclusions.py`; no LLM is used. The language avoids operational recommendations and uses phrases such as:

- "En el periodo analizado..."
- "Bajo los supuestos del modelo..."
- "Historicamente..."
- "La metrica indica..."
- "No debe interpretarse como..."

## PDF

PDF export is implemented as a best-effort local renderer. The platform first tries WeasyPrint, then Pandoc. If neither is available, report generation still succeeds with Markdown/HTML and metadata records `PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER`.
