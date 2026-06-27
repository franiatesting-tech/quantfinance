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
  -> metadata.json
  -> UI Academic Reports inspection
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
├── metadata.json
└── figures/*.html
```

Generated outputs are ignored by Git.

## Document Structure

Each report has 23 sections:

1. Portada.
2. Executive Summary.
3. Plain-English Summary.
4. Research Question.
5. Data & Provenance.
6. Data Quality Review.
7. Price Dynamics.
8. Return Construction.
9. Performance Metrics.
10. Risk Metrics.
11. CAPM-Based Metrics.
12. Value at Risk Analysis.
13. Monte Carlo Simulation.
14. Backtesting Analysis.
15. Options Analytics.
16. Comparison Against Benchmark.
17. Statistical Interpretation.
18. Model Performance Assessment.
19. Stock-Specific Conclusions.
20. Limitations.
21. Reproducibility.
22. Mathematical Appendix.
23. Bibliography & Method Traceability.

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

PDF export is intentionally not implemented in Iteration 010. Markdown and HTML are primary outputs. A future PDF layer should detect Pandoc or WeasyPrint and return `PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER` when unavailable.
