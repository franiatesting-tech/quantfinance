# Iteration 010 Quant Audit Decision Reports

## Summary

This iteration audited and corrected the professional quant terminal before expanding output layers.

## Implemented

- Mathematical audit in `docs/research/quant_math_audit.md`.
- `ml_forecasting.py` with walk-forward validation and baselines.
- `decision_engine.py` with research-only quantitative decision signal.
- Corrected GBM drift, VaR MC horizon, put-call parity with dividends and Pearson kurtosis.
- Academic report section model updated to 24 required sections.
- `pdf_export.py` with WeasyPrint/Pandoc fallback status.
- One-page Streamlit `Stock Research Terminal`.

## Validation

- `py -3 -m pytest`: 253 passed.
- `py -3 -m ruff check .`: OK.

## Limits

- `scikit-learn` is not installed; ML complex models are optional and baselines run by default.
- PDF generation depends on local renderers. If unavailable, HTML remains the printable artifact.
- Parametric options, fixed income, rates, hedging and exposure are educational models.
