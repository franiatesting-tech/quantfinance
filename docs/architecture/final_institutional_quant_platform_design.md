# Final Institutional Quant Platform Design

## Objective

The final platform produces a reproducible institutional research package from public read-only market data. It combines terminal analytics, state-of-the-art audit modules, a professional paper generator and a local read-only UI.

## Pipeline

1. Load `configs/quant_terminal_10_stocks.yaml`.
2. Fetch read-only provider data through `yfinance`.
3. Build the professional quant terminal report.
4. Generate the institutional state-of-the-art study.
5. Write CSV tables and HTML figures.
6. Render the final Markdown and HTML paper.
7. Export PDF with WeasyPrint or Playwright/Pandoc fallback.
8. Write reproducibility metadata and manifests.

## State-Of-Art Modules

- Covariance shrinkage compares sample covariance against Ledoit-Wolf or a regularized fallback.
- Tail-risk backtesting implements Kupiec and Christoffersen diagnostics.
- Multiple-testing controls expose PSR/DSR-style Sharpe deflation.
- Model confidence gates compare ML evidence against naive baselines.
- Portfolio robustness evaluates sensitivity to covariance, return and cost assumptions.
- Factor-model integration fails closed until factor returns are ingested.
- Execution-cost sensitivity uses spread and square-root impact proxies.
- Final research gates block broker promotion when evidence is incomplete.

## UI Design

The UI is a local read-only terminal with these tabs:

- Overview
- Asset Results
- Portfolio
- Tail Risk
- ML Confidence
- Robustness
- Decision Gates
- Final Paper
- Methodology

The UI reads generated metadata and study JSON. It exposes paper download links when PDF metadata exists. It does not execute provider downloads, backtests, orders or email delivery.

## Research Boundaries

- No live trading.
- No paper trading.
- No broker order path.
- No investment advice.
- No synthetic data in the final package unless explicitly allowed for non-final demos.
- Factor alpha and broker promotion are blocked until missing institutional data and controls are implemented.

## Reproducibility Controls

- Source data mode is recorded.
- Provider warnings are preserved.
- Tables and figures are exported separately.
- The final paper includes manual dependency/configuration instructions.
- The reproducibility manifest stores paths and hashes for generated artifacts.
