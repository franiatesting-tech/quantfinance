# Iteration 009 Professional Quant Terminal

## Objective

Transform the local UI from a status console into a professional Quant Finance terminal for three daily US stocks over a 10-year horizon, while preserving read-only safety and testability.

## Implemented

- Bibliographic gate: `docs/research/professional_quant_methods_review.md`.
- Architecture design: `docs/architecture/professional_quant_terminal_design.md`.
- Config: `configs/quant_terminal_3_stocks.yaml` for AAPL, MSFT, NVDA, SPY, `^IRX`, 10Y daily, USD.
- New package: `quant_platform.research`.
- Single-asset analytics: returns, CAGR-style annualized return, volatility, Sharpe, Sortino, drawdown, beta, Treynor, and Jensen alpha.
- Portfolio analytics: covariance, correlation, equal-weight, inverse-volatility, long-only grid-search min-variance and max-Sharpe portfolios, efficient frontier, and capital allocation line.
- Monte Carlo: correlated-normal portfolio paths with deterministic seed and percentile fan chart data.
- VaR: historical, normal parametric, and Monte Carlo VaR/ES using positive-loss convention.
- Backtesting: buy-and-hold, rebalanced equal-weight, moving-average crossover, and momentum 12-1 with explicit `t+1` execution lag and costs.
- Options: Black-Scholes-Merton, CRR binomial, Greeks, put-call parity, and payoff profiles as `PARAMETRIC_EDUCATIONAL_MODEL`.
- Fixed income/rates: bond price/duration/convexity, simplified swap NPV/DV01, and SOFR futures implied rate as parametrized educational models.
- Hedging/exposure: minimum-variance hedge ratio, contract count estimate, EE/EPE/ENE/PFE from simulated exposure paths.
- CLI: `build-quant-terminal-report` with `--dry-run` and `--offline-synthetic`.
- Generated artifacts are local/ignored: terminal JSON report and frontier CSV.
- UI: new `Quant Terminal` section with tabs A through I for stock detail, portfolio, Monte Carlo, VaR, backtesting, options, fixed income/rates, hedging/exposure, and spreadsheet/methods.

## Safety

- No trading, paper trading, orders, brokers, private endpoints, account data, margin, futures/perpetuals, funding-real, leverage-real, financial recommendations, or automatic advisor logic.
- The UI reads local JSON only and does not launch downloads or simulations.
- Provider errors are sanitized before CLI output.
- Reports serialize no API key values and no private URLs.
- Educational/parametric blocks are labeled with `PARAMETRIC_EDUCATIONAL_MODEL`, `DATA_REQUIRED_FOR_REAL_MARKET_VALUATION`, or `DATA_REQUIRED_FOR_REAL_HEDGE` where applicable.

## Commands

```powershell
py -3 -m quant_platform.cli build-quant-terminal-report --config configs/quant_terminal_3_stocks.yaml --dry-run
py -3 -m quant_platform.cli build-quant-terminal-report --config configs/quant_terminal_3_stocks.yaml --offline-synthetic
py -3 -m quant_platform.cli ui-status
py -3 -m quant_platform.cli launch-ui --dry-run
```

## Validation Snapshot

Focused validation during implementation:

- `py -3 -m pytest tests/unit/research tests/unit/test_cli.py`: `23 passed`.
- `py -3 -m pytest tests/unit/ui/test_charts.py tests/unit/ui/test_report_loader.py`: `8 passed` before adding terminal chart coverage.
- `py -3 -m pytest tests/unit/ui/test_charts.py`: `6 passed` after terminal chart coverage.
- `py -3 -m ruff check src/quant_platform/research src/quant_platform/cli.py src/quant_platform/ui/report_loader.py`: passed.
- `py -3 -m quant_platform.cli build-quant-terminal-report --config configs/quant_terminal_3_stocks.yaml --offline-synthetic`: generated local ignored report and frontier CSV.

Final full-suite validation is recorded in audit notes after completion.

Final validation observed:

- `py -3 -m pytest`: `225 passed in 136.32s`.
- `py -3 -m ruff check .`: passed.
- Traceability matrix: `rows=54 duplicate_ids=[]`.
- `py -3 -m quant_platform.cli ui-status`: read-only UI ready, `no_secrets_exposed=true`.
- `py -3 -m quant_platform.cli launch-ui --dry-run`: returned launch command only.
- `py -3 -m quant_platform.cli build-quant-terminal-report --config configs/quant_terminal_3_stocks.yaml`: completed with `synthetic_fallback` because public provider returned no normalized OHLCV rows in this environment.
