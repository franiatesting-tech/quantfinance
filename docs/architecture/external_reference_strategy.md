# External Reference Strategy

Iteration 002 records selected open-source quant projects as architecture references, not dependencies. Iteration 003 keeps that rule while adding vectorized offline backtesting and validation foundations.

## Reference Projects

| Project | Used for | Dependency status |
| --- | --- | --- |
| Microsoft Qlib | Workflow and dataset-handler separation | `NO_DEPENDENCIA_ACTUAL` |
| QuantConnect LEAN | Event-driven backtesting and transaction model boundaries | `NO_DEPENDENCIA_ACTUAL` |
| QuantLib | Pricing architecture boundaries for later phases | `NO_DEPENDENCIA_ACTUAL` |
| PyPortfolioOpt | Portfolio API shape and baseline/optimizer separation | `NO_DEPENDENCIA_ACTUAL` |
| Riskfolio-Lib | Risk-measure and allocation taxonomy | `NO_DEPENDENCIA_ACTUAL` |

## Rules

- External repositories may inform architecture, naming, and validation checklists.
- External repositories must not be imported or required by `pyproject.toml` without a separate dependency decision.
- Reference notes must describe what is adopted and what is explicitly not adopted.
- The local platform remains limited to `numpy`, `pandas`, `pytest`, and `ruff` in this iteration.
- No live trading, broker/exchange connection, API key handling, real-data download, optimizer solver, MLflow, or deep learning is introduced.

## Current Adoption

- Data foundation: synthetic OHLCV generator plus local CSV registry.
- Pipeline foundation: quality checks, close-price alignment, equal-weight portfolio returns, risk/performance summary.
- Portfolio foundation: non-optimized long-only baselines.
- Backtesting foundation: turnover and basic transaction cost return drag.

## Iteration 003 Adoption

- Qlib remains a conceptual reference for separating data, strategy, backtest, validation, and report layers.
- LEAN remains a conceptual reference for separating signal timing, execution lag, costs, and portfolio accounting.
- QuantLib remains a reminder to keep engines and model assumptions isolated; no pricing engine is added.
- PyPortfolioOpt remains an API-style reference for clean weight outputs; no optimizer or CVXPY dependency is added.
- Riskfolio-Lib remains a taxonomy reference for risk metrics and allocations; no advanced optimizer is added.

No external repository is imported by runtime code.
