# Iteration 002 - Data Registry, Pipeline, References

Date: 2026-06-27

## Created

- External reference notes for Qlib, LEAN, QuantLib, PyPortfolioOpt, and Riskfolio-Lib.
- Architecture note for using those repositories as references only.
- Synthetic OHLCV generator with deterministic seeds and canonical quality validation.
- Local CSV dataset registry with JSON manifests and immutable versions by default.
- Foundation report pipeline from market bars to equal-weight returns, performance metrics, drawdown, VaR, and ES.
- Portfolio baseline module for equal weight, inverse volatility, buy-and-hold, and simple momentum weights.
- Basic transaction cost utilities for turnover, commission, half-spread, slippage, and scalar funding drag.

## Tests Added

- `tests/unit/data/test_synthetic.py`
- `tests/unit/data/test_registry.py`
- `tests/integration/test_foundation_pipeline.py`
- `tests/unit/portfolio/test_baselines.py`
- `tests/unit/backtesting/test_costs.py`

## Guardrails Preserved

- No real data download.
- No broker or exchange connections.
- No API keys.
- No optimizer solver dependency.
- No predictive models, deep learning, CVXPY, Heston, advanced Monte Carlo, paper trading, dashboards, or MLflow.
- External repositories are documented as `NO_DEPENDENCIA_ACTUAL`.

## Mathematical Notes

- Synthetic prices are not a market model and are used only for local tests and pipeline validation.
- Foundation risk metrics convert portfolio returns to non-negative realized losses with `max(-R_t, 0)` before calling VaR/ES functions.
- Momentum weights labeled at `t` use trailing prices through `t`; execution timing remains `t+1` in future backtesting modules.
- Transaction costs are always subtracted from gross returns.

## Validation Status

- Unit and integration tests cover deterministic generation, registry round trips, no overwrite by default, baseline weights, buy-and-hold returns, turnover, transaction costs, and the synthetic data -> registry -> report pipeline.
- Final command results are recorded in the root audit notes for this iteration.
