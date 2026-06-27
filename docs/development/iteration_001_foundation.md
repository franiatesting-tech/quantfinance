# Iteration 001 - Foundation

Date: 2026-06-27

## Created

- Python package skeleton under `src/quant_platform`.
- Minimal `pyproject.toml` with Python 3.11+, numpy, pandas, pytest, and ruff.
- Static config placeholders for assets, data sources, risk limits, and backtest defaults.
- Mathematical conventions document.
- Data schemas and market data quality validators.
- Returns engine.
- Drawdown, historical VaR, historical Expected Shortfall, and basic performance metrics.
- Unit tests for all implemented formulas and validators.

## Formulas Implemented

- Simple return: `R_t = P_t / P_{t-1} - 1`.
- Log return: `r_t = log(P_t) - log(P_{t-1})`.
- Cumulative return curve: `V_t = V_0 * prod(1 + R_t)`.
- Excess return: `r^e_t = r_t - r_{f,t}`.
- Drawdown: `DD_t = V_t / max_{s<=t}(V_s) - 1`.
- Historical VaR: empirical `alpha` quantile of positive losses.
- Historical ES: mean of losses satisfying `L >= VaR_alpha(L)`.
- Annualized return, annualized volatility, Sharpe, Sortino, Calmar, and hit rate.

## Conventions Fixed

- Market bars require `timestamp` and `available_at`; `available_at >= timestamp`.
- All timestamps are interpreted as UTC.
- Prices must be positive and finite.
- VaR and ES operate on positive losses.
- Equity daily annualization defaults to 252; crypto daily annualization defaults to 365.
- Signals at `t` must be aligned with forward returns at `t+1`.
- Costs must be subtracted from PnL in future backtests.

## Bibliography Consulted

- `Acerbi_Tasche_2001_ES_Natural_Alternative_to_VaR.pdf`: ES as average loss in worst cases and coherent alternative to VaR.
- `Acerbi_Tasche_2001_Coherence_Expected_Shortfall.pdf`: ES coherence, tail mean, discontinuous distributions.
- `Benhamou_2018_Sharpe_t_statistic.pdf`: Sharpe ratio estimation error and relation to Student t-statistic under assumptions.
- Roadmap LaTeX phases and `references.bib`.

## Tests

- `tests/unit/test_import.py`
- `tests/unit/data/test_quality.py`
- `tests/unit/features/test_returns.py`
- `tests/unit/risk/test_drawdown.py`
- `tests/unit/risk/test_var.py`
- `tests/unit/risk/test_expected_shortfall.py`
- `tests/unit/backtesting/test_metrics.py`

## Open Risks

- No Git repository yet.
- No real data provider selected.
- No universe selected beyond synthetic placeholders.
- No dataset versioning implemented yet.
- ES implementation is empirical tail mean only; discontinuous distribution policy remains pending for advanced use.
- VaR backtesting is not implemented yet.
- Backtester, costs engine, benchmarks, and walk-forward validation are not implemented yet.

## Not Implemented

- Predictive models.
- Portfolio optimizers.
- CVaR optimization.
- Heston.
- Monte Carlo advanced simulation.
- Paper trading.
- Broker/exchange/API integrations.
- MLflow.
- Deep learning or DRL.

## Recommended Next Iteration

Implement a simulated market data generator and a small local dataset registry, then add integration tests for `synthetic bars -> quality checks -> returns -> risk metrics` without connecting to external APIs.
