# Iteration 003 - Vectorized Backtesting and Validation

Date: 2026-06-27

## Summary

Iteration 003 implements the first minimal vectorized backtesting engine and validation controls required before any predictive model or optimizer is introduced.

Pipeline now covered by tests:

```text
synthetic data -> quality checks -> close prices -> returns -> target weights -> execution lag -> costs -> net returns -> equity curve -> report -> benchmark comparison
```

## Modules Created

- `src/quant_platform/backtesting/engine.py`
- `src/quant_platform/backtesting/benchmarks.py`
- `src/quant_platform/reporting/backtest_report.py`
- `src/quant_platform/validation/splits.py`
- `src/quant_platform/validation/var_backtesting.py`
- `src/quant_platform/validation/overfitting_registry.py`

## Modules Updated

- `src/quant_platform/backtesting/costs.py`: bps/rate conversion and `TransactionCostConfig.from_bps`.
- `configs/backtest.yaml`: explicit `spread_bps` and `funding_bps` placeholders.

## Formulas Implemented

- Bps conversion: `rate = bps * 0.0001`.
- Applied weights: `applied_weights_t = target_weights_{t-execution_lag}`.
- Gross return: `gross_return_t = sum_i applied_weight_{i,t} * return_{i,t}`.
- Turnover: `turnover_t = sum_i |w_{i,t} - w_{i,t-1}|`.
- Cost drag: `cost_t = turnover_t * (commission_rate + 0.5 * spread_rate + slippage_rate)`.
- Net return: `net_return_t = gross_return_t - cost_t - funding_rate`.
- Equity curve: `V_t = V_0 * prod_{s<=t}(1 + net_return_s)`.
- VaR exception: `I_t = 1{L_t > VaR_t}`.
- Expected exception rate: `1 - alpha`.
- Binomial exception probability: `C(n,k) p^k (1-p)^(n-k)`.

## Mathematical Decisions

- `execution_lag < 1` is rejected to prevent same-row look-ahead.
- Initial shifted weights are cash/zero exposure until a prior target weight exists.
- Target weights are validated for no shorts unless `allow_short=True`.
- Leverage is validated with `sum_i |w_i| <= max_leverage`.
- Rows without enough benchmark history receive zero weights.
- VaR backtesting uses positive losses and non-negative VaR forecasts.

## Bibliography and References

- `Benhamou_2018_Sharpe_t_statistic.pdf`: performance ratio caveats remain relevant for reports.
- `BIS_1996_Backtesting_Internal_Models_Market_Risk.pdf`: regulatory anchor for future VaR backtesting evolution.
- `Koshiyama_Firoozye_2019_Backtesting_Overfitting_Covariance_Penalties.pdf`: overfitting control motivation.
- `Rej_2019_Discount_Backtest_PnL.pdf`: motivation for not trusting raw backtest PnL without audit trail.
- `Gort_2022_DRL_Crypto_Backtest_Overfitting.pdf`: crypto/DRL overfitting risk motivation.

## External Repositories Used as Support

- Microsoft Qlib: conceptual layer separation.
- QuantConnect LEAN: conceptual signal/execution/accounting separation.
- QuantLib: strict engine/model separation reminder.
- PyPortfolioOpt: clean weight-output API inspiration.
- Riskfolio-Lib: risk metric taxonomy inspiration.

All remain `NO_DEPENDENCIA_ACTUAL`.

## Tests Added

- `tests/unit/backtesting/test_engine.py`
- `tests/unit/backtesting/test_benchmarks.py`
- `tests/unit/reporting/test_backtest_report.py`
- `tests/unit/validation/test_splits.py`
- `tests/unit/validation/test_var_backtesting.py`
- `tests/unit/validation/test_overfitting_registry.py`
- `tests/integration/test_backtest_pipeline.py`

## Commands Executed

Initial validation:

```powershell
py -3 -m pytest
py -3 -m ruff check .
```

Intermediate validation after implementation:

```powershell
py -3 -m pytest
py -3 -m ruff check .
```

Intermediate result before documentation updates:

- `98 passed in 5.97s`
- `All checks passed!`

Final results are recorded in `../docs/audit/07_iteration_003_audit_notes.md`.

## Risks Mitigated

- Look-ahead bias: explicit execution lag and tests.
- Cost underestimation: bps/rate normalization and integrated transaction cost drag.
- Overfitting audit gap: JSONL trial registry.
- Missing temporal validation: chronological split and walk-forward utilities.
- Missing benchmark comparison: benchmark weight suite and backtest report comparison.

## Risks Open

- No real data, delisting, corporate action, liquidity, or capacity handling.
- No event-driven fills, order book, market impact, or exchange funding series.
- No optimizer, predictive model, walk-forward model training, or model card.
- VaR independence, clustering, and regulatory traffic-light tests remain future work.

## Not Implemented

- Data providers, APIs, brokers, exchanges, keys, trading real, paper trading.
- ML, deep learning, DRL, CVXPY, mean-variance, CVaR optimization, HRP.
- Heston, advanced Monte Carlo, dashboards, MLflow.
