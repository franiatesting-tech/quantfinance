# Portfolio Baselines

Iteration 002 implements non-optimized long-only baselines in `src/quant_platform/portfolio/baselines.py`.

## Implemented Baselines

| Baseline | Formula or rule | Notes |
| --- | --- | --- |
| Equal weight | `w_i = 1/N` | Required benchmark before any optimizer or model promotion. |
| Inverse volatility | `w_i = (1/sigma_i) / sum_j(1/sigma_j)` | Rejects zero, negative, NaN, and infinite volatility. |
| Buy-and-hold | Fixed shares from initial weights | Returns are generated from portfolio value changes. |
| Momentum weights | Equal-weight top `N` trailing performers | Weights at `t` use only prices at `t` and `t-lookback`; backtests must shift before applying `t+1` returns. |

## Non-Goals

- No mean-variance optimization.
- No CVaR optimization.
- No HRP or risk parity.
- No solver dependency.
- No model selection based on these baselines without out-of-sample validation.

## Validation Rules

- Asset identifiers must be unique.
- Long-only weights must be finite, non-negative, and normalized when supplied.
- Prices must be finite and strictly positive.
- Momentum weights do not use future rows, but execution timing remains a backtesting responsibility.
