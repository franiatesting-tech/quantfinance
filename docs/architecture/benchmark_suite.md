# Benchmark Suite

Iteration 003 adds benchmark target-weight generators in `src/quant_platform/backtesting/benchmarks.py`.

## Mandatory Minimum Benchmarks

- Cash.
- Buy-and-hold.
- Equal weight.
- Inverse volatility.
- Simple momentum.

## Implemented Functions

| Function | Input | Rule |
| --- | --- | --- |
| `cash_target_weights` | prices or returns | Zero asset weights on every row. |
| `equal_weight_target_weights` | prices or returns | `w_i = 1/N` on every row. |
| `buy_and_hold_target_weights` | prices | Fixed initial equal-weight shares, weights drift with prices through `t`. |
| `inverse_volatility_target_weights` | returns | Rolling inverse volatility using only the trailing window through `t`. |
| `momentum_target_weights` | prices | Equal-weight top trailing performers through `t`. |

## Temporal Rule

All target weights are labeled at `t`. The vectorized backtester applies `target_weights.shift(1)` when `execution_lag=1`, so no benchmark can use same-row returns for same-row PnL.

Rows without enough trailing history receive zero weights rather than invented estimates.

## Non-Goals

- No market-cap benchmark until market-cap data exists.
- No SPY/Nasdaq/BTC/ETH external benchmark until real data providers are approved.
- No optimizer benchmark, HRP, risk parity, mean-variance, or CVaR optimization.
