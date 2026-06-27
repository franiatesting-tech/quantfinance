# Backtesting Costs

Iteration 002 implemented basic vectorized cost utilities in `src/quant_platform/backtesting/costs.py`. Iteration 003 normalizes the boundary between human configuration and engine internals.

## Bps and Decimal Rates

Human-readable configs use basis points:

```text
1 bps = 0.0001
10 bps = 0.001
```

The internal engine uses decimal rates only. Use:

- `bps_to_rate(bps)` for config ingestion.
- `rate_to_bps(rate)` for reporting/debugging.
- `TransactionCostConfig.from_bps(...)` for constructing engine configs from `commission_bps`, `spread_bps`, `slippage_bps`, and `funding_bps`.

Negative bps and negative rates fail fast.

## Turnover

Portfolio turnover is computed per rebalance row as:

```text
turnover_t = sum_i |w_{i,t} - w_{i,t-1}|
```

The first row is compared to zero weights unless explicit initial weights are supplied.

## Transaction Cost Drag

Transaction cost drag is modeled as:

```text
cost_t = turnover_t * (commission_rate + 0.5 * spread_rate + slippage_rate)
```

`spread_rate` is the full bid-ask spread, so the basic model charges half-spread per unit turnover.

## Net Return

Net simple return is:

```text
net_return_t = gross_return_t - cost_t - funding_rate
```

Funding is a scalar per-period drag in this iteration. Crypto-specific funding series and exchange-level mechanics remain future work.

## Backtester Integration

The vectorized engine computes:

```text
applied_weights_t = target_weights_{t-execution_lag}
gross_return_t = sum_i applied_weight_{i,t} * return_{i,t}
net_return_t = gross_return_t - cost_t - funding_rate
```

`execution_lag` must be at least one bar in Iteration 003.

## Non-Goals

- No fill simulation.
- No order book model.
- No market impact model.
- No broker/exchange integration.
- No live or paper trading.
