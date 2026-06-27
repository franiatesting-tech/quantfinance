# QuantConnect LEAN Reference Notes

Status: external reference only, no runtime dependency.

Useful ideas for this platform:

- Event-driven separation between data arrival, signal computation, order simulation, fills, and portfolio accounting.
- Explicit transaction models for fees, spreads, slippage, and buying power constraints.
- Strong guardrails between backtesting and live execution.

Current decision:

- Do not add LEAN or broker/exchange connectors.
- Iteration 002 implements only basic vectorized cost utilities, not an execution engine.
- Future backtesting must preserve the platform rule: signal at `t`, execution at `t+1`.
