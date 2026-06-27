# Riskfolio-Lib Reference Notes

Status: external reference only, no runtime dependency.

Useful ideas for this platform:

- Risk measures and allocation methods must be explicit and comparable against simple benchmarks.
- CVaR, drawdown-aware objectives, and risk parity require careful solver and data validation.
- Reporting should distinguish gross and net performance and include assumptions.

Current decision:

- Do not add Riskfolio-Lib or optimization dependencies.
- Use current risk metrics only as audited scalar diagnostics.
- Add advanced risk allocation only after portfolio constraints, covariance checks, and cost-aware backtests are implemented.
