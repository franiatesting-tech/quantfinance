# Seven Quant Algorithms Replication Study

This document describes the reproducible implementation of the seven algorithms from the supplied introductory PDF:

1. Black-Scholes option pricing
2. Monte Carlo simulation
3. Markowitz portfolio optimization
4. Pairs trading with Ornstein-Uhlenbeck mean reversion
5. Kalman filter state estimation
6. GARCH volatility modeling
7. Machine learning for alpha

## Bibliography Pack

The local pack is expected at:

```text
../quant_finance_papers_pack
```

The study reads `manifest.json` and verifies whether each PDF exists under `papers/`. The SSRN paper `12_deep_order_flow_imbalance_kolm_turiel_westray_2023.pdf` can require browser cookies and is therefore marked as manual download when unavailable.

## Generate The Study

From the `quant-platform` folder:

```powershell
py -3.13 -m quant_platform.cli build-seven-algorithm-study --config configs/quant_terminal_10_stocks.yaml --provider yfinance --output-dir reports/generated/seven_algorithms_study --bibliography-pack ../quant_finance_papers_pack --max-table-rows 30
```

Outputs:

- `reports/generated/seven_algorithms_study/seven_quant_algorithms_study.json`
- `reports/generated/seven_algorithms_study/seven_quant_algorithms_paper.md`
- `reports/generated/seven_algorithms_study/seven_quant_algorithms_paper.html`
- `reports/generated/seven_algorithms_study/seven_quant_algorithms_paper.pdf`
- `reports/generated/seven_algorithms_study/tables/`
- `reports/generated/seven_algorithms_study/figures/`
- `reports/generated/seven_algorithms_study/reproducibility_manifest.json`

## Broker-Readiness Interpretation

The report evaluates broker implications but does not enable broker actions. Each algorithm is blocked for broker promotion until the missing institutional controls exist:

- Broker execution controls and kill switches
- Risk-free and factor data reliability
- Option-chain and implied-volatility calibration
- Short borrow, locate, slippage and market-impact calibration
- Intraday or LOB data for microstructure papers
- Multiple-testing and cost-adjusted ML validation

The generated decisions use `BLOCKED_FOR_BROKER_PROMOTION` and keep `broker_action_allowed=False`.

## Current Generated Warnings

The latest generated study used `provider_yfinance` and recorded:

- `PROVIDER_PARTIAL_SYMBOL_FAILURES`
- `RISK_FREE_PROXY_UNAVAILABLE_USING_ZERO_RATE`

These warnings are preserved in the data-quality gate and broker-readiness table.
