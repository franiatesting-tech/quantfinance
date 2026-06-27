# Mathematical Conventions

This document fixes the minimum conventions that all implemented formulas must follow. It is intentionally conservative: ambiguous inputs should fail fast instead of producing silently biased results.

## Price Convention

Market bars use OHLCV columns:

- `open`
- `high`
- `low`
- `close`
- `volume`

Required identifiers and timestamps:

- `asset_id`
- `timestamp`
- `available_at`
- `market_type`
- `currency`
- `source`
- `venue`, optional

Equity returns should use adjusted prices when corporate actions matter. Raw and adjusted prices must not be mixed without documentation.

All timestamps are interpreted as UTC. `available_at` is mandatory for market data and must satisfy:

```text
available_at >= timestamp
```

A feature computed at time `t` may only use rows with `available_at <= t`.

## Return Convention

Simple return:

```text
R_t = P_t / P_{t-1} - 1
```

Log return:

```text
r_t = log(P_t) - log(P_{t-1})
```

The first observation is `NaN` for both simple and log returns because no previous price exists.

Simple cumulative equity curve from periodic simple returns:

```text
V_t = V_0 * prod_{s<=t}(1 + R_s)
```

Excess return:

```text
r^e_t = r_t - r_{f,t}
```

In code, scalar `risk_free_rate` is treated as a periodic rate already aligned to the return frequency.

## Loss Convention

Risk metrics use positive losses:

```text
L_t = -r_{p,t}
```

VaR and Expected Shortfall must consume positive losses. Implementations must not mix raw negative returns with positive loss conventions.

Historical VaR at confidence level `alpha` is the `alpha` quantile of positive losses:

```text
VaR_alpha(L) = quantile_alpha(L)
```

Historical Expected Shortfall is implemented as the empirical mean of losses in the tail:

```text
ES_alpha(L) = mean(L | L >= VaR_alpha(L))
```

This is the finite-sample tail-mean convention used in the initial implementation. More advanced treatment of discontinuous distributions remains `PENDIENTE_VALIDACION`.

## Annualization Convention

Daily equity annualization:

```text
A = 252
```

Daily crypto annualization:

```text
A = 365
```

Any other frequency must pass `periods_per_year` explicitly and document the chosen convention.

## Temporal Convention

Signals and features follow this rule:

```text
feature_t = f(data available up to t)
trade at t+1
PnL_{t+1} = position_t * return_{t+1} - costs_{t+1}
```

It is prohibited to use the close of `t+1` to construct features labeled at `t`.

## Cost Convention

Costs are always subtracted from PnL. Future backtest modules must model, at minimum:

- commission
- spread
- slippage
- funding for crypto only when funding data is available

Gross returns and net returns must be labeled separately.

## NaN and Infinity Convention

Core formulas do not hide invalid data. Inputs are validated and should fail fast for:

- NaN values
- infinite values
- zero or negative prices
- negative volume
- `available_at < timestamp`

Missing-data imputation is not part of the foundation iteration and must be explicit in future data pipelines.

## Bibliographic Anchors

- Expected Shortfall: Acerbi and Tasche, `Acerbi_Tasche_2001_ES_Natural_Alternative_to_VaR.pdf`, pages 4-7; `Acerbi_Tasche_2001_Coherence_Expected_Shortfall.pdf`, pages 5-7 and appendix.
- Sharpe ratio uncertainty: Benhamou, `Benhamou_2018_Sharpe_t_statistic.pdf`, pages 1 and 5-8.
- Roadmap formulas: `01_fase_analisis_requisitos.tex`, `02_fase_diseno.tex`, `03_fase_construccion.tex`, `04_fase_validacion.tex`.
