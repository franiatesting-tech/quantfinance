# Quant Math Audit

Date: 2026-06-28

Scope: current `src/quant_platform/research`, reporting formulas, CLI report assembly and UI-facing quantitative outputs. This audit blocks new features until critical implementation errors are fixed.

Mandatory loss convention:

```text
L_t = -R_t
```

Losses are represented as positive numbers when returns are negative. Do not clip gains to zero before VaR/ES unless a specific one-sided loss distribution is explicitly documented.

## Audit Table

| Area | Formula | Implementation current | Correcto/incorrecto | Problema | Correccion | Test asociado |
| --- | --- | --- | --- | --- | --- | --- |
| Simple returns | `R_t = P_t/P_{t-1}-1` | `returns.daily_simple_returns()` delegates to feature helper and drops first NaN. | Correcto | Needs explicit alignment tests for OHLCV pivot. | Add return identity tests. | `test_returns.py` planned |
| Log returns | `r_t = ln(P_t/P_{t-1})` | `returns.daily_log_returns()` delegates to feature helper. | Correcto | MC validator treats log returns as bounded by `>-1`, which is incorrect. | Separate simple-return and log-return validators. | `test_monte_carlo.py` planned |
| Cumulative returns | `V_t = V_0 prod(1+R_t)` | Uses `cumulative_returns()`. | Correcto | Some reports mix cumulative return and price-derived total return with shifted start. | Compute stock total return from return product or unshifted price endpoints. | `test_single_asset_metrics.py` planned |
| Continuous compounding | `P_t=P_0 exp(sum r_t)` | Formula documented, log returns present. | Parcial | No explicit reconstruction test. | Add identity test. | `test_returns.py` planned |
| Annualization | `mu_ann = geom/arith convention`, `sigma_ann=sigma_d sqrt(252)` | Metrics use existing backtesting helpers. | Parcial | Need document arithmetic mean vs CAGR. | Label expected arithmetic returns separately from realized CAGR. | `test_portfolio.py` planned |
| Annualized return/CAGR | `CAGR=(V_T/V_0)^(252/n)-1` | Used via `annualized_return()`, `cagr` alias. | Parcial | `single_asset_metrics.total_return` omits first return interval. | Use full price endpoints or product of returns. | `test_single_asset_metrics.py` planned |
| Volatility | `std(R_t) sqrt(252)` | Implemented. | Correcto | Needs zero/low variance tests. | Add edge tests. | `test_single_asset_metrics.py` planned |
| Sharpe | `(mean(R)-rf_p)/std(R) sqrt(252)` | Implemented via metrics helpers. | Correcto | Risk-free basis must be documented as effective annual to periodic. | Document rate basis. | Existing and planned tests |
| Sortino | `mean(excess)/downside_dev sqrt(252)` | Implemented via metrics helpers. | Correcto | Downside threshold convention should be documented. | Add threshold documentation. | Existing and planned tests |
| Calmar | `CAGR/abs(maxDD)` | Used in report. | Correcto | Needs explicit maxDD zero handling. | Add tests. | `test_single_asset_metrics.py` planned |
| Hit rate | `count(R_t>0)/n` | Implemented. | Correcto | Must clarify net/gross returns in backtests. | Document. | Existing metrics tests |
| Beta | `Cov(R_i,R_m)/Var(R_m)` | `single_asset_metrics.beta()`. | Correcto | Needs known-beta golden test. | Add test. | `test_single_asset_metrics.py` planned |
| Treynor | `(R_i-rf)/beta` | Implemented with annual return. | Correcto/convencion | Uses annual realized return; acceptable if documented. | Document annual realized convention. | `test_single_asset_metrics.py` planned |
| Jensen alpha | `R_i-[rf+beta(R_m-rf)]` | Implemented with annualized returns. | Correcto/convencion | CAPM regression alpha not implemented. | Document as annual realized CAPM residual. | `test_single_asset_metrics.py` planned |
| Max drawdown | `V_t/max(V)-1` | Implemented. | Correcto | Drawdown stored negative; decision thresholds must use negative drawdown. | Keep convention explicit. | Existing risk tests |
| Historical VaR | `VaR_alpha=quantile_alpha(L)` | `var_models.py` uses `losses=-returns`. | Correcto | Formula table still says `L=max(-R,0)`. | Fix formula doc/report. | `test_var_models.py` planned |
| Historical ES | `E[L|L>=VaR]` | Delegates to historical ES. | Correcto/convencion | Finite sample convention simplistic. | Document empirical convention. | `test_var_models.py` planned |
| Parametric VaR normal | `-mu+sigma z_alpha` | Implemented. | Correcto | May return negative VaR for strongly positive/low-vol assets; mathematically valid under convention. | Do not clamp; document. | Existing tests |
| Parametric ES normal | `-mu+sigma phi(z)/(1-alpha)` | Implemented. | Correcto | Add closed-form tests. | Add test. | `test_var_models.py` planned |
| Monte Carlo VaR/ES | Quantiles of simulated losses | Present in `compute_var_summary()`. | Incorrecto en report | Report passes 1-year terminal returns into daily VaR summary. | Separate horizon labels and use MC daily simulated returns for daily VaR or label terminal VaR. | `test_report.py` update |
| Skewness/kurtosis | Sample moments | Report uses pandas skew/kurtosis. | Correcto | Pandas kurtosis is excess kurtosis, but conclusions compare to 3. | Either convert to Pearson kurtosis or update wording. | `test_single_asset_metrics.py` planned |
| Downside deviation | `std(min(excess,0))` | In Sortino helper. | Correcto | Needs convention doc. | Document. | Existing metrics tests |
| Covariance/correlation | `cov(R)*252`, `corr(R)` | Implemented. | Correcto | Annual covariance labelled correctly. | Add numeric test. | `test_portfolio.py` planned |
| Expected returns | `mean(R)*252` | Implemented as annualized mean returns. | Correcto/convencion | Not CAGR. | Label arithmetic expected return. | `test_portfolio.py` planned |
| Portfolio stdev | `sqrt(w' Sigma w)` | Optimization computes from portfolio return series. | Correcto | Good for rebalanced portfolio. | Add analytical test. | `test_portfolio_optimization.py` |
| Min variance | `argmin sigma_p` | Grid-search. | Correcto aproximado | Infeasible grids not guarded; reported grid step may differ after coarsening. | Validate feasibility, report effective step. | `test_portfolio_optimization.py` |
| Max Sharpe | `argmax (R_p-rf)/sigma_p` | Grid-search. | Correcto aproximado | Same grid limitations. | Document and test. | `test_portfolio_optimization.py` |
| Efficient frontier | Pareto efficient set | Current `frontier` is sorted grid, not filtered frontier. | Parcial | May include dominated portfolios. | Add Pareto filtering or label as grid cloud. | `test_efficient_frontier.py` planned |
| Capital allocation line | `rf + Sharpe_t * sigma` | Implemented. | Correcto | Negative Sharpe creates negative slope; acceptable if documented. | Add numeric test. | `test_efficient_frontier.py` planned |
| Historical bootstrap MC | Resample returns with replacement | Implemented. | Correcto | Horizon/path validation missing. | Add limits/validation. | `test_monte_carlo.py` planned |
| Block bootstrap MC | Resample contiguous blocks | Implemented. | Correcto | Validate `horizon_days`, `n_paths`, `block_size`. | Add tests. | `test_monte_carlo.py` planned |
| Parametric normal MC | `N(mean,std)` simple returns | Implemented. | Parcial | Can simulate returns below -100%. | Clip? No; better reject only in path summarization or switch to lognormal for prices. | `test_monte_carlo.py` planned |
| GBM MC | `S_t=S_0 exp(sum(log shocks))` | Uses `mu_log - 0.5 sigma^2` as log-return mean. | Incorrecto | Historical log returns already include the `-0.5 sigma^2` term. | Use `mu_log` directly for log-return shocks or estimate arithmetic drift separately. | `test_monte_carlo.py` planned |
| Multivariate MC | Correlated normal returns | Implemented with PSD covariance repair. | Correcto/parametrico | Need seed/shape tests. | Add tests. | `test_portfolio_monte_carlo.py` planned |
| Put-call parity | `C-P=S e^{-qT}-K e^{-rT}` | Function ignores dividend yield. | Incorrecto | Parity gap wrong when `q != 0`. | Add dividend_yield parameter and report wiring. | `test_options.py` planned |
| Black-Scholes | Standard BSM with continuous dividend | Implemented. | Correcto | Need known-value tests. | Add test. | `test_options.py` planned |
| Binomial CRR | Risk-neutral tree | Implemented no dividend. | Parcial | Dividend yield absent. | Add optional dividend yield in probability growth `exp((r-q)dt)`. | `test_options.py` planned |
| Greeks | Delta/Gamma/Vega/Theta/Rho | Implemented annual theta, per-unit vega/rho. | Correcto | Units need documentation. | Add units to output. | `test_options.py` planned |
| Protective put | Stock plus put payoff | Missing. | Incorrecto por omision | Required output absent. | Implement payoff/table. | `test_options.py` planned |
| Backtest buy and hold | Signal at t, execution t+1 | Engine configured `execution_lag=1`. | Parcial | Single-stock section is simple hold curve, not full strategy engine. | Document or use engine consistently. | `test_backtesting_strategies.py` planned |
| MA crossover | Short MA > long MA | Implemented. | Correcto | Need explicit no-look-ahead test. | Add small-path test. | `test_backtesting_strategies.py` planned |
| Momentum 12-1 | 12-month lookback, skip recent month | Implemented daily approximation. | Correcto/convencion | Label as daily 252/21 approximation. | Document. | `test_backtesting_strategies.py` planned |
| Costs | Commission/spread bps | Engine config uses costs. | Correcto | Need reporting of cost assumptions. | Include in report. | Existing engine tests |
| Fixed income price | PV of cash flows | Implemented. | Correcto parametrico | Public functions lack validation; no stub periods. | Validate in public funcs or document level-coupon only. | `test_fixed_income.py` planned |
| Macaulay/modified duration | Weighted PV time; modified = Mac/(1+y/m) | Implemented. | Correcto | Add zero-coupon/par tests. | `test_fixed_income.py` planned |
| Convexity | Discrete convexity | Implemented. | Correcto | Validate `yield_bump>0` for DV01. | Add guard. | `test_fixed_income.py` planned |
| Swap PV | Fixed leg vs simplified floating coupon leg | Implemented. | Parcial | Not market curve valuation. | Label educational; add par-rate simple test. | `test_rates_derivatives.py` planned |
| SOFR futures implied rate | `(100-price)/100` | Implemented. | Correcto educational | Add range docs. | `test_rates_derivatives.py` planned |
| Hedge ratio | `Cov(spot,hedge)/Var(hedge)` | Implemented. | Correcto | Contract sign convention missing. | Add explicit `hedge_side`/signed contract semantics. | `test_hedging.py` planned |
| Exposure EE/EPE/ENE/PFE | Positive/negative path stats | Implemented. | Correcto educational | ENE returned positive magnitude; document. | Add small-path test. | `test_exposure.py` planned |

## Critical Corrections Required Before New Features

1. Replace investment-decision/BUY language with research-only quantitative decision signal.
2. Restore required three-stock config for AAPL/MSFT/NVDA.
3. Fix report section numbering and formula table for VaR loss convention.
4. Fix GBM drift and log-return validation.
5. Fix put-call parity dividend handling and add protective put/scenario outputs.
6. Add tests for corrected formulas before expanding UI/reporting.
