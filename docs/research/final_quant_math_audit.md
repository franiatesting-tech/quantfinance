# Final Quant Math Audit

Date: 2026-07-01

Scope: final audit of data, returns, CAPM/factor interpretation, risk, portfolio construction, ML validation and decision gates for the institutional quant research platform.

Mandatory language: This is not investment advice. This is a research-only quantitative signal based on historical data, assumptions and model limitations.

## Data

| Topic | Audit result | Risk | Required control | Associated tests |
| --- | --- | --- | --- | --- |
| OHLC integrity | Provider quality checks reject invalid bars; current final universe is aligned but source warnings show invalid bars in rejected/fallback symbols. | Invalid OHLC can contaminate returns and volatility. | Preserve provider warnings, block broker promotion, require licensed point-in-time feed for production. | `tests/unit/data/test_quality.py`, `tests/unit/research/test_asset_dataset.py` |
| Duplicates | Existing data and registry tests check deterministic datasets. | Duplicate timestamps distort returns and exception counts. | Keep timestamp uniqueness in provider ingestion and registry validation. | `tests/unit/data/test_registry.py`, `tests/unit/data/test_ingestion.py` |
| Timestamps | Daily timestamps are UTC-aware in generated study tables. | Calendar mismatches can create look-ahead or missing-day bias. | Align assets by timestamp and document daily frequency. | `tests/unit/research/test_report.py` |
| Adjusted close | Provider-adjusted close is used, but independent split/dividend validation is not present. | Corporate action drift can bias total returns. | Document limitation; future split/dividend audit with licensed data. | Future data-quality tests |
| Missing data | Aligned 10-stock matrix has 2511 observations; failed symbols are disclosed. | Survivorship and substitution risk remain. | Keep fallback evidence and explicit warnings. | `tests/unit/research/test_asset_dataset.py` |
| Fallback symbols | `V` was replaced by `AVGO`. | Universe selection changed; conclusions are not identical to intended universe. | Treat fallback as blocker for broker promotion. | `tests/unit/research/test_institutional_study.py` |
| Survivorship bias | Current universe uses surviving liquid large-cap equities. | Historical performance may overstate investability. | Document point-in-time universe requirement before publication-grade claims. | Documentation-only now |
| Risk-free proxy | `^IRX` failed; zero annual rate used. | Sharpe, Treynor and Jensen alpha are not fully reliable. | Add final blocker and require Treasury/bill curve or fixed daily risk-free series. | `tests/unit/research/test_multiple_testing.py`, `tests/unit/research/test_institutional_study.py` |

## Returns

| Topic | Formula | Audit result | Associated tests |
| --- | --- | --- | --- |
| Simple returns | `R_t = P_t/P_{t-1} - 1` | Correct for current daily equity return construction. | `tests/unit/features/test_returns.py`, `tests/unit/research/test_report.py` |
| Log returns | `r_t = log(P_t/P_{t-1})` | Implemented in modules but not central in current institutional output; should be documented as continuous-compounding diagnostic. | `tests/unit/features/test_returns.py` |
| Cumulative wealth | `V_t = prod_{s<=t}(1+R_s)` | Correct for drawdown and wealth paths when simple returns are finite and greater than `-100%`. | `tests/unit/risk/test_drawdown.py` |
| Annualization | `sigma_ann = std(R_t) sqrt(252)`; arithmetic expected return `mean(R_t) * 252`; CAGR from compounded wealth | Correct if labels separate realized CAGR from arithmetic expected return. | `tests/unit/backtesting/test_metrics.py`, `tests/unit/research/test_portfolio_optimization.py` |
| Daily frequency | Uses 252 trading-day convention. | Correct for US daily equity approximation; not valid for intraday execution. | Existing metrics tests |

## CAPM And Factors

| Topic | Formula / method | Audit result | Required change | Associated tests |
| --- | --- | --- | --- | --- |
| Beta | `beta_i = Cov(R_i,R_m)/Var(R_m)` | Correct market-proxy beta against `^GSPC`. | Label as benchmark proxy, not true market portfolio. | `tests/unit/research/test_report.py` |
| Treynor | `(R_i-r_f)/beta_i` | Conventionally acceptable but weakened by zero risk-free fallback and beta instability. | Keep descriptive only. | Existing report tests |
| Jensen alpha | `R_i - [r_f + beta_i(R_m-r_f)]` | Current value is benchmark-proxy residual, not factor alpha. | Add factor-model scaffold and `FACTOR_DATA_REQUIRED`. | `tests/unit/research/test_factor_models.py` |
| Market proxy | `^GSPC` | Reasonable broad US equity proxy, but not complete market portfolio. | Add factor/fundamental data before alpha claims. | Documentation-only now |
| Fama-French/Carhart alpha | Regression on market, SMB, HML, MOM, RMW, CMA depending on model | Not computed because factor returns are not available. | Implement CSV-based scaffold and fail closed without factors. | `tests/unit/research/test_factor_models.py` |

## Risk

| Topic | Formula / method | Audit result | Required change | Associated tests |
| --- | --- | --- | --- | --- |
| Historical VaR | `VaR_alpha(L)=quantile_alpha(L)`, `L=-R` | Correct descriptive empirical loss threshold. | Add exception backtesting to final institutional study. | `tests/unit/research/test_tail_risk_backtesting.py`, `tests/unit/risk/test_var.py` |
| Historical ES | `ES_alpha=E[L | L >= VaR_alpha]` | Correct descriptive tail mean under empirical convention. | Formal ES backtesting remains future work. | `tests/unit/risk/test_expected_shortfall.py` |
| Parametric VaR/ES | Normal closed-form using mean and volatility | Correct as educational model; normality may understate fat tails. | Keep `PARAMETRIC_EDUCATIONAL_MODEL` status. | `tests/unit/research/test_var_models.py` |
| Monte Carlo VaR/ES | Quantiles of simulated losses | Useful as scenario model if horizon labels are clear. | Ensure horizon conventions in final paper. | `tests/unit/research/test_var_models.py` |
| Kupiec | Unconditional coverage likelihood-ratio test | Needs final institutional exposure. | Implement reusable module and final table. | `tests/unit/research/test_tail_risk_backtesting.py` |
| Christoffersen | Exception independence likelihood-ratio test | Needs final institutional exposure. | Implement reusable module and final table. | `tests/unit/research/test_tail_risk_backtesting.py` |
| ES backtesting | Acerbi-style ES tests | Not implemented. | Document as future because robust ES backtesting needs more design. | Documentation-only now |

## Portfolio

| Topic | Formula / method | Audit result | Required change | Associated tests |
| --- | --- | --- | --- | --- |
| Covariance | `Sigma = cov(R) * 252` | Correct annual covariance convention. | Add shrinkage comparison. | `tests/unit/research/test_covariance_shrinkage.py` |
| Correlation | Pairwise daily return correlation | Correct descriptive diversification metric. | Add rolling/yearly regime stability. | `tests/unit/research/test_portfolio_robustness.py` |
| Shrinkage | `Sigma_delta=(1-delta)Sigma+delta diag(Sigma)`; Ledoit-Wolf if available | Diagonal sensitivity exists; final module must compare sample vs shrinkage. | Implement module with sklearn fallback. | `tests/unit/research/test_covariance_shrinkage.py` |
| Frontier | Long-only grid cloud | Correct approximate grid, but not a perfect continuous frontier. | Label as grid frontier/cloud. | `tests/unit/research/test_portfolio_optimization.py` |
| Max Sharpe | Grid maximization of `(mu_p-r_f)/sigma_p` | Correct approximation but in-sample and affected by risk-free failure. | Treat as descriptive, not allocation advice. | `tests/unit/research/test_portfolio_optimization.py` |
| Min variance | Grid minimization of `w' Sigma w` | Correct approximation. | Compare sample and shrinkage weights. | `tests/unit/research/test_covariance_shrinkage.py` |
| Concentration | HHI and effective holdings | Correct and useful. | Add max-weight audit in final robustness table. | `tests/unit/research/test_portfolio_robustness.py` |
| Effective number of bets | `N_eff=1/sum(w_i^2)` | Correct concentration proxy. | Keep in final paper. | `tests/unit/research/test_portfolio_robustness.py` |
| Turnover | `0.5 * sum(abs(w_new-w_old))` | Needs final execution-cost sensitivity. | Implement turnover and cost model. | `tests/unit/research/test_execution_costs.py` |
| CVaR optimization | `min_w ES_alpha(-w'R)` long-only | Partial empirical grid implementation acceptable as diagnostic. | Mark convex LP as future. | `tests/unit/research/test_portfolio_robustness.py` |

## Machine Learning

| Topic | Audit result | Required change | Associated tests |
| --- | --- | --- | --- |
| Train/test chronology | Existing ML uses chronological walk-forward diagnostics. | Keep no-shuffle controls in final paper. | `tests/unit/research/test_ml_forecasting.py`, `tests/unit/validation/test_splits.py` |
| Walk-forward | Present for stock-level ML diagnostics. | Add final confidence table. | `tests/unit/research/test_model_confidence.py` |
| Feature leakage | No obvious random shuffle in current diagnostics. | Keep future leakage audit for richer features. | Existing ML tests |
| Baseline | Naive baseline comparison exists. | Expose model-vs-baseline status in final study. | `tests/unit/research/test_model_confidence.py` |
| OOS R-squared | Present and strict gates reject negative/weak values. | Keep as necessary but not sufficient. | Existing ML/report tests |
| Information coefficient | Present. | Keep positive IC as gate. | Existing ML/report tests |
| Directional accuracy | Present. | Compare against baseline and minimum edge. | Existing ML/report tests |
| False discovery | Not fully controlled by White/SPA/MCS. | Add PSR/DSR approximation; document White/SPA/MCS future. | `tests/unit/research/test_multiple_testing.py` |
| Model confidence | Not yet final institution-facing. | Implement accepted/weak/rejected/requires-more-tests statuses. | `tests/unit/research/test_model_confidence.py` |

## Decision Engine

| Topic | Audit result | Required change | Associated tests |
| --- | --- | --- | --- |
| Research-only signal | Existing decision engine is research-only and blocks live orders. | Use only allowed vocabulary in final tables. | `tests/unit/research/test_decision_engine.py`, `tests/unit/research/test_institutional_study.py` |
| Limits | Drawdown, VaR/ES and predictive-edge limits exist. | Add final blocker evidence table. | `tests/unit/research/test_institutional_study.py` |
| Gates | Current generated study allows zero live orders. | Preserve fail-closed broker gate. | `tests/unit/research/test_institutional_study.py` |
| No broker order | No broker order is generated. | Keep no broker, no paper trading, no private endpoints. | `tests/unit/ui/test_safety.py`, `tests/unit/test_cli.py` |
| No personalized advice | Paper and UI must avoid buy/sell/recommendation language. | Add tests checking final paper text. | `tests/unit/research/test_final_academic_paper.py` |

## Final Audit Position

The current platform is mathematically sound for descriptive historical research after the existing corrections, but it is not broker-grade. The correct final position is to strengthen diagnostics, make assumptions explicit, generate reproducible paper outputs, and keep every decision as research-only until risk-free data, factor data, robust multiple-testing controls, calibrated execution costs, paper-trading controls and compliance review exist.
