# Professional Quant Terminal Design

Iteration 009 redirige la UI desde consola de estado hacia una terminal profesional de analisis cuantitativo para tres acciones liquidas, manteniendo alcance research-only.

## Arquitectura

```text
Data Providers
  -> Asset Universe Loader
  -> 10Y Daily OHLCV Dataset
  -> Data Quality Engine
  -> Single-Asset Analytics Engine
  -> Multi-Asset Portfolio Engine
  -> Risk Metrics Engine
  -> VaR Engine
  -> Monte Carlo Engine
  -> Strategy Backtesting Engine
  -> Options Analytics Engine
  -> Fixed Income Analytics Engine
  -> Exposure Analytics Engine
  -> Research Report Builder
  -> Professional Quant Terminal UI
```

| Bloque | Input | Output | Formula/Metodo | Validaciones | Graficos | Limitaciones | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Data Providers | Config de tickers y fechas | OHLCV diario read-only | yfinance/read-only provider | No secrets, no endpoints privados, fallback documentado | N/A | Licencias, rate limits, ajustes corporativos | `test_asset_dataset.py` |
| Asset Universe Loader | `configs/quant_terminal_3_stocks.yaml` | Universo AAPL/MSFT/NVDA, SPY, ^IRX | YAML subset local | 3 stocks, benchmark, base currency | Selector UI | Ticker fallido requiere sustituto permitido | `test_asset_dataset.py` |
| 10Y Daily OHLCV Dataset | OHLCV proveedor | Dataset local versionado | Registry CSV/JSON | timestamps ordenados, duplicados, OHLC, available_at | Price/volume | Survivorship bias; adjusted close proveedor | `test_asset_dataset.py` |
| Data Quality Engine | Dataset | Quality summary | checks existentes | coverage, warnings, failed checks | coverage/warnings | No valida licencia profesional | existing data tests |
| Single-Asset Analytics | Precios, benchmark, risk-free | Metricas por stock | returns, beta, Sharpe, Sortino, Treynor, Jensen alpha | alineacion temporal, finite values | price, returns, drawdown, rolling metrics | SPY como proxy de mercado | `test_single_asset_metrics.py` |
| Multi-Asset Portfolio | Retornos 3 stocks | covariance, correlation, portfolios | `w^T mu`, `sqrt(w^T Sigma w)` | PSD/finitos, long-only | heatmaps, weights | Mu/Sigma ruidosos | `test_portfolio.py` |
| Portfolio Optimization | Expected returns, covariance | min-var, max-Sharpe weights | grid search long-only | sum(weights)=1, no leverage | Efficient frontier, CAL | Precision discreta | `test_portfolio_optimization.py` |
| Risk Metrics Engine | Returns/equity curves | drawdown, hit rate, skew, kurtosis | formulas historicas | positive losses for risk | distribution/drawdown | Historico no predice futuro | metric tests |
| VaR Engine | Returns/simulated returns | Historical, parametric, MC VaR/ES | `L=-R`, quantiles, normal | confidence in (0,1), losses positive | loss distribution with thresholds | Normal/MC assumptions | `test_var_models.py` |
| Monte Carlo Engine | Returns, covariance, seed | price/portfolio paths, percentiles | bootstrap, block bootstrap, normal, GBM, Cholesky | seed, PSD fallback | paths/fan/histogram | Simulacion no prediccion | `test_monte_carlo.py` |
| Strategy Backtesting | Price/returns | equity curves, drawdowns, metrics | buy-hold, MA crossover, momentum | signal t, execution t+1, costs | equity/drawdown comparison | No fills/order book | `test_backtesting_strategies.py` |
| Options Analytics | Spot, strike, vol, rate, maturity | BSM/binomial prices, Greeks, parity, payoff | Black-Scholes, CRR | positive inputs, option type | payoff/Greeks | `PARAMETRIC_EDUCATIONAL_MODEL`; no option chain | `test_options.py` |
| Fixed Income Analytics | Bond parameters | price, duration, convexity, sensitivity | PV cash flows, duration, convexity | positive maturity/frequency | sensitivity | `PARAMETRIC_EDUCATIONAL_MODEL` | `test_fixed_income.py` |
| Rates Derivatives | Swap/SOFR parameters | swap NPV, DV01, implied rate | PV fixed/floating, `100-price` | positive notional/time | rates table | `DATA_REQUIRED_FOR_REAL_MARKET_VALUATION` for real pricing | `test_rates_derivatives.py` |
| Hedging | Spot/futures returns, exposure | hedge ratio, contracts | `h*=rho*sigmaS/sigmaF` | aligned returns, positive contract value | hedge summary | `DATA_REQUIRED_FOR_REAL_HEDGE` without futures specs | `test_hedging.py` |
| Exposure Analytics | Simulated exposure paths | EE, EPE, ENE, PFE | means and percentiles | 2D paths, confidence | exposure profile | `PARAMETRIC_EDUCATIONAL_MODEL`; no CSA/netting | `test_exposure.py` |
| Research Report Builder | All module outputs | JSON report and CSV exports | deterministic pipeline | no secrets, no trading, warnings | UI consumes JSON | Local reports ignored by Git | integration tests |
| Professional UI | JSON report | Interactive analytics | Plotly/Streamlit | read-only, no auto network | all terminal charts | Depends on report freshness | UI tests |

## Seguridad

- No trading real, paper trading, ordenes, brokers ni endpoints privados.
- `.env` no se lee ni se muestra desde UI.
- Reports no guardan API keys ni URLs con claves.
- Datos y reports generados quedan bajo rutas ignoradas.
