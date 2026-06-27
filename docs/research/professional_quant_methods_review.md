# Professional Quant Methods Review

Fecha: 2026-06-27

Esta revision es el gate bibliografico de Iteration 009. No se inventan paginas, secciones ni validaciones que no esten ya documentadas en los mapas locales. Cuando no existe validacion a nivel pagina/seccion se marca `PENDING_PAGE_LEVEL_VALIDATION`.

## Fuentes Locales Revisadas

- `../docs/audit/00_inventory.md`.
- `../docs/audit/02_literature_map.md`.
- `../references.bib`.
- `quant-platform/docs/literature/bibliography_map.md`.
- `quant-platform/docs/architecture/math_conventions.md`.
- `quant-platform/docs/architecture/portfolio_baselines.md`.
- `quant-platform/docs/architecture/var_backtesting.md`.
- `quant-platform/docs/architecture/backtesting_costs.md`.
- PDFs locales detectados bajo `01_Fundamentos_Quant_Portfolio_CAPM_Roles/`, `02_Probabilidad_Procesos_Estocasticos_No_Arbitraje/`, `03_Pricing_Black_Scholes_Greeks_Calibracion/`, `04_Monte_Carlo_Pricing_Convergencia_LSMC/`, `05_Gestion_Riesgo_VaR_ES_Sharpe_Backtesting_Drawdown/`, `06_Proyecto_Final_Backtest_Model_Validation_Portfolio_Quant/` y `docs/`.

## Tabla De Metodos

| Bloque | Metodo | Formula | Fuente local | Estado validacion | Implementar ahora | Limitaciones |
| --- | --- | --- | --- | --- | --- | --- |
| Single stock returns | Retorno simple | `R_t=P_t/P_{t-1}-1` | `math_conventions.md`; `02_literature_map.md` | TEST_OK existente | Si | Requiere precios positivos y ajuste corporativo documentado. |
| Single stock returns | Log-retorno | `r_t=log(P_t)-log(P_{t-1})` | `math_conventions.md`; `02_literature_map.md` | TEST_OK existente | Si | Diferencias con retornos simples crecen con movimientos grandes. |
| Single stock performance | CAGR | `(V_T/V_0)^(A/N)-1` | Roadmap y convenciones de annualization | PENDING_PAGE_LEVEL_VALIDATION | Si | Sensible a fecha inicial/final y supervivencia del activo. |
| Single stock performance | Volatilidad anualizada | `std(R_t)*sqrt(A)` | `math_conventions.md`; Benhamou para Sharpe/t-stat | TEST_OK parcial | Si | Volatilidad historica no captura colas futuras. |
| Single stock performance | Sharpe | `mean(R-R_f)/std(R-R_f)*sqrt(A)` | `Benhamou_2018_Sharpe_t_statistic.pdf` | Pages 1, 5-8 ya registradas | Si | Incertidumbre estadistica y no normalidad quedan advertidas. |
| Single stock performance | Sortino | `mean(R-T)/downside_deviation*sqrt(A)` | Roadmap; Sortino PDF no local | PENDING_PAGE_LEVEL_VALIDATION | Si | Fuente local de pagina pendiente; implementacion educativa de metrica comun. |
| CAPM analytics | Beta | `Cov(R_i,R_m)/Var(R_m)` | `sharpe1964`; `Sharpe_1990_Nobel_Lecture_Capital_Asset_Prices.pdf` | PENDING_PAGE_LEVEL_VALIDATION | Si | Proxy SPY no es mercado completo. |
| CAPM analytics | Treynor | `(R_p-R_f)/Beta_p` | CAPM/Sharpe referencias locales | PENDING_PAGE_LEVEL_VALIDATION | Si | No definido si beta es cero o inestable. |
| CAPM analytics | Jensen alpha | `R_p-[R_f+Beta_p(R_m-R_f)]` | CAPM/Sharpe referencias locales | PENDING_PAGE_LEVEL_VALIDATION | Si | Interpretacion depende de benchmark y periodo. |
| Risk | Max drawdown | `DD_t=V_t/max_{s<=t}(V_s)-1` | `math_conventions.md`; `02_literature_map.md` | TEST_OK existente | Si | Futuro drawdown puede ser peor que historico. |
| Risk | Historical VaR | `VaR_alpha(L)=quantile_alpha(L)`, `L=-R` | `math_conventions.md`; BIS 1996 | TEST_OK parcial | Si | No mide cola mas alla del umbral. |
| Risk | Historical ES | `E[L | L>=VaR_alpha]` | Acerbi-Tasche PDFs | Pages 4-7; 5-7,17 ya registradas | Si | Tratamiento avanzado de discontinuidades pendiente. |
| Risk | Parametric VaR/ES normal | Normal mean/std sobre retornos | BIS 1996; roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si | Supuesto normal puede fallar con colas gruesas. |
| Risk | Monte Carlo VaR/ES | Cuantiles de retornos simulados | Bormetti 2015; BIS 1996 | PENDING_PAGE_LEVEL_VALIDATION | Si | Depende de modelo y seed. |
| Portfolio | Equal weight | `w_i=1/N` | DeMiguel et al.; `portfolio_baselines.md` | TEST_OK parcial | Si | Benchmark naive, no optimizador. |
| Portfolio | Inverse volatility | `(1/sigma_i)/sum(1/sigma_j)` | `portfolio_baselines.md` | TEST_OK parcial | Si | Ignora correlaciones. |
| Portfolio | Mean-variance performance | `mu_p=w^T mu`, `sigma_p=sqrt(w^T Sigma w)` | Markowitz 1952/1990 local | PENDING_PAGE_LEVEL_VALIDATION | Si | Estimacion de mu/Sigma es ruidosa. |
| Portfolio optimization | Minimum variance | Min `w^T Sigma w`, s.a. `sum(w)=1`, long-only | Markowitz local | PENDING_PAGE_LEVEL_VALIDATION | Si, grid search 3 activos | Precision discreta si no se usa solver. |
| Portfolio optimization | Maximum Sharpe | Max `(w^T mu-r_f)/sqrt(w^T Sigma w)` | Markowitz/Sharpe locales | PENDING_PAGE_LEVEL_VALIDATION | Si, grid search 3 activos | Inestable si retornos esperados son ruidosos. |
| Efficient frontier | Frontier discreta | Portfolios long-only por grid/random | Markowitz local | PENDING_PAGE_LEVEL_VALIDATION | Si | Frontier aproximada sin CVXPY/SciPy. |
| Efficient frontier | Capital Allocation Line | `E[R]=R_f + Sharpe_tangent*sigma` | Sharpe/CAPM locales | PENDING_PAGE_LEVEL_VALIDATION | Si | Supone activo libre de riesgo y tangency portfolio estable. |
| Monte Carlo stock | Bootstrap historico | Sampleo con reemplazo de retornos | Bormetti 2015 local; roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si | No crea nuevos escenarios fuera de la muestra. |
| Monte Carlo stock | Block bootstrap | Sampleo de bloques de retornos | Bormetti 2015 local; roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si | Block size arbitrario. |
| Monte Carlo stock | Normal parametric | `R~N(mu,sigma)` o log-return normal | Bormetti 2015 local; roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si | Normalidad puede subestimar colas. |
| Monte Carlo stock | GBM baseline | `S_t=S_0 exp((mu-0.5 sigma^2)t + sigma W_t)` | Black-Scholes/Merton/Monte Carlo referencias | PENDING_PAGE_LEVEL_VALIDATION | Si | Modelo educativo; volatilidad constante. |
| Monte Carlo portfolio | Multivariante normal | Cholesky de covariance | NumPy/SciPy refs; Markowitz covariance | PENDING_PAGE_LEVEL_VALIDATION | Si | Requiere matriz PSD; fallback documentado. |
| Backtesting | Buy and hold | Pesos iniciales fijos | `benchmark_suite.md`; existing backtester | TEST_OK parcial | Si | No modela fills ni liquidez. |
| Backtesting | Moving average crossover | Senal por medias moviles con execution t+1 | Roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si | Parametrico simple; no ML. |
| Backtesting | Momentum 12-1 | Retorno trailing saltando mes reciente | Moskowitz et al.; roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si | Requiere datos suficientes y no look-ahead. |
| Options | Put-call parity | `C-P=S-Ke^{-rT}` | Black-Scholes/Merton refs; local pricing PDFs | PENDING_PAGE_LEVEL_VALIDATION | Si, teorico | Sin option chain real. |
| Options | Black-Scholes | `C=S N(d1)-K e^{-rT}N(d2)` | `blackscholes1973`; `merton1973`; Majumdar PDF | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | Volatilidad constante, sin microestructura ni smiles. |
| Options | Binomial CRR | Arbol discreto risk-neutral | CRR metadatos no PDF local | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | Fuente local page-level pendiente. |
| Options | Greeks | Derivadas BSM | Black-Scholes/Merton refs | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | Sensibles a supuestos BSM. |
| Fixed income | Bond price/duration/convexity | PV cash flows, duration, convexity | Roadmap; quant finance refs locales genericas | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | No usa bonos reales ni curva real. |
| Rates derivatives | Interest rate swap NPV | PV fixed leg - PV floating leg simplificada | Roadmap; no curva local real | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | DATA_REQUIRED_FOR_REAL_MARKET_VALUATION para pricing profesional. |
| Rates derivatives | SOFR futures implied rate | `rate=(100-price)/100` | Roadmap; no fuente local especifica | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | Convexity adjustment placeholder. |
| Hedging | Cross hedge ratio | `h*=rho*sigma_S/sigma_F` | Grasselli-Hurd hedging ref; roadmap | PENDING_PAGE_LEVEL_VALIDATION | Si, parametrico | DATA_REQUIRED_FOR_REAL_HEDGE si no hay futuros reales. |
| Counterparty exposure | EE/EPE/ENE/PFE | Mean exposure paths; positive/negative parts; percentile | Roadmap; risk architecture | PENDING_PAGE_LEVEL_VALIDATION | Si, PARAMETRIC_EDUCATIONAL_MODEL | Requiere CSA/netting/models reales para uso profesional. |

## Decisiones De Implementacion

- No se anade CVXPY ni SciPy en esta iteracion; la optimizacion long-only de 3 activos usa grid search reproducible y documenta precision limitada.
- Black-Scholes, binomial, swaps, SOFR futures, fixed income, hedging y exposure se implementan como `PARAMETRIC_EDUCATIONAL_MODEL` cuando no existe market data real suficiente.
- Valuaciones que requieran curvas reales, option chains, futures contract specs, CSA/netting o datos contractuales se marcan con `DATA_REQUIRED_FOR_REAL_MARKET_VALUATION` o `DATA_REQUIRED_FOR_REAL_HEDGE`.
- Todas las simulaciones usan seed explicito y reportan que no son predicciones garantizadas.
- No se implementa trading real, paper trading, ordenes, brokers ni endpoints privados.
