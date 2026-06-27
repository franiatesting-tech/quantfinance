# Tech Stack Decisions

Iteration 005 adds only the minimum dependencies needed for read-only real data research. Trading, broker, optimizer, deep learning, experiment-tracking, and large-storage dependencies remain out of scope.

| Dependencia | Estado | Motivo | Riesgo | Decision |
| --- | --- | --- | --- | --- |
| `numpy` | USADA | Calculo numerico vectorial. | Bajo; dependencia core. | Mantener. |
| `pandas` | USADA | Series temporales, DataFrames, CSV registry. | Medio si los datasets crecen mucho. | Mantener. |
| `pytest` | USADA | Tests unitarios e integracion. | Bajo. | Mantener como dev dependency. |
| `ruff` | USADA | Lint local rapido. | Bajo. | Mantener como dev dependency. |
| `requests` | APROBADA_ITERACION_005 | HTTP basico para Binance public API y futuros metadatos CoinGecko read-only. | Rate limits, errores HTTP, terminos de uso. | Usar solo endpoints publicos read-only y tests con mocks. |
| `yfinance` | APROBADA_ITERACION_005 | Datos diarios de ETFs, acciones e indices para research. | No oficial, cobertura/licencia variables. | Usar como research provider, documentar fallos por ticker. |
| `python-dotenv` | APROBADA_ITERACION_005 | Carga local de `.env` sin hardcodear secrets. | Puede ocultar dependencias de entorno si se usa mal. | Permitida; `.env` nunca se versiona. |
| `pandas-market-calendars` | APROBADA_ITERACION_005 | Calendarios bursatiles para validacion equity diaria futura. | Cobertura y reglas de calendario deben revisarse. | Permitida; uso operativo avanzado pendiente. |
| `ccxt` | NO_APROBADA_ITERACION_005 | Normalizacion de exchanges crypto. | Superficie amplia y posible acceso a trading. | No instalar. |
| `duckdb` | EVALUAR_FUTURO | Query local sobre datasets grandes. | Nueva capa de almacenamiento. | Evaluar cuando CSV sea insuficiente. |
| `polars` | EVALUAR_FUTURO | DataFrames rapidos/lazy. | Conversiones pandas/polars y complejidad. | No instalar todavia. |
| `pyarrow` | EVALUAR_FUTURO | Parquet. | Dependencia binaria y compatibilidad Windows. | No instalar todavia. |
| `pydantic` | EVALUAR_FUTURO | Validacion fuerte de configs. | Nueva dependencia y versioning. | No instalar todavia. |
| `cvxpy` | PENDIENTE_FUTURO | Optimizacion convexa futura. | Instalacion/solvers/scope creep. | No instalar. |
| `arch` | PENDIENTE_FUTURO | GARCH/volatilidad avanzada. | Dependencia especializada prematura. | No instalar. |
| `mlflow` | NO_INICIAL | Tracking avanzado. | Sobrecarga operativa temprana. | No instalar. |
| `torch` | NO_INICIAL | Deep learning. | Alto riesgo de overfitting y complejidad. | No instalar. |

## Dependency Gate

Any future dependency must have a specific module, security review, tests, and documentation update before being added to `pyproject.toml`.
