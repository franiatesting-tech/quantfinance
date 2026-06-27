# Quant Platform

Auditable Quant Finance research platform for equity and crypto experiments. The current implementation includes mathematical conventions, OHLCV quality checks, synthetic data, local dataset registry, returns, drawdown, historical VaR, historical Expected Shortfall, performance metrics, portfolio baselines, transaction costs, vectorized backtesting, temporal validation splits, benchmark suite, VaR exception backtesting, anti-overfitting registry, settings guardrails, read-only real data ingestion, data quality reports, and profile comparison.

## Current State

- Live trading is prohibited.
- Paper trading is not implemented.
- No broker, order, private exchange, account, margin, futures, perpetuals, funding-real, or leverage-real path exists.
- Real data ingestion is read-only and intended for research simulations with fictitious capital.
- Equity/ETF data uses `yfinance` as a public research provider.
- Crypto spot OHLCV uses Binance public `/api/v3/klines` through `requests`.
- Optional keyed read-only providers are available for Alpha Vantage, Polygon, Nasdaq Data Link, and CryptoCompare.
- CoinGecko is a metadata placeholder, not the primary OHLCV source.
- Tests for providers, ingestion, CLI, and the real-data backtest pipeline use mocks and do not require network access.
- Data quality reports summarize coverage, gaps, failed symbols, assumptions, and suitability for demo backtests.
- No predictive ML, deep learning, DRL, CVXPY, Heston, Black-Scholes operative pricing, MLflow, or dashboards are implemented.

## Local Setup

```powershell
py -3 -m pip install -e ".[dev]"
```

Use `py -3` on this Windows environment. The `python` alias may point to the Microsoft Store stub unless configured manually.

## Run Validation

```powershell
py -3 -m pytest
py -3 -m ruff check .
py -3 -c "import collections, csv, pathlib; p=pathlib.Path(r'..\\docs\\audit\\01_traceability_matrix.csv'); rows=list(csv.DictReader(p.open(newline='', encoding='utf-8'))); ids=[r['requirement_id'] for r in rows]; dup=[i for i,c in collections.Counter(ids).items() if c>1]; print(f'rows={len(rows)} duplicate_ids={dup}')"
```

## CLI

```powershell
py -3 -m quant_platform.cli show-settings
py -3 -m quant_platform.cli download-real-data --config configs/universe_etfs_crypto_daily.yaml --start 2020-01-01 --end 2024-12-31 --limit-equity 5 --limit-crypto 3 --dry-run
py -3 -m quant_platform.cli run-backtest-demo --dataset-id real_daily_demo --version v1 --profile conservative
py -3 -m quant_platform.cli compare-profiles --dataset-id real_daily_demo --version v1 --config configs/universe_etfs_crypto_daily.yaml --risk-config configs/risk_profiles.yaml --dry-run
py -3 -m quant_platform.cli list-providers
py -3 -m quant_platform.cli validate-providers
```

The CLI loads safe settings and refuses live-trading scope.

## Structure

```text
configs/                 Static YAML configuration
docs/architecture/       Mathematical and architectural conventions
docs/development/        Iteration notes and Git setup
docs/literature/         Bibliography metadata; no PDFs committed
src/quant_platform/      Python package
tests/unit/              Unit tests
tests/integration/       Integration tests with mocks
tests/fixtures/          Synthetic test fixtures
```

## Environment

`.env.example` defines the environment contract. It contains safe defaults and empty placeholders only.

Current safety defaults:

- `QUANT_PLATFORM_ALLOW_LIVE_TRADING=false`
- `QUANT_PLATFORM_ALLOW_PAPER_TRADING=false`
- `REQUIRE_EXPLICIT_LIVE_TRADING_CONFIRMATION=true`
- `I_UNDERSTAND_LIVE_TRADING_RISK=false`
- `BASE_CURRENCY=USD`
- `INITIAL_CAPITAL=10000`
- `DEFAULT_FREQUENCY=1d`
- `MAX_ALLOWED_LEVERAGE=1.0`

Do not commit `.env`, `.env.*`, raw data, processed data, generated reports, caches, or PDFs without explicit license approval.

Keyed read-only providers load local secrets from `.env` through `python-dotenv`. CLI output reports only `configured: true/false`; it never prints API key values.

Generated real-data artifacts are local only:

- `data/registry/`: CSV datasets, manifests, and data quality reports.
- `reports/generated/`: backtest reports, profile comparison reports, and trial registries.

## Initial Universe And Risk Profiles

- Universe: ETFs, US equities watchlist, Europe/Spain proxies, indices, and crypto majors spot.
- Frequency: daily `1d`.
- Base currency: USD.
- Initial simulated capital: 10000.
- Risk profiles: `conservative` with approximate 15% target max drawdown and `aggressive` with approximate 30% target max drawdown.
- Drawdown targets are evaluation thresholds, not guarantees. The demo backtest does not enforce drawdown control.
- Storage: CSV registry remains the default; DuckDB/Parquet remain future evaluations.

## Reports

- Coverage/data quality report: generated beside the local registry manifest as `data_quality_report.json`; it answers whether data coverage is adequate and what assumptions/warnings apply.
- Backtest report: generated under `reports/generated/`; it reports strategy and benchmark metrics after costs.
- Profile comparison report: generated under `reports/generated/`; it compares conservative and aggressive profiles on the same dataset and benchmark set.
- Provider comparison report: compares coverage/quality metadata across provider reports, not tick-by-tick price equality.

Free public providers can revise data, fail per ticker, impose rate limits, and have licensing constraints. yfinance adjusted-price handling and corporate actions require review before professional use. Current universes can have survivorship bias.

## Mathematical Safety Rules

- Prices must be positive and finite before return calculations.
- Market data must include `available_at >= timestamp` to control look-ahead bias.
- VaR and Expected Shortfall use positive losses, not raw negative returns.
- Equity daily annualization defaults to `252`; crypto daily annualization defaults to `365`.
- Signals computed at `t` are intended to be executed at `t+1`.
- Costs are subtracted from gross returns.
- Human configs express costs in bps; internal cost calculations use decimal rates.
- Backtests require `execution_lag >= 1` to avoid same-row look-ahead.
- Validation splits are chronological; random splits are not allowed for time series.

## Documentation

- User decisions: `docs/development/decisions_needed.md`.
- Git setup: `docs/development/git_setup.md`.
- Iteration 005 notes: `docs/development/iteration_005_real_data_readonly_research.md`.
- Iteration 006 notes: `docs/development/iteration_006_real_data_validation.md`.
- Iteration 007 notes: `docs/development/iteration_007_keyed_readonly_providers.md`.
- Provider architecture: `docs/architecture/real_data_pipeline.md`.
- Universe policy: `docs/architecture/universe_selection.md`.
- Local PDF bibliography metadata: `docs/literature/bibliography_map.md`.
