# Iteration 005 - Real Data Read-Only Research

## Scope

Iteration 005 adapts the platform to the user-approved research flow: daily ETFs, equity watchlists, indices, and crypto majors spot using public read-only providers. It remains a research platform with fictitious capital only.

## Decisions Incorporated

- Git local enabled in `quant-platform/`.
- GitHub remote configured as `https://github.com/franiatesting-tech/quantfinance`.
- Initial universe: ETFs plus crypto majors, with Europe/Spain/watchlists added progressively.
- Initial frequency: daily `1d`.
- Base currency: USD.
- Initial simulated capital: 10000.
- Risk profiles: conservative target drawdown 15%, aggressive target drawdown 30%.
- Providers: free/public/read-only.
- Trading real: prohibited.
- Paper trading: not implemented.
- PDFs: local bibliography only, not committed unless permission is confirmed.

## Dependencies Added

- `requests`: public Binance/CoinGecko-style HTTP access with no private endpoints.
- `yfinance`: daily equity/ETF/index research data.
- `python-dotenv`: local environment loading without hardcoding secrets.
- `pandas-market-calendars`: equity calendar validation foundation.

The following remain intentionally excluded: `ccxt`, `duckdb`, `polars`, `pyarrow`, `pydantic`, `cvxpy`, `arch`, `mlflow`, and `torch`.

## Providers

- `YFinanceProvider`: read-only daily equity/ETF/index OHLCV normalization and quality checks.
- `BinancePublicProvider`: read-only public spot OHLCV from `/api/v3/klines`, no API key.
- `CoinGeckoMetadataProvider`: placeholder for future metadata enrichment, not primary OHLCV.

Provider tests use mocks and do not call the network.

## Universe And Risk Profiles

- Universe config: `configs/universe_etfs_crypto_daily.yaml`.
- User decisions config: `configs/user_decisions.yaml`.
- Risk profile config: `configs/risk_profiles.yaml`.
- Risk profile loader: `src/quant_platform/risk/profiles.py`.

Ticker failures are recorded per symbol and do not fail the full pipeline unless no usable data remains.

## CLI

Commands:

```powershell
py -3 -m quant_platform.cli show-settings
py -3 -m quant_platform.cli download-real-data --config configs/universe_etfs_crypto_daily.yaml --start 2020-01-01 --end 2024-12-31 --limit-equity 5 --limit-crypto 3 --dry-run
py -3 -m quant_platform.cli run-backtest-demo --dataset-id real_daily_demo --version v1 --profile conservative
```

The CLI loads safe settings, requires no API keys, supports dry-run behavior, and does not expose any trading command.

## Pipeline

The real-data backtest pipeline loads or downloads a local dataset, validates OHLCV quality, extracts close prices, computes returns, builds target weights, runs the vectorized backtester, applies risk-profile costs, writes a generated JSON report, and registers the trial.

Generated reports are ignored by Git under `reports/generated/`.

## Local PDFs

Detected local PDFs under `../docs`:

- `mgf_mir.pdf`
- `black-scholes.pdf`
- `udea,+10222-40504-1-CE.pdf`
- `quantfinancepdflibro1.pdf`

Expected `modelo.pdf` was not detected by exact filename. PDF metadata is recorded in `docs/literature/bibliography_map.md` with `PENDING_PAGE_LEVEL_EXTRACTION`. No pricing module is implemented from these PDFs in this iteration.

## Validation

- Initial validation observed in this iteration: `129 passed`, `ruff` OK, traceability matrix `rows=47 duplicate_ids=[]`.
- Final validation before commit: `py -3 -m pytest` returned `129 passed`; `py -3 -m ruff check .` returned `All checks passed!`; traceability matrix returned `rows=47 duplicate_ids=[]`.

## Limits And Prohibitions

- No live trading.
- No paper trading.
- No broker or exchange account integration.
- No orders.
- No private endpoints or private keys.
- No Binance account endpoints, margin, futures, perpetuals, funding-real, or leverage-real.
- No Black-Scholes operative pricing, Heston, CVaR optimization, HRP, deep learning, DRL, MLflow, or complex dashboards.

## Open Risks

- yfinance terms, stability, and corporate-action behavior require review before professional use.
- Survivorship bias remains open for current visible tickers.
- Crypto delisting and stablecoin depeg risk are documented but not modeled.
- CSV storage remains sufficient for this iteration but may need DuckDB/Parquet later.
- PDF page-level extraction remains pending.
