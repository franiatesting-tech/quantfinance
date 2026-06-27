# Real Data Integration Plan

Iteration 005 implements the first read-only real data research layer. It does not implement trading, paper trading, broker connectivity, orders, account access, private endpoints, derivatives, margin, or leverage.

## Phase A - Data Contract

- OHLCV observations use `timestamp`, `open`, `high`, `low`, `close`, `volume`, `asset_id`, `source`, `venue`, `market_type`, `currency`, and `available_at`.
- `available_at >= timestamp` remains mandatory to control look-ahead bias.
- Equity data uses adjusted close when available from yfinance for return calculations while keeping OHLC fields normalized.
- Provider metadata records source, venue, frequency, symbol coverage, failed symbols, retrieval status, and row counts.

## Phase B - Read-Only Providers

- Equity/ETF: `yfinance`, public research data, no API key.
- Crypto spot: Binance public `/api/v3/klines`, no API key.
- CoinGecko: metadata/universe enrichment placeholder, not OHLCV primary source.
- Tests use monkeypatch/mocks and never require network access.
- Failed symbols are recorded and do not fail the full universe.

## Phase C - Validation

- Run quality checks for required columns, positive prices, OHLC consistency, non-negative volume, duplicates, and `available_at >= timestamp`.
- Detect missing values and gaps before feature generation in future iterations.
- Document survivorship bias because the current universe is based on currently visible tickers.
- For crypto, document venue coverage, stablecoin quote assumptions, 24/7 calendar, and delisting limitations.

## Phase D - Registry

- Continue using the existing CSV + JSON manifest registry.
- Keep raw and processed data under ignored local folders.
- Store dataset ID, version, source, frequency, market type, rows, columns, and coverage metadata.
- Never overwrite datasets silently.

## Phase E - Backtest

- The real-data/mock pipeline loads or downloads a dataset, validates quality, extracts close prices, computes returns, builds target weights, runs the vectorized backtester, applies profile-based costs, generates a JSON report, and records a trial.
- Reports are generated under `reports/generated/`, which is ignored by Git.
- The pipeline remains offline research; it is not an execution simulator.

## Open Gates

- Review provider terms before any professional or commercial use.
- Add delisting/inactive asset handling before claiming unbiased universe coverage.
- Add liquidity/capacity checks before using single-name equities beyond watchlists.
- Evaluate DuckDB/Parquet only after CSV storage becomes a bottleneck.
