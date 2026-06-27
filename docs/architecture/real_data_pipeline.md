# Real Data Pipeline

The Iteration 005 pipeline is read-only and research-only. It downloads or mocks daily market data, validates it, registers local datasets, and runs an offline vectorized backtest with fictitious capital.

## Providers

| Provider | Use | Endpoint/API | Keys | Status |
| --- | --- | --- | --- | --- |
| `yfinance` | Equity, ETF, index daily OHLCV | Public yfinance API | None | Implemented with mocked tests |
| `binance_public` | Crypto spot daily OHLCV | Public `/api/v3/klines` | None | Implemented with mocked tests |
| `coingecko` | Future metadata/universe enrichment | Public metadata APIs | None | Placeholder, no OHLCV primary use |

## Flow

1. Load `configs/universe_etfs_crypto_daily.yaml`.
2. Select equity and crypto symbols with optional limits for quick runs.
3. Download through read-only providers or use mocks in tests.
4. Normalize to canonical OHLCV schema.
5. Run market data quality checks.
6. Register local datasets with coverage metadata.
7. Build close-price matrix and returns.
8. Build target weights for the demo pipeline.
9. Run vectorized backtest with `execution_lag >= 1`.
10. Apply profile-specific costs from `configs/risk_profiles.yaml`.
11. Write generated JSON report under ignored `reports/generated/`.
12. Append trial metadata to the overfitting registry.

## Safety Boundaries

- No orders.
- No brokers.
- No account endpoints.
- No private exchange endpoints.
- No API keys required.
- No futures, perpetuals, margin, funding-real, or leverage-real logic.
- No paper trading.
- No live trading.

## Failure Handling

Provider failures are captured per symbol. A bad ticker does not fail the whole universe unless no usable data remains. Coverage metadata must include successful symbols, failed symbols, row counts, date coverage, source, venue, and market type.
