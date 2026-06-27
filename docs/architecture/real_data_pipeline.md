# Real Data Pipeline

The pipeline is read-only and research-only. It downloads or mocks daily market data, validates it, registers local datasets, creates coverage/quality reports, and runs offline vectorized backtests with fictitious capital.

## Providers

| Provider | Use | Endpoint/API | Keys | Status |
| --- | --- | --- | --- | --- |
| `yfinance` | Equity, ETF, index daily OHLCV | Public yfinance API | None | Implemented with mocked tests |
| `binance_public` | Crypto spot daily OHLCV | Public `/api/v3/klines` | None | Implemented with mocked tests |
| `alpha_vantage` | Equity/ETF daily adjusted OHLCV | `TIME_SERIES_DAILY_ADJUSTED` | `ALPHA_VANTAGE_API_KEY` | Implemented with mocked tests |
| `polygon` | Equity/ETF daily aggregates | Daily aggregate endpoint | `POLYGON_API_KEY` | Implemented with mocked tests |
| `nasdaq_data_link` | Configurable datasets | Dataset API | `NASDAQ_DATA_LINK_API_KEY` | Minimal provider, dataset code required |
| `cryptocompare` | Crypto spot daily OHLCV | `histoday` | `CRYPTOCOMPARE_API_KEY` | Implemented with mocked tests |
| `coingecko` | Future metadata/universe enrichment | Public metadata APIs | None | Placeholder, no OHLCV primary use |

## Flow

1. Load `configs/universe_etfs_crypto_daily.yaml`.
2. Select equity and crypto symbols with optional limits for quick runs.
3. Download through read-only providers or use mocks in tests.
4. Normalize to canonical OHLCV schema.
5. Run market data quality checks and diagnostic coverage summaries.
6. Register local datasets with coverage metadata and `data_quality_report.json`.
7. Build close-price matrix and returns.
8. Build target weights for the demo pipeline.
9. Run vectorized backtest with `execution_lag >= 1`.
10. Apply profile-specific costs from `configs/risk_profiles.yaml`.
11. Write generated JSON reports under ignored `reports/generated/`.
12. Compare conservative and aggressive profiles on the same dataset when requested.
13. Append trial metadata to the overfitting registry.
14. Inspect local manifests and reports from the read-only UI when requested by the user.

## Reports

| Report | Path | Purpose | Git policy |
| --- | --- | --- | --- |
| Coverage/data quality | `data/registry/<dataset>/<version>/data_quality_report.json` | Symbol coverage, date range, gaps, warnings, provider metadata, backtest suitability | Ignored |
| Backtest | `reports/generated/*_report.json` | Strategy and benchmark metrics after costs | Ignored |
| Profile comparison | `reports/generated/*_profile_comparison.json` | Conservative/aggressive comparison and drawdown target warnings | Ignored |
| UI status | CLI JSON output from `ui-status` | Provider/report/readiness summary with no secrets | Not persisted by default |

## Safety Boundaries

- No orders.
- No brokers.
- No account endpoints.
- No private exchange endpoints.
- No API keys required.
- API keys are optional for keyed read-only providers and must never be logged or committed.
- No futures, perpetuals, margin, funding-real, or leverage-real logic.
- No paper trading.
- No live trading.
- No automatic downloads, backtests, or network smoke tests from the UI.

## Failure Handling

Provider failures are captured per symbol. A bad ticker does not fail the whole universe unless no usable data remains. Coverage metadata must include successful symbols, failed symbols, row counts, date coverage, source, venue, and market type.

## Known Limitations

- yfinance adjusted OHLC treatment must be reviewed against corporate actions before professional use.
- Free providers can be unavailable, rate-limited, revised, or incomplete.
- Equity gaps are warnings until exchange-calendar classification is implemented.
- Crypto daily gaps are failed checks because spot crypto is expected to trade 24/7.
- Pivoting a mixed dataset to a complete close matrix can silently reduce observations; coverage reports must be reviewed alongside backtest reports.
- Drawdown targets in risk profiles are not guarantees.
- Mixing providers in the same backtest can introduce timestamp, adjustment, and survivorship differences. Provider comparison reports should be reviewed first.

## Using Keyed Read-Only Providers

```powershell
py -3 -m quant_platform.cli list-providers
py -3 -m quant_platform.cli validate-providers
py -3 -m quant_platform.cli download-real-data --config configs/universe_etfs_crypto_daily.yaml --start 2024-01-01 --end 2024-03-31 --limit-equity 2 --limit-crypto 2 --equity-provider polygon --crypto-provider cryptocompare
```

The commands never print API key values. `validate-providers --network-smoke` may call configured providers and reports sanitized success/failure status.

## Local UI Inspection

```powershell
py -3 -m quant_platform.cli ui-status
py -3 -m quant_platform.cli launch-ui --host localhost --port 8501
```

The UI reads existing local artifacts only. It can show provider rows, configured booleans, dataset manifests, data quality reports, backtest reports, profile comparisons, and the explicit safety policy. Any operation that would contact providers remains a separate CLI command and must be run explicitly outside the UI.

The UI can explain formulas and show charts from local JSON reports. If a backtest report does not include time series such as `equity_curve` or `net_returns`, the UI shows aggregate metrics and states that curve charts require those series in a future pipeline. This prevents inventing data for visualization.
