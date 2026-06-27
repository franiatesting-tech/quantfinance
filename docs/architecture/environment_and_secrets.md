# Environment And Secrets

Iteration 005 keeps the platform in local research mode. Real data providers are read-only, public/free, and used only for simulations with fictitious capital.

## Rules

- Never store secrets in source code, tests, docs, configs, or notebooks.
- Never commit a real `.env` file.
- Use `.env.example` only as a contract for variable names and safe defaults.
- Live trading is prohibited by project decision, even though settings still test a double-confirmation guardrail.
- Paper trading is not implemented and remains outside the current scope.
- Provider tests must use mocks or local fixtures, never live HTTP calls.
- No module may call broker, account, private exchange, order, margin, futures, or perpetual endpoints.
- Raw data, processed data, generated reports, and PDFs remain outside Git unless explicitly approved.

## Current Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QUANT_PLATFORM_ENV` | `local` | Runtime environment label. |
| `QUANT_PLATFORM_ALLOW_LIVE_TRADING` | `false` | Global live trading switch; must remain false for project scope. |
| `QUANT_PLATFORM_ALLOW_PAPER_TRADING` | `false` | Global paper trading switch; must remain false. |
| `REQUIRE_EXPLICIT_LIVE_TRADING_CONFIRMATION` | `true` | Defensive guardrail tested in settings. |
| `I_UNDERSTAND_LIVE_TRADING_RISK` | `false` | Defensive acknowledgement flag; not a permission to implement live trading. |
| `BASE_CURRENCY` | `USD` | Base currency for configs and reports. |
| `INITIAL_CAPITAL` | `10000` | Fictitious starting capital for simulations. |
| `DEFAULT_FREQUENCY` | `1d` | Default daily research frequency. |
| `DEFAULT_EQUITY_PROVIDER` | `yfinance` | Default public read-only equity/ETF provider. |
| `DEFAULT_CRYPTO_PROVIDER` | `binance_public` | Default public read-only crypto spot provider. |
| `YFINANCE_ENABLED` | `true` | Enables the read-only yfinance provider. |
| `STOOQ_ENABLED` | `false` | Future placeholder. |
| `COINGECKO_ENABLED` | `true` | Enables metadata placeholder only. |
| `BINANCE_PUBLIC_ENABLED` | `true` | Enables the public Binance spot OHLCV provider. |
| `CCXT_ENABLED` | `false` | CCXT is not installed and not approved for this iteration. |
| `ALPHA_VANTAGE_API_KEY` | empty | Optional future API key placeholder. |
| `POLYGON_API_KEY` | empty | Optional future API key placeholder. |
| `NASDAQ_DATA_LINK_API_KEY` | empty | Optional future API key placeholder. |
| `CRYPTOCOMPARE_API_KEY` | empty | Optional future API key placeholder. |
| `MAX_ALLOWED_LEVERAGE` | `1.0` | Safety cap; leverage above one is rejected for current risk profiles. |
| `CONSERVATIVE_TARGET_MAX_DRAWDOWN` | `0.15` | Conservative profile drawdown target. |
| `AGGRESSIVE_TARGET_MAX_DRAWDOWN` | `0.30` | Aggressive profile drawdown target. |

## Git Safety

The repository ignores `.env`, `.env.*`, caches, local data folders, generated reports, and PDFs under `docs/`. `.env.example`, YAML configs, bibliography metadata, and documentation are safe to commit.

## Data Safety

Real data ingestion writes only local research datasets and registry metadata. It must record source, venue, symbols, success/failure lists, coverage, and retrieval metadata. Failed symbols are recorded without failing the whole universe.
