# Iteration 007 - Keyed Read-Only Providers

## Scope

Iteration 007 adds read-only providers that can use local API keys from `.env` without ever printing, logging, caching, or committing those secrets. It remains a research-only layer: no trading, no paper trading, no private endpoints, no account endpoints, no margin, no futures/perpetuals, and no leverage.

## Environment Variables

Expected local variables, with values never shown:

- `ALPHA_VANTAGE_API_KEY`
- `POLYGON_API_KEY`
- `NASDAQ_DATA_LINK_API_KEY`
- `CRYPTOCOMPARE_API_KEY`

Provider flags and runtime controls:

- `ALPHA_VANTAGE_ENABLED`
- `POLYGON_ENABLED`
- `NASDAQ_DATA_LINK_ENABLED`
- `CRYPTOCOMPARE_ENABLED`
- `PROVIDER_HTTP_TIMEOUT_SECONDS`
- `PROVIDER_MAX_RETRIES`
- `PROVIDER_RETRY_BACKOFF_SECONDS`
- `PROVIDER_CACHE_ENABLED`

`.env.example` contains names only. `.env` is local and ignored by Git.

## Providers Implemented

| Provider | Market | Endpoint type | Scope |
| --- | --- | --- | --- |
| Alpha Vantage | Equity/ETF | `TIME_SERIES_DAILY_ADJUSTED` | Daily adjusted OHLCV |
| Polygon | Equity/ETF | Daily aggregates | Daily adjusted OHLCV |
| Nasdaq Data Link | Configurable | Dataset endpoint | Minimal OHLCV-compatible dataset support |
| CryptoCompare | Crypto spot | `histoday` | Daily spot OHLCV |

All providers use read-only public data APIs. API keys are passed only to HTTP params/headers and sanitized from errors and cache keys.

## CLI

```powershell
py -3 -m quant_platform.cli list-providers
py -3 -m quant_platform.cli validate-providers
py -3 -m quant_platform.cli validate-providers --network-smoke
py -3 -m quant_platform.cli download-real-data --config configs/universe_etfs_crypto_daily.yaml --start 2024-01-01 --end 2024-03-31 --limit-equity 2 --limit-crypto 2 --equity-provider polygon --crypto-provider cryptocompare
```

`list-providers` and `validate-providers` return booleans/status only. They do not include API key values or suffixes.

## Config

`configs/data_providers.yaml` defines provider metadata, market types, env var names, and priority. Lower priority number means higher preference.

## Smoke Result

Observed in this run:

- `.env` exists locally and is ignored by Git.
- `list-providers` and `validate-providers` executed without printing key values.
- Expected keyed providers were detected as `configured=false` in this environment.
- `validate-providers --network-smoke` skipped keyed providers with `skipped_missing_api_key`.
- Small Polygon/CryptoCompare and AlphaVantage/CryptoCompare download attempts failed safely with sanitized missing-key errors.

No keyed-provider real dataset was generated in this iteration because expected keys were not detected by settings as non-empty values.

## Tests

Tests use mocks only for HTTP/provider behavior. No unit test performs a real network request or uses real `.env` values.

Final validation:

- `py -3 -m pytest`: `183 passed`.
- `py -3 -m ruff check .`: `All checks passed!`.
- Traceability matrix: `rows=47 duplicate_ids=[]`.

## Security Controls

- API key dataclass fields use `repr=False`.
- `show-settings` uses a public sanitized settings snapshot.
- HTTP errors sanitize params named `apikey`, `api_key`, `apiKey`, `token`, and `key`.
- Optional cache filenames are hashes of sanitized URL/params and do not include key values.
- `.env`, `.env.*`, `.cache/`, `cache/`, `data/registry/`, and `reports/generated/` are ignored.

## Limitations

- Provider plan/rate-limit behavior depends on the user's account.
- Nasdaq Data Link needs a concrete `dataset_code` before operational use.
- Vendor adjusted-close methodologies and timestamp conventions may differ.
- Comparing providers currently focuses on coverage/quality metadata, not tick-level price reconciliation.

## Still Prohibited

- Trading real or paper trading.
- Orders, brokers, account/private endpoints, margin, futures, perpetuals, funding real, leverage real.
- ML/deep learning/DRL, advanced optimization, Heston/Black-Scholes operational pricing, dashboards, MLflow.
