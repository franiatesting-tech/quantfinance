# Iteration 006 - Real Data Validation

## Scope

Iteration 006 validates the read-only daily research flow with real public data. It adds data diagnostics, JSON data quality reports, registry coverage metadata, and conservative/aggressive profile comparison. It does not add trading, paper trading, private endpoints, predictive models, optimizers, or pricing modules.

## Commands Executed

Initial validation:

```powershell
git status --short --branch
git remote -v
git pull --ff-only
py -3 -m pytest
py -3 -m ruff check .
py -3 -c "import collections, csv, pathlib; p=pathlib.Path(r'..\\docs\\audit\\01_traceability_matrix.csv'); rows=list(csv.DictReader(p.open(newline='', encoding='utf-8'))); ids=[r['requirement_id'] for r in rows]; dup=[i for i,c in collections.Counter(ids).items() if c>1]; print(f'rows={len(rows)} duplicate_ids={dup}')"
```

Smoke read-only real data:

```powershell
py -3 -m quant_platform.cli download-real-data --config configs/universe_etfs_crypto_daily.yaml --start 2021-01-01 --end 2025-12-31 --limit-equity 5 --limit-crypto 3
py -3 -m quant_platform.cli run-backtest-demo --dataset-id real_daily_demo --version v1 --profile conservative
py -3 -m quant_platform.cli compare-profiles --dataset-id real_daily_demo --version v1 --config configs/universe_etfs_crypto_daily.yaml --risk-config configs/risk_profiles.yaml
```

## Network And Provider Result

- Network/proveedores: available during smoke.
- Dataset registered: `real_daily_demo/v1`.
- Local registry path: `data/registry/real_daily_demo/v1/manifest.json`.
- Quality report path: `data/registry/real_daily_demo/v1/data_quality_report.json`.
- Generated backtest reports: `reports/generated/real_daily_demo_v1_conservative_report.json`, `reports/generated/real_daily_demo_v1_aggressive_report.json`, and `reports/generated/real_daily_demo_v1_profile_comparison.json`.
- These paths are ignored by Git and must not be committed.

## Symbols And Coverage

- Successful symbols: `BTCUSDT`, `EFA`, `ETHUSDT`, `IWM`, `QQQ`, `SOLUSDT`, `SPY`, `VTI`.
- Failed symbols: none in this smoke.
- Date range in quality report: `2021-01-01T00:00:00+00:00` to `2025-12-30T00:00:00+00:00`.
- Coverage warnings: `equity_date_gaps_require_calendar_review`.
- Failed quality checks: none.
- Suitable for backtest demo: true.

## Backtest Demo Summary

- Profile: `conservative`.
- Number of assets: 8.
- Number of complete return observations after pivot/dropna: 343.
- Cost policy: `mixed_equity_crypto_conservative_max`.
- Final equity: `19236.60765343768`.

## Profile Comparison Summary

| Metric | Conservative | Aggressive | Difference aggressive - conservative |
| --- | ---: | ---: | ---: |
| final_equity | 19236.60765343768 | 19224.754970460406 | -11.852682977274526 |
| annualized_return | 0.6171394061258386 | 0.6164072940563226 | -0.0007321120695160666 |
| annualized_volatility | 0.3913477406753106 | 0.3913338150671249 | -0.000013925608185694216 |
| sharpe | 1.4257968688013627 | 1.4246742247363544 | -0.0011226440650082825 |
| max_drawdown | -0.42249578182669767 | -0.42249578182669756 | 0.000000000000000111 |
| VaR_95 | 0.038566660484713224 | 0.038566660484713224 | 0.0 |
| ES_95 | 0.0544133994456504 | 0.0544133994456504 | 0.0 |
| turnover | 1.0 | 1.0 | 0.0 |
| transaction_costs | 0.0022500000000000003 | 0.002875 | 0.0006249999999999997 |

## Drawdown Warning

The realized max drawdown exceeded both profile targets in the smoke:

- Conservative realized max drawdown: approximately `42.25%` versus target `15%`.
- Aggressive realized max drawdown: approximately `42.25%` versus target `30%`.

The 15% and 30% drawdown values are risk evaluation targets, not guarantees. This demo uses equal-weight exposure and does not enforce dynamic drawdown control.

## Implementation Notes

- Added `src/quant_platform/data/diagnostics.py` for coverage and gap diagnostics.
- Added `src/quant_platform/reporting/data_quality_report.py` for JSON quality reports.
- Added `src/quant_platform/pipelines/profile_comparison.py` for conservative/aggressive comparison.
- Updated `src/quant_platform/data/registry.py` and quality parsing to handle mixed timestamp precision from equity and crypto providers.
- Updated CLI with `compare-profiles`.
- Tests use mocks and local temporary registries; no test requires network access.

## Limitations

- yfinance is a free public research provider with potential terms, adjustment, and coverage limitations.
- Adjusted OHLC handling for equity remains a research approximation and needs corporate-action validation.
- Equity date gaps require calendar-aware review; current diagnostics warn rather than fail.
- Crypto daily gaps are treated as failed checks because spot crypto trades 24/7.
- Survivorship bias remains open because the universe uses currently selected visible tickers.
- Stablecoin quote risk for USDT pairs is documented but not modeled.
- Pivoting to a complete close matrix can drop many dates when assets have different histories; coverage reports must be reviewed before interpreting backtests.

## Next Steps

- Add calendar-aware equity gap classification using `pandas-market-calendars`.
- Add report summarizing rows lost during pivot/dropna.
- Add provider terms/licensing review checklist.
- Add inactive/delisted asset metadata before broader universe claims.

## Final Validation

- `py -3 -m pytest`: `143 passed`.
- `py -3 -m ruff check .`: `All checks passed!`.
- Traceability matrix: `rows=47 duplicate_ids=[]`.
