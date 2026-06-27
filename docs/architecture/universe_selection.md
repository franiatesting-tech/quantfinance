# Universe Selection

Iteration 005 starts with a daily research universe combining liquid ETFs and crypto majors spot.

## Why Liquid ETFs First

- ETFs such as `SPY`, `QQQ`, `IWM`, `VTI`, `EFA`, `EEM`, `TLT`, `GLD`, and `VNQ` give broad exposure with relatively high liquidity and clean daily histories.
- ETF coverage reduces single-name corporate-action complexity during early validation.
- Sector/style ETFs allow controlled diversification without introducing a large stock-selection problem.

## Why Crypto Majors Spot First

- Crypto majors quoted in USDT on Binance spot offer public OHLCV without private keys.
- Spot data avoids futures, perpetuals, margin, funding, liquidation, and leverage mechanics.
- Crypto remains 24/7; annualization and calendar assumptions must stay explicit.

## Europe And Spain

- Europe/Spain starts through proxies such as `VGK`, `FEZ`, and `EWP`, plus a watchlist of Spanish yfinance tickers.
- Coverage quality for European and Spanish tickers can vary in yfinance.
- Expansion should be progressive and based on observed coverage, missing data, currency treatment, and corporate-action handling.

## Research Universe Versus Tradable Universe

- The research universe is a candidate set for data collection and analysis.
- The tradable universe is not defined yet because there is no broker, no execution layer, no liquidity model, and no trading permission.
- Watchlists are intentionally separated from investable production universes.

## Bias And Limitations

- Survivorship bias remains open because the first universe uses currently visible assets.
- Delisted equities, inactive ETFs, and delisted crypto pairs are not yet represented.
- Provider outages and ticker-specific failures must be recorded, not hidden.
- Coverage reports must be reviewed after every real download because provider coverage can differ by ticker and date.
- Complete close matrices can drop dates for assets with shorter histories, which affects backtest interpretation.
- The conservative 15% and aggressive 30% drawdown targets are risk review thresholds, not guaranteed outcomes.
- Any future production-grade universe must include liquidity, listing history, currency, venue, and tradability metadata.

## Iteration 006 Smoke Coverage

The first controlled real-data smoke downloaded 8 symbols successfully: `SPY`, `QQQ`, `IWM`, `VTI`, `EFA`, `BTCUSDT`, `ETHUSDT`, and `SOLUSDT`. No symbol failed in that smoke, but equity calendar gaps were flagged for review.
