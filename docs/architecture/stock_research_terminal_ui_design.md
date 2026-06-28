# Stock Research Terminal UI Design

The UI is a single local Streamlit page named `Stock Research Terminal`.

## Principles

- Research-only, read-only and no automatic network calls.
- Default view is a selected stock, not internal app diagnostics.
- Diagnostics remain available only inside `Developer diagnostics`.
- Tabs organize one page: Overview, Data, Risk, Monte Carlo, ML, Backtesting, Options, Report.

## Main Controls

- Report directory selector.
- Terminal report selector.
- Stock selector.
- Regeneration command display, not automatic execution.

## Core Cards

Cards show last price, cumulative return, CAGR, volatility, Sharpe, Sortino, Treynor, Jensen alpha, max drawdown, VaR 95, ES 95, ML directional accuracy and research-only decision signal.

## Safety

The UI never sends orders, never connects to brokers, never displays `.env` values and never presents the research-only signal as financial advice.
