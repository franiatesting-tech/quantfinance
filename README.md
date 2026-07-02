# Quant Platform

Auditable Quant Finance research platform for equity and crypto experiments. The current implementation includes mathematical conventions, OHLCV quality checks, synthetic data, local dataset registry, returns, drawdown, historical/parametric/Monte Carlo VaR, Expected Shortfall, performance metrics, portfolio optimization, Monte Carlo, responsible walk-forward ML diagnostics, research-only decision signals, Black-Scholes/binomial options, fixed income, rates, hedging, exposure, vectorized backtesting, academic per-stock reports, a final institutional state-of-the-art research package, and a local read-only Streamlit Institutional Quant Research Terminal.

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
- Predictive ML is limited to responsible walk-forward diagnostics and baselines unless optional `scikit-learn` is installed; it never claims guaranteed prediction.
- Black-Scholes, binomial options, fixed income, rates derivatives, hedging, and counterparty exposure blocks are parametric educational models unless real market inputs are supplied in future iterations.
- The local UI inspects public settings, provider status, local dataset manifests, quality reports, generated reports, professional terminal reports, and allowed/prohibited actions. It does not run data downloads, backtests, network smoke tests, orders, or paper trading automatically.

## Local Setup

```powershell
py -3 -m pip install -e ".[dev]"
```

Optional local UI dependencies:

```powershell
py -3 -m pip install -e ".[dev,ui]"
```

Use `py -3.13` for the validated final institutional package workflow in this Windows environment. The `python` alias may point to the Microsoft Store stub unless configured manually.

Final package and PDF dependencies:

```powershell
py -3.13 -m pip install -e ".[dev,ui,ml,pdf]"
py -3.13 -m playwright install chromium
```

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
py -3 -m quant_platform.cli build-quant-terminal-report --config configs/quant_terminal_10_stocks.yaml
py -3 -m quant_platform.cli generate-institutional-study --terminal-report reports/generated/quant_terminal/10stocks_10y_report.json --output-dir reports/generated/institutional_study --max-table-rows 20
py -3 -m quant_platform.cli build-institutional-study --config configs/quant_terminal_10_stocks.yaml --provider yfinance --output-dir reports/generated/institutional_study --max-table-rows 20
py -3 -m quant_platform.cli generate-stock-academic-report --asset AAPL --terminal-report reports/generated/quant_terminal/10stocks_10y_report.json --format md --format html --format pdf --include-figures --overwrite
py -3 -m quant_platform.cli generate-all-stock-academic-reports --terminal-report reports/generated/quant_terminal/10stocks_10y_report.json --output-dir reports/generated/academic_stock_reports --format md --format html --format pdf --include-figures --overwrite
py -3.13 -m quant_platform.cli build-final-institutional-package --config configs/quant_terminal_10_stocks.yaml --provider yfinance --output-dir reports/generated/final_package --max-table-rows 20
py -3.13 -m quant_platform.cli build-seven-algorithm-study --config configs/quant_terminal_10_stocks.yaml --provider yfinance --output-dir reports/generated/seven_algorithms_study --bibliography-pack ../quant_finance_papers_pack --max-table-rows 30
py -3 -m quant_platform.cli list-providers
py -3 -m quant_platform.cli validate-providers
py -3 -m quant_platform.cli ui-status
py -3 -m quant_platform.cli launch-ui --dry-run
```

The CLI loads safe settings and refuses live-trading scope.

## Local UI

```powershell
py -3 -m quant_platform.cli launch-ui --host localhost --port 8501
```

The UI is local and read-only. The main screen is an `Institutional Quant Research Terminal` when a final package exists, with tabs for Overview, Asset Results, Portfolio, Tail Risk, ML Confidence, Robustness, Decision Gates, Final Paper and Methodology. Developer diagnostics are hidden in an expander.

The terminal consumes `reports/generated/quant_terminal/10stocks_10y_report.json` and shows stock detail, data quality, risk, Monte Carlo, ML, backtesting, options and report outputs without running downloads or orders.

The `Formulas` section explains simple return, log return, portfolio return, equity curve, volatility, Sharpe, Sortino, max drawdown, VaR, Expected Shortfall, turnover, and transaction costs with LaTeX plus plain-language interpretation.

The UI is not financial advice. Backtests are historical simulations with fictitious capital and do not guarantee future outcomes.

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
- `reports/generated/quant_terminal/10stocks_10y_report.json`: professional 10-stock terminal report.
- `reports/generated/portfolio_optimization/10stocks_10y_frontier.csv`: CSV spreadsheet equivalent for the optimization frontier.
- `reports/generated/institutional_study/`: strict real-data institutional study package with JSON, summary Markdown, academic paper Markdown, CSV tables, and HTML/SVG figures.
- `reports/generated/academic_stock_reports/<asset>/`: per-stock academic Markdown, HTML, optional PDF, metadata, and Plotly figure HTML files.
- `reports/generated/final_package/`: final institutional package with terminal report, institutional study, CSV tables, HTML figures, reproducibility manifest, and final Markdown/HTML/PDF paper.
- `reports/generated/seven_algorithms_study/`: seven-algorithm replication study with JSON, Markdown, HTML, PDF, CSV tables, Plotly HTML figures, bibliography trace and broker-readiness decisions.

If these folders are empty, the UI shows safe CLI commands instead of failing. Demo artifacts, if generated in future, must be clearly marked `DEMO_SYNTHETIC_NOT_REAL_DATA` and remain ignored by Git.

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
- Professional quant terminal report: generated with `build-quant-terminal-report`; it analyzes AAPL/MSFT/NVDA, SPY benchmark, risk-free proxy/config rate, portfolio optimization, Monte Carlo, VaR, strategy backtests, options, fixed income, rates, hedging, exposure, and method bibliography.
- Institutional state-of-the-art study: generated with `generate-institutional-study` from an existing terminal report, or with `build-institutional-study` from provider data and config. It rejects synthetic data by default, writes first-row data tables, metrics, VaR/ES, ML audit, correlation/covariance matrices, portfolio weights, portfolio tail risk, concentration diagnostics, shrinkage sensitivity, critical findings, and HTML/SVG figures for normalized prices, cumulative returns, drawdowns, correlation heatmap, risk-return scatter, efficient frontier, and research decision scores. It also writes `institutional_quant_finance_paper.md`, a detailed academic paper with glossary, formulas, economic interpretation, limitations, roadmap, and implementation plan.
- Academic stock reports: generated with `generate-stock-academic-report` or `generate-all-stock-academic-reports`; each report contains 25 sections, formulas, figures, ML assessment, quantitative decision signal, conclusions, limitations, reproducibility, mathematical appendix, and bibliography traceability. PDF export uses WeasyPrint, Playwright Chromium, or Pandoc when available; otherwise HTML remains printable and metadata records `PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER`.
- Final institutional package: generated with `build-final-institutional-package`; it creates the terminal report, institutional study, final paper, reproducibility manifest, CSV tables, HTML figures and optional PDF. The paper includes dependency/configuration instructions for PDF rendering, factor data and manual email delivery.
- Seven algorithms study: generated with `build-seven-algorithm-study`; it replicates Black-Scholes, Monte Carlo, Markowitz, OU pairs trading, Kalman dynamic hedge ratio, GARCH and ML alpha diagnostics using provider data and the downloaded bibliography pack. Broker decisions remain blocked for real execution.
- Local UI: reads these reports for inspection only and does not generate new reports automatically.

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
- Research-only quantitative decision signals are limited to `FAVORABLE`, `NEUTRAL`, `CAUTION`, `UNFAVORABLE`, and `INSUFFICIENT_DATA` and are not investment advice.

## Documentation

- User decisions: `docs/development/decisions_needed.md`.
- Git setup: `docs/development/git_setup.md`.
- Iteration 005 notes: `docs/development/iteration_005_real_data_readonly_research.md`.
- Iteration 006 notes: `docs/development/iteration_006_real_data_validation.md`.
- Iteration 007 notes: `docs/development/iteration_007_keyed_readonly_providers.md`.
- Iteration 008 notes: `docs/development/iteration_008_interactive_research_ui.md`.
- Iteration 009 notes: `docs/development/iteration_009_professional_quant_terminal.md`.
- Professional quant terminal design: `docs/architecture/professional_quant_terminal_design.md`.
- Final institutional platform design: `docs/architecture/final_institutional_quant_platform_design.md`.
- Final institutional package user guide: `docs/usage/final_institutional_package_user_guide.md`.
- Seven algorithms replication study: `docs/research/seven_quant_algorithms_replication_study.md`.
- Academic stock report design: `docs/architecture/academic_stock_report_design.md`.
- Professional methods review: `docs/research/professional_quant_methods_review.md`.
- Interactive UI design: `docs/architecture/interactive_ui_design.md`.
- Provider architecture: `docs/architecture/real_data_pipeline.md`.
- Universe policy: `docs/architecture/universe_selection.md`.
- Local PDF bibliography metadata: `docs/literature/bibliography_map.md`.
