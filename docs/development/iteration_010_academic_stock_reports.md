# Iteration 010 Academic Stock Reports

## Objective

Generate a rigorous, visual, two-level academic report for each stock in the professional quant terminal report.

## Implemented

- Enriched terminal JSON schema with `metadata`, `data_provenance`, `stocks`, `bibliography`, and top-level `warnings`.
- Per-stock payloads include data used, OHLCV summary, quality review, price/volume/return/log-return/cumulative/drawdown/rolling-volatility/rolling-Sharpe/rolling-beta series, metrics, VaR/ES, Monte Carlo scenarios, backtesting results, options analytics, warnings, and bibliography references.
- `quant_platform.reporting.academic_figures` writes Plotly HTML figures.
- `quant_platform.reporting.academic_conclusions` writes deterministic conclusions without recommendations.
- `quant_platform.reporting.academic_stock_report` builds report models and renders Markdown/HTML.
- CLI commands:
  - `generate-stock-academic-report`.
  - `generate-all-stock-academic-reports`.
- UI section `Academic Reports` for generated metadata, paths, figure list, and regeneration commands.
- Tests for conclusions, figures, renderer, CLI, report loader, enriched terminal report, and integration generation.

## Commands

```powershell
py -3 -m quant_platform.cli build-quant-terminal-report --config configs/quant_terminal_3_stocks.yaml
py -3 -m quant_platform.cli generate-all-stock-academic-reports --terminal-report reports/generated/quant_terminal/3stocks_10y_report.json --output-dir reports/generated/academic_stock_reports --format md --format html --include-figures --overwrite
```

## Safety

- No network required for academic report generation.
- No trading, paper trading, brokers, orders, private endpoints, or account data.
- Generated Markdown/HTML/figure outputs are local and ignored by Git.
- Conclusions are deterministic and do not recommend operational actions.
- Reports include `Research-only. This is not investment advice.`.

## PDF Status

PDF export is not part of this iteration. Markdown and HTML are the supported outputs.

## Validation Snapshot

Focused validation during implementation:

- `py -3 -m pytest tests/unit/reporting/test_academic_conclusions.py tests/unit/reporting/test_academic_figures.py tests/unit/reporting/test_academic_stock_report.py tests/integration/test_academic_stock_report_generation.py tests/unit/test_cli.py tests/unit/ui/test_report_loader.py tests/unit/research/test_report.py`: `27 passed`.
- `py -3 -m ruff check` on modified academic reporting, CLI, UI, report loader, and enriched report files: passed.

Final validation is recorded in audit notes.
