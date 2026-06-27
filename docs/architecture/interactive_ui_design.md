# Interactive UI Design

Iteration 008 adds a local Streamlit/Plotly interface for read-only research inspection. The UI is an operations console, not an execution system.

## Scope

- Show public settings from `public_settings_dict` only.
- Show provider rows from `list_available_providers` with `enabled`, `configured`, `requires_api_key`, `market_types`, and `status`.
- Distinguish `configured_but_disabled` for keyed providers that have local credentials but disabled flags.
- Show universe and risk profile summaries from local YAML configs.
- Discover local dataset manifests under `data/registry/` without loading CSV files by default.
- Show data quality reports, generated backtest reports, profile comparison reports, provider comparison reports, and strategy trial counts when local files exist.
- Show allowed and prohibited actions explicitly.
- Explain formulas with LaTeX plus non-technical intuition.
- Explain concepts such as provider, OHLCV, dataset, quality report, backtest, equity curve, drawdown, read-only, overfitting trials, and configured vs enabled.

## Non-Scope

- No live trading.
- No paper trading.
- No orders, brokers, account endpoints, private exchange endpoints, margin, futures, perpetuals, or leverage-real logic.
- No automatic provider downloads.
- No automatic network smoke tests.
- No automatic backtest execution.
- No API key value display, logging, caching, or serialization.

## Modules

| Module | Responsibility |
| --- | --- |
| `quant_platform.ui.app` | Streamlit page composition and local visual design. |
| `quant_platform.ui.actions` | UI status, launch command construction, safe command catalog. |
| `quant_platform.ui.safety` | Allowed/prohibited action policy and sensitive-key guards. |
| `quant_platform.ui.report_loader` | Read local manifests, JSON reports, and JSONL trial registry. |
| `quant_platform.ui.view_models` | Pure data transforms for providers, universe, risk profiles, datasets, and reports. |
| `quant_platform.ui.charts` | Plotly chart builders with optional dependency import. |
| `quant_platform.ui.formulas` | Formula cards with LaTeX, plain explanation, intuition, inputs, outputs, usage, and misinterpretation warnings. |
| `quant_platform.ui.explainers` | Non-technical concept explainers and risk warnings. |

## CLI Entry Points

```powershell
py -3 -m quant_platform.cli ui-status
py -3 -m quant_platform.cli launch-ui --dry-run
py -3 -m quant_platform.cli launch-ui --host localhost --port 8501
```

`launch-ui --dry-run` returns the Streamlit command without starting a server. Tests use this path so they do not start Streamlit.

## Data Flow

1. `load_settings_from_env` loads settings through the existing safe path.
2. `public_settings_dict` strips credential values and keeps configured booleans only.
3. `build_platform_snapshot` composes provider rows, universe summary, risk profile summary, local dataset manifests, generated reports, trial counts, and safety policy.
4. Streamlit renders the snapshot and optional Plotly figures.

## Screens

- `Inicio`: mode, simulated capital, local artifact counts, provider status, and safe terminal commands.
- `Flujo conceptual`: the end-to-end research pipeline and what can fail at each step.
- `Estado del sistema`: UI dependency readiness, read-only status, and public settings.
- `Providers`: `configured`, `enabled`, `configured_but_disabled`, `missing_api_key`, and `available`.
- `Universo`: ETF/equity/index/crypto groups and counts.
- `Datasets` and `Calidad de datos`: local manifests, coverage, warnings, failed checks, gaps, and suitability flags.
- `Backtests`: metrics, conceptual VaR/ES, transaction cost chart, and series charts when reports include series.
- `Comparacion de perfiles`: conservative/aggressive reports side-by-side.
- `Formulas`: formula cards rendered with `st.latex`.
- `Acciones permitidas` and `Riesgos y limites`: safety boundaries and caveats.

## Chart Policy

All chart builders accept simple local data, do not read `.env`, do not make network calls, and return an explanatory empty figure when data is absent. Charts include titles, axis labels, hover text, and notes where a chart could be misread.

## Visual Direction

The interface uses a dark risk-operations console style: compact status cards, amber/cyan status accents, and dense tables for audit work. The design favors clarity and traceability over marketing-style presentation.

## Testing Policy

- UI tests do not start Streamlit.
- UI tests do not make network calls.
- UI tests do not require real `.env` values.
- Chart tests exercise Plotly figure builders only.
- Loader tests create temporary manifests/reports and do not read real data CSVs.
