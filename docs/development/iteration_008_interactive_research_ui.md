# Iteration 008 Interactive Research UI

## Objective

Add a local read-only interface for inspecting platform status, provider configuration, local datasets, data quality, generated reports, profile comparisons, and allowed/prohibited actions.

## Implemented

- Optional `[project.optional-dependencies].ui` extra with `streamlit` and `plotly`.
- `quant_platform.ui.safety` for explicit read-only safety policy.
- `quant_platform.ui.report_loader` for local manifest, JSON report, and JSONL trial discovery.
- `quant_platform.ui.view_models` for provider, universe, risk, dataset, report, and backtest view models.
- `quant_platform.ui.charts` for Plotly chart builders.
- `quant_platform.ui.formulas` for explained mathematical formulas with LaTeX.
- `quant_platform.ui.explainers` for non-technical concept explanations and warnings.
- `quant_platform.ui.actions` for `ui-status`, launch command construction, and safe command catalog.
- `quant_platform.ui.app` as a Streamlit local UI.
- CLI commands: `ui-status` and `launch-ui`.
- Provider status now distinguishes keyed providers that are `configured_but_disabled`.
- `validate-providers --network-smoke` skips disabled providers before instantiation.

## Safety

- API key values are not printed, cached, serialized, or included in reports.
- UI displays only `configured` booleans and provider status.
- UI does not run provider downloads, network smoke tests, backtests, paper trading, live trading, orders, brokers, or private endpoints.
- Tests do not start Streamlit and do not use network calls.

## Commands

```powershell
py -3 -m pip install -e ".[dev,ui]"
py -3 -m quant_platform.cli ui-status
py -3 -m quant_platform.cli launch-ui --dry-run
py -3 -m quant_platform.cli launch-ui --host localhost --port 8501
```

## Validation

Validation after implementation:

- `py -3 -m ruff check .`: passed.
- `py -3 -m pytest tests/unit`: `200 passed` after installing the UI extra.
- `py -3 -m pytest`: `205 passed`.
- Traceability matrix: `rows=48 duplicate_ids=[]`.

Expanded UI validation after formulas, explainers, chart coverage, and richer navigation:

- `py -3 -m pytest tests/unit`: `208 passed`.
- `py -3 -m pytest`: `213 passed`.
- `py -3 -m ruff check .`: passed.
- Traceability matrix: `rows=48 duplicate_ids=[]`.
- `py -3 -m quant_platform.cli ui-status`: returns `ui_available`, dependency status, app path, read-only state, provider summary, artifact counts, and `no_secrets_exposed=true`.
- `py -3 -m quant_platform.cli launch-ui --dry-run`: returns the Streamlit command without starting a server.
