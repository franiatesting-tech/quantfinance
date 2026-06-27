# Microsoft Qlib Reference Notes

Status: external reference only, no runtime dependency.

Useful ideas for this platform:

- Clear separation between data layer, feature generation, model training, backtesting, and workflow records.
- Dataset handlers that make data availability and preprocessing explicit.
- Experiment workflow discipline before introducing predictive models.

Current decision:

- Do not add Qlib as a dependency.
- Keep the local foundation smaller: synthetic bars, CSV registry, quality checks, returns, baselines, costs, and reports.
- Revisit only after validation splits, experiment tracking, and benchmarks are mature.
