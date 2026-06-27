"""Compare conservative and aggressive profiles on the same local dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_platform.pipelines.real_data_backtest import (
    RealDataBacktestPipelineError,
    run_real_data_backtest_pipeline,
)
from quant_platform.risk.profiles import get_risk_profile, load_risk_profiles


class ProfileComparisonPipelineError(ValueError):
    """Raised when profile comparison cannot be run safely."""


COMPARISON_METRICS = {
    "final_equity": "final_equity",
    "annualized_return": "annualized_return",
    "annualized_volatility": "annualized_volatility",
    "sharpe": "sharpe_ratio",
    "max_drawdown": "max_drawdown",
    "VaR_95": "historical_var_95",
    "ES_95": "historical_expected_shortfall_95",
    "turnover": "total_turnover",
    "transaction_costs": "total_transaction_cost",
}


def run_profile_comparison_pipeline(
    dataset_id: str,
    version: str,
    universe_config: str | Path,
    risk_profiles_config: str | Path,
    registry_dir: str | Path = "data/registry",
    profiles: tuple[str, str] = ("conservative", "aggressive"),
    strategies: tuple[str, ...] = ("equal_weight_real_data_demo",),
    benchmarks: tuple[str, ...] = ("buy_and_hold",),
    start: str | None = None,
    end: str | None = None,
    output_dir: str | Path = "reports/generated",
    trial_registry_path: str | Path = "reports/generated/strategy_trials.jsonl",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run both risk profiles on the same dataset and compare key risk metrics."""

    if len(profiles) != 2:
        raise ProfileComparisonPipelineError("Exactly two profiles are required for comparison.")
    output_path = Path(output_dir)
    comparison_report_path = output_path / f"{dataset_id}_{version}_profile_comparison.json"
    if dry_run:
        return {
            "dry_run": True,
            "dataset_id": dataset_id,
            "version": version,
            "profiles": list(profiles),
            "strategies": list(strategies),
            "benchmarks": list(benchmarks),
            "start": start,
            "end": end,
            "intended_report_path": str(comparison_report_path),
        }

    profile_config = load_risk_profiles(risk_profiles_config)
    summaries: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for profile_name in profiles:
        get_risk_profile(profile_config, profile_name)
        try:
            summary = run_real_data_backtest_pipeline(
                dataset_id=dataset_id,
                version=version,
                registry_dir=registry_dir,
                risk_profiles_path=risk_profiles_config,
                profile_name=profile_name,
                universe_config_path=universe_config,
                start=start,
                end=end,
                report_dir=output_path,
                trial_registry_path=trial_registry_path,
                download_if_missing=False,
            )
        except RealDataBacktestPipelineError as exc:
            raise ProfileComparisonPipelineError(str(exc)) from exc
        report_path = Path(str(summary["report_path"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summaries[profile_name] = summary
        reports[profile_name] = report
        target = float(profile_config[profile_name]["target_max_drawdown"])
        realized_drawdown = abs(float(report["max_drawdown"] or 0.0))
        if realized_drawdown > target:
            warnings.append(
                f"{profile_name} realized max drawdown {realized_drawdown:.4f} exceeds "
                f"target {target:.4f}; targets are not guarantees."
            )

    comparison_table = _comparison_table(reports, profiles)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": dataset_id,
        "version": version,
        "profiles": summaries,
        "strategies": list(strategies),
        "benchmarks": list(benchmarks),
        "comparison_table": comparison_table,
        "warnings": warnings,
        "assumptions": [
            "Risk profile drawdown targets are evaluation thresholds, not guarantees.",
            "Both profiles use the same dataset, strategy, execution lag, and benchmark set.",
            "Cost differences come from risk profile configuration, not from live execution.",
        ],
    }
    comparison_report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload["report_path"] = str(comparison_report_path)
    return payload


def _comparison_table(
    reports: dict[str, dict[str, Any]],
    profiles: tuple[str, str],
) -> list[dict[str, float | str | None]]:
    first, second = profiles
    rows: list[dict[str, float | str | None]] = []
    for output_name, report_key in COMPARISON_METRICS.items():
        first_value = _clean_number(reports[first].get(report_key))
        second_value = _clean_number(reports[second].get(report_key))
        difference = None
        if first_value is not None and second_value is not None:
            difference = second_value - first_value
        rows.append(
            {
                "metric": output_name,
                first: first_value,
                second: second_value,
                f"difference_{second}_minus_{first}": difference,
            }
        )
    return rows


def _clean_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
