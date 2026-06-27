"""Read-only real-data backtest pipeline foundation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.backtesting.benchmarks import (
    buy_and_hold_target_weights,
    equal_weight_target_weights,
)
from quant_platform.backtesting.costs import TransactionCostConfig
from quant_platform.backtesting.engine import BacktestConfig, run_vectorized_backtest
from quant_platform.data.ingestion import (
    download_combined_daily_universe,
    load_universe_config,
    register_real_dataset,
)
from quant_platform.data.providers.base import MarketDataProvider, OHLCVResponse
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.registry import load_dataset
from quant_platform.data.schemas import MarketType
from quant_platform.features.returns import simple_returns
from quant_platform.reporting.backtest_report import build_backtest_report
from quant_platform.risk.profiles import (
    build_cost_config_for_market,
    get_risk_profile,
    load_risk_profiles,
)
from quant_platform.validation.overfitting_registry import (
    StrategyTrialRecord,
    record_strategy_trial,
)


class RealDataBacktestPipelineError(ValueError):
    """Raised when the read-only backtest pipeline cannot run safely."""


def _close_matrix(market_bars: pd.DataFrame) -> pd.DataFrame:
    bars = market_bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, format="mixed")
    close = bars.pivot(index="timestamp", columns="asset_id", values="close").sort_index()
    close = close.astype(float).dropna(axis=0, how="any")
    if close.shape[0] < 2:
        raise RealDataBacktestPipelineError("At least two complete close rows are required.")
    return close


def _cost_config_for_dataset(
    profile: dict[str, Any],
    market_bars: pd.DataFrame,
) -> tuple[TransactionCostConfig, str]:
    market_types = {
        MarketType(value)
        for value in market_bars["market_type"].dropna().astype(str).unique().tolist()
    }
    if not market_types:
        raise RealDataBacktestPipelineError("Dataset must include at least one market_type.")
    if len(market_types) == 1:
        market_type = next(iter(market_types))
        return build_cost_config_for_market(profile, market_type), market_type.value
    supported_mixed = {MarketType.EQUITY, MarketType.CRYPTO}
    if not market_types.issubset(supported_mixed):
        raise RealDataBacktestPipelineError(f"Unsupported mixed market types: {market_types}")

    equity_costs = build_cost_config_for_market(profile, MarketType.EQUITY)
    crypto_costs = build_cost_config_for_market(profile, MarketType.CRYPTO)
    return (
        TransactionCostConfig(
            commission_rate=max(equity_costs.commission_rate, crypto_costs.commission_rate),
            spread_rate=max(equity_costs.spread_rate, crypto_costs.spread_rate),
            slippage_rate=max(equity_costs.slippage_rate, crypto_costs.slippage_rate),
            funding_rate=0.0,
        ),
        "mixed_equity_crypto_conservative_max",
    )


def run_real_data_backtest_pipeline(
    dataset_id: str,
    version: str,
    registry_dir: str | Path,
    risk_profiles_path: str | Path,
    profile_name: str = "conservative",
    universe_config_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    equity_provider: MarketDataProvider | None = None,
    crypto_provider: MarketDataProvider | None = None,
    report_dir: str | Path = "reports/generated",
    trial_registry_path: str | Path = "reports/generated/strategy_trials.jsonl",
    limit_equity: int | None = None,
    limit_crypto: int | None = None,
    download_if_missing: bool = False,
) -> dict[str, Any]:
    """Run a read-only data -> backtest -> report -> trial registry pipeline."""

    try:
        market_bars, registered = load_dataset(registry_dir, dataset_id, version)
        dataset_metadata = registered.metadata
    except Exception as exc:  # noqa: BLE001 - fallback is controlled by explicit flag.
        if not download_if_missing:
            raise RealDataBacktestPipelineError(str(exc)) from exc
        if universe_config_path is None or start is None or end is None:
            raise RealDataBacktestPipelineError(
                "universe_config_path, start, and end are required when downloading."
            ) from exc
        universe = load_universe_config(universe_config_path)
        response: OHLCVResponse = download_combined_daily_universe(
            universe,
            start=start,
            end=end,
            equity_provider=equity_provider,
            crypto_provider=crypto_provider,
            limit_equity=limit_equity,
            limit_crypto=limit_crypto,
        )
        registered = register_real_dataset(
            response,
            registry_dir,
            dataset_id,
            version,
            MarketType.EQUITY,
        )
        market_bars = response.data
        dataset_metadata = registered.metadata

    run_market_data_quality_checks(market_bars)
    close = _close_matrix(market_bars)
    returns = simple_returns(close).dropna(axis=0, how="any")
    profiles = load_risk_profiles(risk_profiles_path)
    profile = get_risk_profile(profiles, profile_name)
    cost_config, cost_policy = _cost_config_for_dataset(profile, market_bars)
    config = BacktestConfig(
        periods_per_year=252,
        execution_lag=1,
        cost_config=cost_config,
        initial_capital=10_000.0,
        allow_short=False,
        max_leverage=float(profile["max_gross_leverage"]),
    )
    target_weights = equal_weight_target_weights(returns)
    result = run_vectorized_backtest(returns, target_weights, config)
    benchmark_weights = (
        buy_and_hold_target_weights(close).reindex(returns.index).ffill().fillna(0.0)
    )
    benchmark = run_vectorized_backtest(returns, benchmark_weights, config)
    report = build_backtest_report(result, {"buy_and_hold": benchmark})

    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{dataset_id}_{version}_{profile_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    record_strategy_trial(
        StrategyTrialRecord(
            trial_id=f"{dataset_id}-{version}-{profile_name}",
            strategy_name="equal_weight_real_data_demo",
            hypothesis="Equal-weight benchmark should be the first real-data simulation baseline.",
            parameters={
                "profile": profile_name,
                "dataset_id": dataset_id,
                "version": version,
                "cost_policy": cost_policy,
            },
            train_period={"start": str(returns.index[0]), "end": str(returns.index[-1])},
            validation_period=None,
            test_period=None,
            created_at=datetime.now(tz=UTC).isoformat(),
            status="DRAFT",
            metrics=result.metrics,
            notes="Iteration 005 read-only demo; no model selection.",
        ),
        trial_registry_path,
    )

    return {
        "dataset_id": dataset_metadata.dataset_id,
        "version": dataset_metadata.version,
        "profile": profile_name,
        "number_of_assets": int(close.shape[1]),
        "number_of_observations": int(len(returns)),
        "cost_policy": cost_policy,
        "report_path": str(report_path),
        "final_equity": report["final_equity"],
    }
