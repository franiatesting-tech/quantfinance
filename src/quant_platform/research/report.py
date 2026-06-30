"""Professional quant terminal report assembly."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    hit_rate,
    sharpe_ratio,
    sortino_ratio,
)
from quant_platform.config.settings import PlatformSettings
from quant_platform.data.providers.factory import create_market_data_provider
from quant_platform.features.returns import cumulative_returns
from quant_platform.research.asset_dataset import (
    QuantTerminalConfig,
    build_ohlcv_request,
    load_quant_terminal_config,
    make_synthetic_ohlcv,
)
from quant_platform.research.backtesting_strategies import (
    run_terminal_backtests,
    serialize_backtest_result,
)
from quant_platform.research.bibliography import method_catalog
from quant_platform.research.decision_engine import build_research_decision_signal
from quant_platform.research.efficient_frontier import (
    capital_allocation_line,
    frontier_scatter_rows,
)
from quant_platform.research.exposure import exposure_profile, simulate_exposure_paths
from quant_platform.research.fixed_income import bond_summary
from quant_platform.research.hedging import hedge_contract_count, minimum_variance_hedge_ratio
from quant_platform.research.ml_forecasting import run_walk_forward_forecast
from quant_platform.research.monte_carlo import (
    simulate_block_bootstrap_returns,
    simulate_bootstrap_returns,
    simulate_gbm_prices,
    simulate_normal_returns,
    simulate_portfolio_normal_returns,
    summarize_simulated_paths,
)
from quant_platform.research.options import (
    binomial_crr_price,
    black_scholes_diagnostics,
    black_scholes_greeks,
    black_scholes_price,
    option_scenario_table,
    payoff_profile,
    protective_put_payoff,
    put_call_parity_gap,
)
from quant_platform.research.portfolio import compute_portfolio_analytics, portfolio_returns
from quant_platform.research.portfolio_optimization import optimize_long_only_portfolio
from quant_platform.research.rates_derivatives import (
    plain_vanilla_swap_summary,
    sofr_futures_implied_rate,
)
from quant_platform.research.returns import (
    annual_rate_to_periodic,
    close_prices_from_ohlcv,
    daily_log_returns,
    daily_simple_returns,
)
from quant_platform.research.single_asset_metrics import compute_single_asset_metrics
from quant_platform.research.var_models import compute_var_summary
from quant_platform.risk.drawdown import drawdown, max_drawdown


class QuantTerminalReportError(ValueError):
    """Raised when terminal report assembly fails."""


def build_and_write_quant_terminal_report(
    config_path: str | Path,
    settings: PlatformSettings,
    output_dir: str | Path = "reports/generated/quant_terminal",
    export_dir: str | Path = "reports/generated/portfolio_optimization",
    provider_name: str = "yfinance",
    offline_synthetic: bool = False,
) -> dict[str, Any]:
    """Build the professional terminal report and write ignored local artifacts."""

    config = load_quant_terminal_config(config_path)
    report = build_quant_terminal_report(
        config=config,
        settings=settings,
        provider_name=provider_name,
        offline_synthetic=offline_synthetic,
    )
    report_path = write_quant_terminal_report(report, output_dir)
    frontier_path = write_frontier_csv(report, export_dir)
    return {
        "report_path": str(report_path),
        "frontier_csv_path": str(frontier_path) if frontier_path is not None else None,
        "data_mode": report["data"]["mode"],
        "symbols": report["universe"]["selected_stocks"],
        "warnings": report["data"].get("warnings", []),
        "research_only": True,
    }


def build_quant_terminal_report(
    config: QuantTerminalConfig,
    settings: PlatformSettings | None = None,
    provider_name: str = "yfinance",
    offline_synthetic: bool = False,
) -> dict[str, Any]:
    """Build a complete terminal report from provider data or synthetic fallback."""

    warnings: list[str] = []
    provider_metadata: dict[str, Any] = {}
    if offline_synthetic:
        ohlcv = make_synthetic_ohlcv(config)
        mode = "offline_synthetic"
        warnings.append("SYNTHETIC_DATA_USED_FOR_OFFLINE_VALIDATION")
    else:
        if settings is None:
            raise QuantTerminalReportError("settings are required unless offline_synthetic=True.")
        try:
            provider = create_market_data_provider(provider_name, settings)
            response = provider.download_ohlcv(build_ohlcv_request(config))
            ohlcv = response.data
            mode = f"provider_{provider_name}"
            provider_metadata = dict(response.metadata)
            provider_metadata["successful_symbols"] = list(response.successful_symbols)
            provider_metadata["failed_symbols"] = dict(response.failed_symbols)
            if response.failed_symbols:
                warnings.append("PROVIDER_PARTIAL_SYMBOL_FAILURES")
            config = _select_available_universe(config, response.data, warnings)
            _validate_required_symbols_available(ohlcv, config)
        except Exception as exc:  # noqa: BLE001 - sanitized fallback for local reproducibility.
            ohlcv = make_synthetic_ohlcv(config)
            mode = "synthetic_fallback"
            warnings.append("PROVIDER_UNAVAILABLE_SYNTHETIC_FALLBACK_USED")
            warnings.append(f"PROVIDER_ERROR_SANITIZED:{_sanitize_message(exc)}")
    return build_quant_terminal_report_from_ohlcv(
        ohlcv=ohlcv,
        config=config,
        data_mode=mode,
        provider_metadata=provider_metadata,
        warnings=warnings,
    )


def build_quant_terminal_report_from_ohlcv(
    ohlcv: pd.DataFrame,
    config: QuantTerminalConfig,
    data_mode: str,
    provider_metadata: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the terminal report from an already-normalized OHLCV frame."""

    data_warnings = list(warnings or [])
    all_prices = close_prices_from_ohlcv(ohlcv)
    all_volume = _volume_from_ohlcv(ohlcv)
    required = (*config.selected_stocks, config.benchmark_symbol)
    missing = [symbol for symbol in required if symbol not in all_prices.columns]
    if missing:
        raise QuantTerminalReportError(f"Missing required symbols in OHLCV data: {missing}")
    aligned = all_prices.loc[:, list(required)].dropna(how="any")
    asset_prices = aligned.loc[:, list(config.selected_stocks)]
    benchmark_prices = aligned.loc[:, config.benchmark_symbol]
    risk_free_rate = _risk_free_rate_from_proxy(all_prices, config, data_warnings)
    asset_returns = daily_simple_returns(asset_prices)
    log_return_frame = daily_log_returns(asset_prices)
    benchmark_returns = daily_simple_returns(benchmark_prices)

    single_assets = compute_single_asset_metrics(
        asset_prices,
        benchmark_prices,
        risk_free_rate_annual=risk_free_rate,
    )
    portfolio = compute_portfolio_analytics(asset_prices, risk_free_rate)
    optimization = optimize_long_only_portfolio(
        asset_returns,
        risk_free_rate_annual=risk_free_rate,
        step=float(config.optimization.get("grid_step", 0.05)),
        max_weight=float(config.optimization.get("max_weight", 1.0)),
    )
    optimization["capital_allocation_line"] = capital_allocation_line(
        optimization["max_sharpe"],
        risk_free_rate,
    )
    max_sharpe_returns = portfolio_returns(asset_returns, optimization["max_sharpe"]["weights"])
    mc_summary = _monte_carlo_section(asset_returns, optimization["max_sharpe"]["weights"], config)
    var_summary = compute_var_summary(
        max_sharpe_returns,
        alpha=0.95,
        simulated_returns=_portfolio_mc_daily_distribution(
            asset_returns,
            optimization["max_sharpe"]["weights"],
            config,
        ),
    )
    backtests = run_terminal_backtests(
        asset_prices,
        initial_capital=float(config.backtesting.get("initial_capital", 10_000.0)),
        short_window=int(config.backtesting.get("short_window", 50)),
        long_window=int(config.backtesting.get("long_window", 200)),
        momentum_lookback_days=int(config.backtesting.get("momentum_lookback_days", 252)),
        momentum_skip_days=int(config.backtesting.get("momentum_skip_days", 21)),
    )
    if data_mode in {"synthetic_fallback", "offline_synthetic"}:
        data_warnings.append("DEMO_SYNTHETIC_NOT_REAL_DATA")
    options = _options_section(asset_prices, log_return_frame, risk_free_rate, config)
    serialized_backtests = {
        name: serialize_backtest_result(result) for name, result in backtests.items()
    }
    report_slug = _report_slug_from_universe(config.selected_stocks, config.lookback_years)
    stocks = _stocks_section(
        ohlcv=ohlcv,
        prices=asset_prices,
        volumes=all_volume,
        returns=asset_returns,
        log_returns=log_return_frame,
        benchmark_returns=benchmark_returns,
        single_assets=single_assets,
        options=options,
        risk_free_rate=risk_free_rate,
        config=config,
        data_mode=data_mode,
        data_warnings=sorted(set(data_warnings)),
    )
    metadata = {
        "report_type": "professional_quant_terminal",
        "report_version": "iteration_010_academic_stock_schema_v1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "not_investment_advice": True,
    }
    data_provenance = {
        "data_mode": data_mode,
        "provider": provider_metadata.get("provider", "synthetic_quant_terminal"),
        "source": provider_metadata.get("provider", "synthetic_quant_terminal"),
        "venue": provider_metadata.get("venue", "LOCAL_SYNTHETIC"),
        "frequency": config.frequency,
        "base_currency": config.base_currency,
        "benchmark": config.benchmark_symbol,
        "risk_free_proxy": config.risk_free_symbol,
        "risk_free_rate_annual": risk_free_rate,
        "start_timestamp": pd.Timestamp(asset_returns.index[0]).isoformat(),
        "end_timestamp": pd.Timestamp(asset_returns.index[-1]).isoformat(),
        "aligned_observations": int(len(asset_returns)),
        "provider_metadata": provider_metadata or {},
        "warnings": sorted(set(data_warnings)),
    }
    bibliography = method_catalog()
    report = {
        "metadata": metadata,
        "report_type": "professional_quant_terminal",
        "generated_at": metadata["generated_at"],
        "research_only": True,
        "data_provenance": data_provenance,
        "universe": {
            "selected_stocks": list(config.selected_stocks),
            "fallback_stocks": list(config.fallback_stocks),
            "benchmark": config.benchmark_symbol,
            "risk_free_proxy": config.risk_free_symbol,
            "frequency": config.frequency,
            "lookback_years": config.lookback_years,
            "base_currency": config.base_currency,
        },
        "data": {
            "mode": data_mode,
            "row_count": int(len(ohlcv)),
            "aligned_observations": int(len(asset_returns)),
            "start_timestamp": pd.Timestamp(asset_returns.index[0]).isoformat(),
            "end_timestamp": pd.Timestamp(asset_returns.index[-1]).isoformat(),
            "warnings": sorted(set(data_warnings)),
            "provider_metadata": provider_metadata or {},
        },
        "methods": bibliography,
        "bibliography": bibliography,
        "single_assets": single_assets,
        "stocks": stocks,
        "ml_forecasting": {
            symbol: stock.get("ml_forecasting", {}) for symbol, stock in stocks.items()
        },
        "decision_signals": {
            symbol: stock.get("decision_signal", {}) for symbol, stock in stocks.items()
        },
        "portfolio": portfolio,
        "optimization": optimization,
        "monte_carlo": mc_summary,
        "var": var_summary,
        "backtesting": serialized_backtests,
        "options": options,
        "fixed_income": bond_summary(**_fixed_income_kwargs(config)),
        "rates_derivatives": _rates_section(config),
        "hedging": _hedging_section(asset_returns, benchmark_prices, config),
        "exposure": _exposure_section(config),
        "warnings": sorted(set(data_warnings)),
        "exports": {
            "frontier_csv": f"reports/generated/portfolio_optimization/{report_slug}_frontier.csv"
        },
        "safety": {
            "no_trading": True,
            "no_broker_endpoints": True,
            "no_private_account_data": True,
            "no_secrets_serialized": True,
            "recommendation_policy": "research_analytics_only_not_financial_advice",
        },
        "limitations": [
            "Historical estimates are sample-dependent and not forecasts.",
            "Options, fixed-income, rates, hedge, and exposure blocks are educational models.",
            "Real market valuation requires curves, option chains, contract specs, "
            "and legal terms.",
        ],
    }
    return _json_safe(report)


def write_quant_terminal_report(
    report: dict[str, Any],
    output_dir: str | Path = "reports/generated/quant_terminal",
) -> Path:
    """Write the terminal JSON report to the ignored generated-report directory."""

    output_path = Path(output_dir) / f"{_report_slug(report)}_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_frontier_csv(
    report: dict[str, Any],
    output_dir: str | Path = "reports/generated/portfolio_optimization",
) -> Path | None:
    """Write efficient-frontier rows as a CSV spreadsheet equivalent."""

    rows = frontier_scatter_rows(report.get("optimization", {}))
    if not rows:
        return None
    output_path = Path(output_dir) / f"{_report_slug(report)}_frontier.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _validate_required_symbols_available(ohlcv: pd.DataFrame, config: QuantTerminalConfig) -> None:
    if ohlcv.empty or "asset_id" not in ohlcv.columns:
        raise QuantTerminalReportError("Provider returned no normalized OHLCV rows.")
    available = set(ohlcv["asset_id"].astype(str))
    required = {*config.selected_stocks, config.benchmark_symbol}
    missing = sorted(required.difference(available))
    if missing:
        raise QuantTerminalReportError(f"Provider data missing required symbols: {missing}")


def _select_available_universe(
    config: QuantTerminalConfig,
    ohlcv: pd.DataFrame,
    warnings: list[str],
) -> QuantTerminalConfig:
    """Replace failed selected stocks with available configured fallbacks."""

    available = set(ohlcv["asset_id"].astype(str)) if "asset_id" in ohlcv.columns else set()
    if config.benchmark_symbol not in available:
        raise QuantTerminalReportError(
            f"Provider data missing benchmark symbol: {config.benchmark_symbol}"
        )
    selected = [symbol for symbol in config.selected_stocks if symbol in available]
    missing = [symbol for symbol in config.selected_stocks if symbol not in available]
    if not missing:
        return config
    replacements: list[tuple[str, str]] = []
    for fallback in config.fallback_stocks:
        if len(selected) >= len(config.selected_stocks):
            break
        if fallback in available and fallback not in selected:
            replacements.append((missing[len(replacements)], fallback))
            selected.append(fallback)
    if len(selected) < len(config.selected_stocks):
        raise QuantTerminalReportError(
            "Provider data missing required symbols and insufficient fallbacks: "
            f"missing={missing}, selected_available={selected}"
        )
    warnings.append(
        "PROVIDER_SYMBOL_FALLBACK_USED:"
        + ",".join(f"{old}->{new}" for old, new in replacements)
    )
    return QuantTerminalConfig(
        selected_stocks=tuple(selected[: len(config.selected_stocks)]),
        fallback_stocks=config.fallback_stocks,
        benchmark_symbol=config.benchmark_symbol,
        risk_free_symbol=config.risk_free_symbol,
        frequency=config.frequency,
        lookback_years=config.lookback_years,
        base_currency=config.base_currency,
        risk_free_rate_annual=config.risk_free_rate_annual,
        monte_carlo=config.monte_carlo,
        optimization=config.optimization,
        backtesting=config.backtesting,
        options=config.options,
        fixed_income=config.fixed_income,
        rates_derivatives=config.rates_derivatives,
        hedging=config.hedging,
        exposure=config.exposure,
    )


def _report_slug(report: dict[str, Any]) -> str:
    universe = report.get("universe", {})
    return _report_slug_from_universe(
        universe.get("selected_stocks", ()),
        int(universe.get("lookback_years", 0) or 0),
    )


def _report_slug_from_universe(symbols: object, lookback_years: int) -> str:
    symbol_count = len(tuple(symbols)) if isinstance(symbols, (list, tuple)) else 0
    return f"{symbol_count}stocks_{lookback_years}y"


def _volume_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"asset_id", "timestamp", "volume"}
    if required.difference(ohlcv.columns):
        return pd.DataFrame()
    frame = ohlcv.loc[:, ["asset_id", "timestamp", "volume"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["asset_id", "timestamp", "volume"])
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot_table(
        index="timestamp",
        columns="asset_id",
        values="volume",
        aggfunc="last",
    ).sort_index()


def _stocks_section(
    ohlcv: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    log_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    single_assets: dict[str, dict[str, Any]],
    options: dict[str, Any],
    risk_free_rate: float,
    config: QuantTerminalConfig,
    data_mode: str,
    data_warnings: list[str],
) -> dict[str, Any]:
    rows = {}
    periodic_rf = annual_rate_to_periodic(risk_free_rate, 252)
    for position, symbol in enumerate(config.selected_stocks):
        price_series = prices[str(symbol)]
        return_series = returns[str(symbol)]
        log_return_series = log_returns[str(symbol)]
        volume_series = (
            volumes[str(symbol)].reindex(price_series.index)
            if str(symbol) in volumes.columns
            else pd.Series(0.0, index=price_series.index)
        )
        equity_curve = cumulative_returns(return_series, initial_value=1.0)
        dd_series = drawdown(equity_curve)
        rolling = _rolling_stock_metrics(
            return_series,
            benchmark_returns,
            periodic_rf,
            window=63,
        )
        stock_ohlcv = ohlcv[ohlcv["asset_id"].astype(str) == str(symbol)].copy()
        benchmark_relative = _benchmark_relative_study(
            return_series, benchmark_returns, risk_free_rate
        )
        active_cumulative = _active_cumulative_return_series(return_series, benchmark_returns)
        momentum_liquidity = _momentum_liquidity_study(price_series, return_series, volume_series)
        range_volatility = _range_volatility_study(stock_ohlcv)
        mc = _stock_monte_carlo_section(
            symbol=symbol,
            returns=return_series,
            log_returns=log_return_series,
            start_price=float(price_series.iloc[-1]),
            config=config,
            seed_offset=position,
        )
        var_summary = compute_var_summary(
            return_series,
            alpha=0.95,
            simulated_returns=_stock_mc_daily_distribution(return_series, config, position),
        )
        tail_backtesting = _tail_risk_backtesting_study(return_series)
        metrics_summary = _stock_metric_summary(
            return_series, equity_curve, single_assets[str(symbol)]
        )
        backtesting_results = _single_stock_backtest_section(return_series)
        sharpe_inference = _sharpe_inference_study(
            return_series,
            risk_free_rate=risk_free_rate,
            trial_count=_trial_count_from_ml_config(),
        )
        execution_costs = _execution_cost_study(
            price_series=price_series,
            volume_series=volume_series,
            annual_volatility=float(metrics_summary.get("annualized_volatility", 0.0)),
        )
        options_results = options.get(str(symbol), {})
        ml_results = run_walk_forward_forecast(
            price_series,
            volume=volume_series,
            benchmark_returns=benchmark_returns,
            min_train_size=756,
            max_test_observations=63,
            refit_frequency_days=21,
        )
        predictive_reliability = _predictive_reliability_audit(
            data_mode=data_mode,
            data_warnings=data_warnings,
            observations=len(return_series),
            ml_results=ml_results,
        )
        decision_signal = build_research_decision_signal(
            metrics=metrics_summary,
            var_results=var_summary,
            monte_carlo=mc,
            ml_forecasting=ml_results,
            backtesting=backtesting_results,
            data_quality_warnings=sorted(set(data_warnings)),
        )
        rows[str(symbol)] = {
            "asset_id": str(symbol),
            "data_used": {
                "ticker": str(symbol),
                "provider": "synthetic_quant_terminal"
                if data_mode in {"synthetic_fallback", "offline_synthetic"}
                else "yfinance",
                "data_mode": data_mode,
                "frequency": config.frequency,
                "currency": config.base_currency,
                "benchmark": config.benchmark_symbol,
                "risk_free_proxy": config.risk_free_symbol,
                "risk_free_rate_annual": risk_free_rate,
                "start_timestamp": pd.Timestamp(return_series.index[0]).isoformat(),
                "end_timestamp": pd.Timestamp(return_series.index[-1]).isoformat(),
                "observations": int(len(return_series)),
                "last_close": float(price_series.iloc[-1]) if price_series is not None else None,
            },
            "ohlcv_summary": _ohlcv_summary(ohlcv, str(symbol)),
            "data_quality": _stock_data_quality(ohlcv, str(symbol), price_series, volume_series),
            "price_series": _series_rows(price_series, "price"),
            "volume_series": _series_rows(volume_series, "volume"),
            "simple_returns": _series_rows(return_series, "return"),
            "log_returns": _series_rows(log_return_series, "log_return"),
            "cumulative_returns": _series_rows(equity_curve - 1.0, "cumulative_return"),
            "active_cumulative_returns": _series_rows(
                active_cumulative, "active_cumulative_return"
            ),
            "drawdown_series": _series_rows(dd_series, "drawdown"),
            "rolling_volatility": _series_rows(rolling["rolling_volatility"], "rolling_volatility"),
            "rolling_sharpe": _series_rows(rolling["rolling_sharpe"], "rolling_sharpe"),
            "rolling_beta": _series_rows(rolling["rolling_beta"], "rolling_beta"),
            "rolling_correlation": _series_rows(
                rolling["rolling_correlation"], "rolling_correlation"
            ),
            "metrics": metrics_summary,
            "benchmark_relative_study": benchmark_relative,
            "momentum_liquidity_study": momentum_liquidity,
            "range_volatility_study": range_volatility,
            "sharpe_inference_study": sharpe_inference,
            "var": var_summary,
            "tail_risk_backtesting_study": tail_backtesting,
            "monte_carlo": mc,
            "ml_forecasting": ml_results,
            "predictive_reliability_audit": predictive_reliability,
            "decision_signal": decision_signal,
            "backtesting_results": backtesting_results,
            "execution_cost_study": execution_costs,
            "literature_implementation_map": _literature_implementation_map(),
            "options_theoretical_analytics": options_results,
            "warnings": sorted(set(data_warnings)),
            "bibliography_references": method_catalog(),
        }
    return rows


def _rolling_stock_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periodic_rf: float,
    window: int,
) -> dict[str, pd.Series]:
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")
    excess = aligned_returns - periodic_rf
    rolling_vol = aligned_returns.rolling(window).std(ddof=1) * np.sqrt(252)
    rolling_sharpe = (
        excess.rolling(window).mean() / excess.rolling(window).std(ddof=1) * np.sqrt(252)
    )
    rolling_cov = aligned_returns.rolling(window).cov(aligned_benchmark)
    rolling_var = aligned_benchmark.rolling(window).var(ddof=1)
    rolling_beta = rolling_cov / rolling_var
    rolling_corr = aligned_returns.rolling(window).corr(aligned_benchmark)
    return {
        "rolling_volatility": rolling_vol.dropna(),
        "rolling_sharpe": rolling_sharpe.dropna(),
        "rolling_beta": rolling_beta.replace([np.inf, -np.inf], np.nan).dropna(),
        "rolling_correlation": rolling_corr.replace([np.inf, -np.inf], np.nan).dropna(),
    }


def _benchmark_relative_study(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float,
) -> dict[str, Any]:
    asset, bench = returns.align(benchmark_returns, join="inner")
    active = (asset - bench).dropna()
    periodic_rf = annual_rate_to_periodic(risk_free_rate, 252)
    excess_asset = asset - periodic_rf
    excess_bench = bench - periodic_rf
    tracking_error = float(active.std(ddof=1) * np.sqrt(252)) if len(active) > 1 else float("nan")
    active_return = float(active.mean() * 252) if len(active) else float("nan")
    information_ratio = active_return / tracking_error if tracking_error else float("nan")
    correlation = float(asset.corr(bench)) if len(asset) > 1 else float("nan")
    downside = bench < 0
    upside = bench > 0
    return {
        "study_name": "Benchmark-relative performance and CAPM diagnostics",
        "literature": ["FamaFrench1993", "Carhart1997", "FamaFrench2015"],
        "status": "IMPLEMENTED_WITH_DAILY_BENCHMARK_PROXY",
        "active_annualized_return": active_return,
        "tracking_error": tracking_error,
        "information_ratio": float(information_ratio),
        "correlation_to_benchmark": correlation,
        "upside_capture": _capture_ratio(asset, bench, upside),
        "downside_capture": _capture_ratio(asset, bench, downside),
        "excess_asset_mean_daily": (
            float(excess_asset.mean()) if len(excess_asset) else float("nan")
        ),
        "excess_benchmark_mean_daily": (
            float(excess_bench.mean()) if len(excess_bench) else float("nan")
        ),
        "limitations": [
            "CAPM/benchmark-relative metrics are not full Fama-French-Carhart alpha.",
            "No fundamentals, market-cap history, or point-in-time factor data are used.",
        ],
    }


def _capture_ratio(asset: pd.Series, benchmark: pd.Series, mask: pd.Series) -> float:
    selected_asset = asset.loc[mask]
    selected_benchmark = benchmark.loc[mask]
    denom = float(selected_benchmark.mean()) if len(selected_benchmark) else 0.0
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(selected_asset.mean() / denom)


def _active_cumulative_return_series(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    asset, bench = returns.align(benchmark_returns, join="inner")
    if asset.empty or bench.empty:
        return pd.Series(dtype=float)
    asset_wealth = cumulative_returns(asset, initial_value=1.0)
    benchmark_wealth = cumulative_returns(bench, initial_value=1.0)
    relative = (asset_wealth / benchmark_wealth) - 1.0
    return relative.replace([np.inf, -np.inf], np.nan).dropna()


def _momentum_liquidity_study(
    prices: pd.Series,
    returns: pd.Series,
    volume: pd.Series,
) -> dict[str, Any]:
    clean_prices = prices.astype(float).dropna()
    clean_volume = volume.astype(float).reindex(clean_prices.index).fillna(0.0)
    aligned_returns = returns.astype(float).reindex(clean_prices.index).dropna()
    dollar_volume = clean_prices * clean_volume
    amihud = (aligned_returns.abs() / dollar_volume.reindex(aligned_returns.index)).replace(
        [np.inf, -np.inf], np.nan
    )
    high_252 = clean_prices.tail(252).max() if len(clean_prices) else float("nan")
    last_price = clean_prices.iloc[-1] if len(clean_prices) else float("nan")
    return {
        "study_name": "OHLCV momentum, trend, and liquidity proxy screen",
        "literature": [
            "Carhart1997",
            "MoskowitzOoiPedersen2012",
            "EasleyLopezPradoOHara2012VolumeClock",
        ],
        "status": "DAILY_OHLCV_PROXY_NOT_FACTOR_ZOO_REPLICATION",
        "momentum_1m": _period_return(clean_prices, 21),
        "momentum_3m": _period_return(clean_prices, 63),
        "momentum_6m": _period_return(clean_prices, 126),
        "momentum_12m": _period_return(clean_prices, 252),
        "momentum_12m_skip_1m": _skip_period_return(clean_prices, 252, 21),
        "distance_to_52w_high": float(last_price / high_252 - 1.0) if high_252 else float("nan"),
        "average_volume_20d": float(clean_volume.tail(20).mean()),
        "average_volume_63d": float(clean_volume.tail(63).mean()),
        "average_dollar_volume_20d": float(dollar_volume.tail(20).mean()),
        "average_dollar_volume_63d": float(dollar_volume.tail(63).mean()),
        "amihud_illiq_mean_252d": float(amihud.tail(252).mean()),
        "limitations": [
            "Momentum and liquidity are price-volume proxies only.",
            "No market-cap, book-to-market, profitability, investment, or point-in-time "
            "fundamentals.",
        ],
    }


def _period_return(prices: pd.Series, window: int) -> float | None:
    if len(prices) <= window:
        return None
    return float(prices.iloc[-1] / prices.iloc[-window - 1] - 1.0)


def _skip_period_return(prices: pd.Series, lookback: int, skip: int) -> float | None:
    if len(prices) <= lookback:
        return None
    end = prices.iloc[-skip - 1]
    start = prices.iloc[-lookback - 1]
    return float(end / start - 1.0) if start else None


def _range_volatility_study(stock_ohlcv: pd.DataFrame) -> dict[str, Any]:
    required = {"open", "high", "low", "close"}
    if stock_ohlcv.empty or required.difference(stock_ohlcv.columns):
        return {"status": "UNAVAILABLE", "reason": "OHLC columns unavailable."}
    frame = stock_ohlcv.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame[(frame[list(required)] > 0).all(axis=1)]
    if frame.empty:
        return {"status": "UNAVAILABLE", "reason": "No positive OHLC rows."}
    log_hl = np.log(frame["high"] / frame["low"])
    log_co = np.log(frame["close"] / frame["open"])
    log_ho = np.log(frame["high"] / frame["open"])
    log_hc = np.log(frame["high"] / frame["close"])
    log_lo = np.log(frame["low"] / frame["open"])
    log_lc = np.log(frame["low"] / frame["close"])
    parkinson_var = (log_hl**2) / (4.0 * np.log(2.0))
    garman_klass_var = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    rogers_satchell_var = log_ho * log_hc + log_lo * log_lc
    return {
        "study_name": "Daily OHLC range-based volatility estimators",
        "literature": ["AndersenBollerslevDieboldLabys2003", "Corsi2009HAR"],
        "status": "DAILY_RANGE_PROXY_NOT_INTRADAY_REALIZED_VOLATILITY",
        "parkinson_volatility_annual": _annualized_var_mean(parkinson_var),
        "garman_klass_volatility_annual": _annualized_var_mean(garman_klass_var),
        "rogers_satchell_volatility_annual": _annualized_var_mean(rogers_satchell_var),
        "observations": int(len(frame)),
        "limitations": [
            "Daily OHLC range estimators are proxies, not intraday realized volatility.",
            "Microstructure-noise studies require intraday/order-book data and are not replicated.",
        ],
    }


def _annualized_var_mean(variance_series: pd.Series) -> float:
    clean = variance_series.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean >= 0]
    if clean.empty:
        return float("nan")
    return float(np.sqrt(clean.mean() * 252.0))


def _tail_risk_backtesting_study(returns: pd.Series) -> dict[str, Any]:
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    window = 252
    if len(clean) <= window + 20:
        return {
            "study_name": "VaR exception backtesting",
            "status": "NOT_SUPPORTED_INSUFFICIENT_ROLLING_HISTORY",
            "observations": int(len(clean)),
            "required_observations": window + 20,
            "loss_sign_convention": "L_t = -R_t; exceptions use L_t > VaR_alpha",
        }
    losses = -clean
    levels = {}
    exception_rows_95: list[dict[str, Any]] = []
    for alpha in (0.95, 0.99):
        var = losses.shift(1).rolling(window).quantile(alpha)
        aligned = pd.concat([losses.rename("loss"), var.rename("var")], axis=1).dropna()
        exceptions = (aligned["loss"] > aligned["var"]).astype(int)
        kupiec = _kupiec_pof_test(int(exceptions.sum()), int(len(exceptions)), 1.0 - alpha)
        christoffersen = _christoffersen_independence_test(exceptions)
        cc_lr = None
        cc_p = None
        if kupiec.get("lr_stat") is not None and christoffersen.get("lr_stat") is not None:
            cc_lr = float(kupiec["lr_stat"] + christoffersen["lr_stat"])
            cc_p = _chi_square_sf(cc_lr, 2)
        levels[f"alpha_{int(alpha * 100)}"] = {
            "alpha": float(alpha),
            "rolling_window_days": window,
            "observations": int(len(exceptions)),
            "exceptions": int(exceptions.sum()),
            "expected_exceptions": float(len(exceptions) * (1.0 - alpha)),
            "exception_rate": float(exceptions.mean()) if len(exceptions) else None,
            "kupiec_pof": kupiec,
            "christoffersen_independence": christoffersen,
            "conditional_coverage_lr_stat": cc_lr,
            "conditional_coverage_p_value": cc_p,
            "conditional_coverage_status": _backtest_status(cc_p),
        }
        if alpha == 0.95:
            exception_rows_95 = [
                {
                    "timestamp": pd.Timestamp(index).isoformat(),
                    "loss": float(row["loss"]),
                    "rolling_historical_var": float(row["var"]),
                    "exception": bool(exceptions.loc[index]),
                }
                for index, row in aligned.iterrows()
            ]
    return {
        "study_name": "VaR exception backtesting with Kupiec and Christoffersen diagnostics",
        "literature": ["Kupiec1995", "Christoffersen1998", "AcerbiTasche2002"],
        "status": "IMPLEMENTED_ROLLING_HISTORICAL_VAR_BACKTEST",
        "loss_sign_convention": "L_t = -R_t; exceptions use L_t > VaR_alpha",
        "levels": levels,
        "rolling_exception_rows_95": exception_rows_95,
        "limitations": [
            "VaR thresholds are rolling historical estimates from daily returns only.",
            "Expected Shortfall is reported, but formal ES elicitability/backtesting is "
            "not replicated.",
            "No intraday liquidity, bid-ask, or stressed market microstructure data are used.",
        ],
    }


def _kupiec_pof_test(exceptions: int, observations: int, expected_prob: float) -> dict[str, Any]:
    if observations <= 0 or expected_prob <= 0.0 or expected_prob >= 1.0:
        return {"status": "UNAVAILABLE", "lr_stat": None, "p_value": None}
    observed_prob = exceptions / observations
    restricted = _bernoulli_log_likelihood(exceptions, observations, expected_prob)
    unrestricted = _bernoulli_log_likelihood(exceptions, observations, observed_prob)
    lr_stat = max(0.0, -2.0 * (restricted - unrestricted))
    p_value = _chi_square_sf(lr_stat, 1)
    return {
        "test": "Kupiec proportion-of-failures",
        "lr_stat": float(lr_stat),
        "p_value": p_value,
        "status": _backtest_status(p_value),
    }


def _christoffersen_independence_test(exceptions: pd.Series) -> dict[str, Any]:
    values = exceptions.astype(int).to_numpy(dtype=int)
    if len(values) < 2:
        return {"status": "UNAVAILABLE", "lr_stat": None, "p_value": None}
    previous = values[:-1]
    current = values[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    total = n00 + n01 + n10 + n11
    if total <= 0:
        return {"status": "UNAVAILABLE", "lr_stat": None, "p_value": None}
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    restricted = _transition_log_likelihood(n00, n01, n10, n11, pi, pi)
    unrestricted = _transition_log_likelihood(n00, n01, n10, n11, pi01, pi11)
    lr_stat = max(0.0, -2.0 * (restricted - unrestricted))
    p_value = _chi_square_sf(lr_stat, 1)
    return {
        "test": "Christoffersen independence",
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_stat": float(lr_stat),
        "p_value": p_value,
        "status": _backtest_status(p_value),
    }


def _bernoulli_log_likelihood(successes: int, observations: int, probability: float) -> float:
    probability = _clip_probability(probability)
    failures = observations - successes
    return successes * math.log(probability) + failures * math.log1p(-probability)


def _transition_log_likelihood(
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    pi01: float,
    pi11: float,
) -> float:
    pi01 = _clip_probability(pi01)
    pi11 = _clip_probability(pi11)
    return (
        n00 * math.log1p(-pi01)
        + n01 * math.log(pi01)
        + n10 * math.log1p(-pi11)
        + n11 * math.log(pi11)
    )


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-12), 1.0 - 1e-12)


def _chi_square_sf(value: float | None, df: int) -> float | None:
    if value is None or not np.isfinite(value) or value < 0.0:
        return None
    if df == 1:
        return float(math.erfc(math.sqrt(value / 2.0)))
    if df == 2:
        return float(math.exp(-value / 2.0))
    return None


def _backtest_status(p_value: float | None) -> str:
    if p_value is None:
        return "UNAVAILABLE"
    return "REJECTS_MODEL_AT_5PCT" if p_value < 0.05 else "DOES_NOT_REJECT_AT_5PCT"


def _sharpe_inference_study(
    returns: pd.Series,
    risk_free_rate: float,
    trial_count: int,
) -> dict[str, Any]:
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    periodic_rf = annual_rate_to_periodic(risk_free_rate, 252)
    excess = clean - periodic_rf
    n = int(len(excess))
    if n < 30 or float(excess.std(ddof=1)) == 0.0:
        return {
            "study_name": "Sharpe inference and deflated Sharpe approximation",
            "status": "NOT_SUPPORTED_INSUFFICIENT_OR_ZERO_VARIANCE_SAMPLE",
            "observations": n,
        }
    mean_excess = float(excess.mean())
    sigma = float(excess.std(ddof=1))
    sharpe_daily = mean_excess / sigma
    sharpe_annual = sharpe_daily * math.sqrt(252.0)
    skew = float(excess.skew())
    kurtosis = float(excess.kurtosis() + 3.0)
    denominator = 1.0 - skew * sharpe_daily + ((kurtosis - 1.0) / 4.0) * sharpe_daily**2
    denominator = max(denominator, 1e-12)
    sr_standard_error_daily = math.sqrt(denominator / max(n - 1, 1))
    normal = NormalDist()
    z_zero = sharpe_daily / sr_standard_error_daily
    psr_zero = normal.cdf(z_zero)
    safe_trials = max(int(trial_count), 1)
    multiple_testing_threshold_daily = normal.inv_cdf(1.0 - 1.0 / (safe_trials + 1.0)) * math.sqrt(
        (1.0 + 0.5 * sharpe_daily**2) / max(n - 1, 1)
    )
    z_deflated = (sharpe_daily - multiple_testing_threshold_daily) / sr_standard_error_daily
    deflated_probability = normal.cdf(z_deflated)
    return {
        "study_name": "Sharpe inference, probabilistic Sharpe ratio, and deflated Sharpe proxy",
        "literature": ["Lo2002", "BaileyLopezDePrado2014", "HarveyLiuZhu2016"],
        "status": "IMPLEMENTED_AS_DAILY_RETURN_INFERENCE_APPROXIMATION",
        "observations": n,
        "annualized_sharpe": float(sharpe_annual),
        "daily_sharpe": float(sharpe_daily),
        "sharpe_standard_error_annualized": float(sr_standard_error_daily * math.sqrt(252.0)),
        "probabilistic_sharpe_ratio_gt_zero": float(psr_zero),
        "multiple_testing_trial_count": safe_trials,
        "deflated_sharpe_threshold_annualized": float(
            multiple_testing_threshold_daily * math.sqrt(252.0)
        ),
        "deflated_sharpe_probability": float(deflated_probability),
        "skewness": skew,
        "kurtosis": kurtosis,
        "limitations": [
            "Deflated Sharpe is approximated from configured model-search breadth, not a "
            "full strategy zoo.",
            "Serial-correlation and non-stationarity adjustments are limited with daily "
            "close data.",
            "Inference describes historical risk-adjusted returns, not tradable alpha.",
        ],
    }


def _trial_count_from_ml_config() -> int:
    return 7


def _execution_cost_study(
    price_series: pd.Series,
    volume_series: pd.Series,
    annual_volatility: float,
) -> dict[str, Any]:
    prices = price_series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    volumes = volume_series.astype(float).reindex(prices.index).replace([np.inf, -np.inf], np.nan)
    dollar_volume = (prices * volumes).dropna()
    adv20 = float(volumes.tail(20).mean()) if len(volumes.dropna()) else float("nan")
    adv_dollar20 = float(dollar_volume.tail(20).mean()) if len(dollar_volume) else float("nan")
    if not np.isfinite(adv_dollar20) or adv_dollar20 <= 0.0:
        return {
            "study_name": "Execution cost and slippage scenario analysis",
            "status": "NOT_SUPPORTED_VOLUME_OR_DOLLAR_VOLUME_UNAVAILABLE",
            "limitations": ["No reliable volume proxy for cost scenarios."],
        }
    scenarios = []
    for participation, spread_bps in ((0.001, 2.5), (0.01, 5.0), (0.05, 10.0)):
        impact_bps = max(0.0, float(annual_volatility)) * math.sqrt(participation) * 10000.0 * 0.10
        total_bps = spread_bps + impact_bps
        scenarios.append(
            {
                "participation_rate_of_adv": participation,
                "hypothetical_notional": adv_dollar20 * participation,
                "assumed_half_spread_bps": spread_bps,
                "square_root_impact_bps": float(impact_bps),
                "total_one_way_cost_bps": float(total_bps),
                "cost_pct": float(total_bps / 10000.0),
                "cost_per_100k_notional": float(100000.0 * total_bps / 10000.0),
            }
        )
    return {
        "study_name": "Execution cost and slippage scenario analysis",
        "literature": ["AlmgrenChriss2001", "Gatheral2010"],
        "status": "HYPOTHETICAL_DAILY_ADV_SCENARIO_NOT_REAL_EXECUTION_MODEL",
        "average_daily_volume_20d": adv20,
        "average_daily_dollar_volume_20d": adv_dollar20,
        "annualized_volatility_input": float(annual_volatility),
        "scenario_rows": scenarios,
        "limitations": [
            "No bid-ask spread, order book, venue, intraday volume curve, or broker fill "
            "data are used.",
            "Rows are cost-sensitivity scenarios only; they are not orders, sizing, or "
            "execution advice.",
            "Square-root impact is a stylized approximation, not a calibrated "
            "Almgren-Chriss implementation.",
        ],
    }


def _literature_implementation_map() -> list[dict[str, Any]]:
    return [
        {
            "research_family": "Machine-learning asset pricing",
            "implemented_status": "proxy_implemented",
            "report_block": "ml_forecasting",
            "what_is_supported": (
                "Walk-forward daily-return forecasting with leakage controls and naive "
                "baselines."
            ),
            "what_is_not_supported": (
                "Large cross-section SDF/IPCA/deep asset pricing replication."
            ),
            "references": ["GuKellyXiu2020", "ChenPelgerZhu2024"],
        },
        {
            "research_family": "Multiple testing and overfitting control",
            "implemented_status": "diagnostic_proxy_implemented",
            "report_block": "sharpe_inference_study",
            "what_is_supported": (
                "Probabilistic Sharpe and approximate deflated Sharpe from daily returns."
            ),
            "what_is_not_supported": (
                "Full strategy-zoo trial reconstruction or page-level replication audit."
            ),
            "references": ["BaileyLopezDePrado2014", "HarveyLiuZhu2016"],
        },
        {
            "research_family": "Factor alpha and benchmark controls",
            "implemented_status": "benchmark_proxy_implemented",
            "report_block": "benchmark_relative_study",
            "what_is_supported": (
                "CAPM/benchmark-relative active return, tracking error, capture, beta "
                "diagnostics."
            ),
            "what_is_not_supported": (
                "Official Fama-French-Carhart factor regression with point-in-time "
                "fundamentals."
            ),
            "references": ["FamaFrench1993", "Carhart1997", "FamaFrench2015"],
        },
        {
            "research_family": "Volatility and microstructure",
            "implemented_status": "daily_range_proxy_implemented",
            "report_block": "range_volatility_study",
            "what_is_supported": "Daily OHLC range-volatility proxies.",
            "what_is_not_supported": (
                "Intraday realized volatility, VPIN, order-flow imbalance, or "
                "market-microstructure replication."
            ),
            "references": ["AndersenBollerslevDieboldLabys2003", "Corsi2009HAR"],
        },
        {
            "research_family": "Tail risk and VaR/ES",
            "implemented_status": "rolling_var_backtest_implemented",
            "report_block": "tail_risk_backtesting_study",
            "what_is_supported": "Historical VaR/ES with rolling exception tests for VaR.",
            "what_is_not_supported": (
                "Full EVT/POT calibration or formal ES regulatory backtesting."
            ),
            "references": [
                "Artzner1999",
                "AcerbiTasche2002",
                "Kupiec1995",
                "Christoffersen1998",
            ],
        },
        {
            "research_family": "Execution costs and market impact",
            "implemented_status": "scenario_proxy_implemented",
            "report_block": "execution_cost_study",
            "what_is_supported": "Daily ADV-based spread/impact sensitivity scenarios.",
            "what_is_not_supported": (
                "Broker fills, live order routing, intraday schedules, or calibrated "
                "market-impact execution."
            ),
            "references": ["AlmgrenChriss2001", "Gatheral2010"],
        },
        {
            "research_family": "Meta-labeling and event bars",
            "implemented_status": "not_supported_with_current_data",
            "report_block": "literature_implementation_map",
            "what_is_supported": "The limitation is explicitly disclosed.",
            "what_is_not_supported": (
                "Triple-barrier labeling, dollar bars, and intraday event-driven "
                "meta-labeling."
            ),
            "references": ["LopezDePrado2018"],
        },
    ]


def _stock_monte_carlo_section(
    symbol: str,
    returns: pd.Series,
    log_returns: pd.Series,
    start_price: float,
    config: QuantTerminalConfig,
    seed_offset: int,
) -> dict[str, Any]:
    mc_config = config.monte_carlo
    horizon = int(mc_config.get("horizon_days", 252))
    n_paths = int(mc_config.get("n_paths", 1000))
    seed = int(mc_config.get("seed", 42)) + seed_offset
    block_size = int(mc_config.get("block_size", 20))
    bootstrap = simulate_bootstrap_returns(returns, horizon, n_paths, seed)
    block = simulate_block_bootstrap_returns(returns, horizon, n_paths, block_size, seed)
    normal = simulate_normal_returns(returns, horizon, n_paths, seed)
    gbm_prices = simulate_gbm_prices(start_price, log_returns, horizon, n_paths, seed)
    return {
        "symbol": symbol,
        "horizon_days": horizon,
        "n_paths": n_paths,
        "seed": seed,
        "historical_bootstrap": _return_path_model_summary(bootstrap, "historical_bootstrap"),
        "block_bootstrap": _return_path_model_summary(block, "block_bootstrap"),
        "parametric_normal": _return_path_model_summary(normal, "parametric_normal"),
        "gbm_baseline": _price_path_model_summary(gbm_prices, start_price, "gbm_baseline"),
        "model_status": "PARAMETRIC_OR_BOOTSTRAP_SIMULATION",
        "warning": "Simulation is not a prediction.",
    }


def _return_path_model_summary(paths: np.ndarray, model: str) -> dict[str, Any]:
    wealth = np.cumprod(1.0 + paths, axis=1)
    terminal_returns = wealth[:, -1] - 1.0
    summary = summarize_simulated_paths(paths, start_value=1.0)
    summary.update(
        {
            "model": model,
            "paths_sample": _path_sample_rows(wealth),
            "terminal_distribution": [float(value) for value in terminal_returns],
            "probability_of_loss": float(np.mean(terminal_returns < 0.0)),
        }
    )
    return summary


def _price_path_model_summary(paths: np.ndarray, start_price: float, model: str) -> dict[str, Any]:
    terminal_returns = paths[:, -1] / start_price - 1.0
    percentiles = [5, 25, 50, 75, 95]
    fan = np.percentile(paths, percentiles, axis=0)
    return {
        "model": model,
        "path_count": int(paths.shape[0]),
        "horizon_days": int(paths.shape[1]),
        "terminal_mean": float(np.mean(terminal_returns)),
        "terminal_median": float(np.median(terminal_returns)),
        "terminal_p05": float(np.percentile(terminal_returns, 5)),
        "terminal_p95": float(np.percentile(terminal_returns, 95)),
        "probability_of_loss": float(np.mean(terminal_returns < 0.0)),
        "fan_chart": [
            {
                "step": int(step + 1),
                **{
                    f"p{percentile}": float(fan[index, step])
                    for index, percentile in enumerate(percentiles)
                },
            }
            for step in range(paths.shape[1])
        ],
        "paths_sample": _path_sample_rows(paths),
        "terminal_distribution": [float(value) for value in terminal_returns],
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
    }


def _path_sample_rows(paths: np.ndarray, max_paths: int = 25) -> list[dict[str, float | int]]:
    clean = np.asarray(paths, dtype=float)
    rows = []
    for path_id in range(min(max_paths, clean.shape[0])):
        for step in range(clean.shape[1]):
            rows.append(
                {
                    "path_id": int(path_id),
                    "step": int(step + 1),
                    "value": float(clean[path_id, step]),
                }
            )
    return rows


def _stock_metric_summary(
    returns: pd.Series,
    equity_curve: pd.Series,
    base_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        **{key: value for key, value in base_metrics.items() if key != "series"},
        "cagr": annualized_return(returns, 252),
        "calmar_ratio": calmar_ratio(returns, 252),
        "hit_rate": hit_rate(returns),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis() + 3.0),
        "final_cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "max_drawdown": float(max_drawdown(equity_curve)),
    }


def _single_stock_backtest_section(returns: pd.Series) -> dict[str, Any]:
    equity_curve = cumulative_returns(returns, initial_value=10_000.0)
    return {
        "buy_and_hold": {
            "metrics": {
                "annualized_return": annualized_return(returns, 252),
                "annualized_volatility": annualized_volatility(returns, 252),
                "sharpe_ratio": sharpe_ratio(returns, periods_per_year=252),
                "sortino_ratio": sortino_ratio(returns, periods_per_year=252),
                "calmar_ratio": calmar_ratio(returns, 252),
                "hit_rate": hit_rate(returns),
                "max_drawdown": float(max_drawdown(equity_curve)),
                "final_equity": float(equity_curve.iloc[-1]),
            },
            "equity_curve": _series_rows(equity_curve, "equity"),
            "drawdown_curve": _series_rows(drawdown(equity_curve), "drawdown"),
            "assumption": "single_stock_buy_and_hold_no_real_execution",
        }
    }


def _predictive_reliability_audit(
    *,
    data_mode: str,
    data_warnings: list[str],
    observations: int,
    ml_results: dict[str, Any],
) -> dict[str, Any]:
    warnings = [str(item) for item in data_warnings]
    da = _float_or_none(ml_results.get("directional_accuracy"))
    baseline_da = _float_or_none(ml_results.get("baseline_directional_accuracy"))
    da_edge = _float_or_none(ml_results.get("directional_accuracy_edge_vs_naive"))
    if da_edge is None and da is not None and baseline_da is not None:
        da_edge = da - baseline_da
    rmse = _float_or_none(ml_results.get("rmse"))
    baseline_rmse = _float_or_none(ml_results.get("baseline_rmse"))
    oos_r2 = _float_or_none(ml_results.get("oos_r_squared"))
    ic = _float_or_none(ml_results.get("information_coefficient"))
    gates = _mapping(ml_results.get("approval_gates"))
    criteria = [
        _audit_criterion(
            "provider_data_not_synthetic",
            data_mode.startswith("provider_"),
            data_mode,
            "provider_*",
        ),
        _audit_criterion(
            "risk_free_proxy_available",
            not any(item.startswith("RISK_FREE_PROXY_") for item in warnings),
            ",".join(item for item in warnings if item.startswith("RISK_FREE_PROXY_"))
            or "available",
            "no risk-free proxy warning",
        ),
        _audit_criterion("minimum_test_history", observations >= 756, observations, ">=756"),
        _audit_criterion(
            "rmse_improves_naive",
            bool(
                gates.get(
                    "rmse_improves_naive",
                    rmse is not None and baseline_rmse is not None and rmse < baseline_rmse,
                )
            ),
            _audit_delta(baseline_rmse, rmse),
            "model RMSE < naive RMSE",
        ),
        _audit_criterion(
            "directional_accuracy_edge_ge_2pct",
            bool(
                gates.get(
                    "directional_accuracy_edge_ge_2pct",
                    da_edge is not None and da_edge >= 0.02,
                )
            ),
            da_edge,
            ">=0.02",
        ),
        _audit_criterion(
            "directional_accuracy_ge_52pct",
            bool(gates.get("directional_accuracy_ge_52pct", da is not None and da >= 0.52)),
            da,
            ">=0.52",
        ),
        _audit_criterion(
            "oos_r_squared_positive",
            bool(gates.get("oos_r_squared_positive", oos_r2 is not None and oos_r2 > 0.0)),
            oos_r2,
            ">0",
        ),
        _audit_criterion(
            "information_coefficient_positive",
            bool(gates.get("information_coefficient_positive", ic is not None and ic > 0.0)),
            ic,
            ">0",
        ),
    ]
    failed = [item["criterion"] for item in criteria if not item["passed"]]
    hard_data_failure = any(
        item in failed for item in {"provider_data_not_synthetic", "minimum_test_history"}
    )
    if hard_data_failure:
        rating = "RED_INVALID_FOR_PREDICTIVE_USE"
    elif failed:
        rating = "AMBER_NO_VALIDATED_PREDICTIVE_EDGE"
    else:
        rating = "GREEN_STRICT_RESEARCH_EDGE_DIAGNOSTICS_PASSED"
    return {
        "rating": rating,
        "strict_predictive_edge_validated": rating.startswith("GREEN_"),
        "criteria": criteria,
        "failed_criteria": failed,
        "status_reason": str(ml_results.get("status_reason", "")),
        "interpretation": _predictive_reliability_interpretation(rating, failed),
    }


def _audit_criterion(
    criterion: str,
    passed: bool,
    observed: object,
    required: object,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _audit_delta(baseline: float | None, model: float | None) -> float | None:
    if baseline is None or model is None:
        return None
    return float(baseline - model)


def _predictive_reliability_interpretation(rating: str, failed: list[str]) -> str:
    if rating.startswith("GREEN_"):
        return (
            "Strict predictive diagnostics passed under the configured walk-forward test; "
            "this remains research-only and not a live trading approval."
        )
    if rating.startswith("RED_"):
        return "Predictive use is invalidated by data or sample-quality failures."
    return "No validated predictive edge; failed criteria: " + ", ".join(failed)


def _float_or_none(value: object) -> float | None:
    try:
        clean = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return clean if np.isfinite(clean) else None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ohlcv_summary(ohlcv: pd.DataFrame, symbol: str) -> dict[str, Any]:
    symbol_rows = ohlcv[ohlcv["asset_id"].astype(str) == symbol].copy()
    if symbol_rows.empty:
        return {"asset_id": symbol, "row_count": 0}
    timestamps = pd.to_datetime(symbol_rows["timestamp"], utc=True, errors="coerce")
    close = pd.to_numeric(symbol_rows["close"], errors="coerce")
    volume = pd.to_numeric(symbol_rows["volume"], errors="coerce")
    return {
        "asset_id": symbol,
        "row_count": int(len(symbol_rows)),
        "start_timestamp": timestamps.min().isoformat(),
        "end_timestamp": timestamps.max().isoformat(),
        "open_min": float(pd.to_numeric(symbol_rows["open"], errors="coerce").min()),
        "high_max": float(pd.to_numeric(symbol_rows["high"], errors="coerce").max()),
        "low_min": float(pd.to_numeric(symbol_rows["low"], errors="coerce").min()),
        "close_min": float(close.min()),
        "close_max": float(close.max()),
        "average_volume": float(volume.mean()),
        "currency": str(symbol_rows["currency"].iloc[0]) if "currency" in symbol_rows else "USD",
        "source": str(symbol_rows["source"].iloc[0]) if "source" in symbol_rows else "unknown",
    }


def _stock_data_quality(
    ohlcv: pd.DataFrame,
    symbol: str,
    prices: pd.Series,
    volume: pd.Series,
) -> dict[str, Any]:
    symbol_rows = ohlcv[ohlcv["asset_id"].astype(str) == symbol]
    duplicate_count = int(symbol_rows.duplicated(subset=["asset_id", "timestamp"]).sum())
    return {
        "missing_prices": int(prices.isna().sum()),
        "missing_volume": int(volume.isna().sum()),
        "duplicate_timestamp_rows": duplicate_count,
        "non_positive_prices": int((prices <= 0).sum()),
        "zero_volume_rows": int((volume == 0).sum()),
        "gaps_note": "Calendar-aware gap classification is not a professional market-data audit.",
    }


def _series_rows(series: pd.Series, value_name: str) -> list[dict[str, float | str]]:
    clean = series.astype(float).dropna()
    return [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), value_name: float(value)}
        for timestamp, value in clean.items()
    ]


def _risk_free_rate_from_proxy(
    all_prices: pd.DataFrame,
    config: QuantTerminalConfig,
    warnings: list[str],
) -> float:
    if config.risk_free_symbol not in all_prices.columns:
        if config.risk_free_rate_annual == 0.0:
            warnings.append("RISK_FREE_PROXY_UNAVAILABLE_USING_ZERO_RATE")
        return float(config.risk_free_rate_annual)
    proxy = all_prices[config.risk_free_symbol].dropna()
    if proxy.empty:
        warnings.append("RISK_FREE_PROXY_EMPTY_USING_CONFIG_RATE")
        return float(config.risk_free_rate_annual)
    last_value = float(proxy.iloc[-1])
    if last_value > 1.0:
        warnings.append("RISK_FREE_PROXY_CLOSE_USED_AS_PERCENT_ANNUAL_RATE")
        return last_value / 100.0
    warnings.append("RISK_FREE_PROXY_CLOSE_USED_AS_DECIMAL_ANNUAL_RATE")
    return last_value


def _monte_carlo_section(
    returns: pd.DataFrame,
    weights: dict[str, float],
    config: QuantTerminalConfig,
) -> dict[str, Any]:
    mc_config = config.monte_carlo
    paths = simulate_portfolio_normal_returns(
        returns,
        weights,
        horizon_days=int(mc_config.get("horizon_days", 252)),
        n_paths=int(mc_config.get("n_paths", 1000)),
        seed=int(mc_config.get("seed", 42)),
    )
    summary = summarize_simulated_paths(paths, start_value=1.0)
    summary["model"] = "correlated_normal_portfolio_returns"
    summary["assumption"] = "PARAMETRIC_EDUCATIONAL_MODEL"
    summary["terminal_returns"] = [float(value) for value in np.prod(1.0 + paths, axis=1) - 1.0]
    return summary


def _portfolio_mc_daily_distribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
    config: QuantTerminalConfig,
) -> pd.Series:
    mc_config = config.monte_carlo
    paths = simulate_portfolio_normal_returns(
        returns,
        weights,
        horizon_days=int(mc_config.get("horizon_days", 252)),
        n_paths=int(mc_config.get("n_paths", 1000)),
        seed=int(mc_config.get("seed", 42)),
    )
    return pd.Series(paths.reshape(-1), dtype=float)


def _stock_mc_daily_distribution(
    returns: pd.Series,
    config: QuantTerminalConfig,
    seed_offset: int,
) -> pd.Series:
    mc_config = config.monte_carlo
    paths = simulate_normal_returns(
        returns,
        horizon_days=int(mc_config.get("horizon_days", 252)),
        n_paths=int(mc_config.get("n_paths", 1000)),
        seed=int(mc_config.get("seed", 42)) + seed_offset,
    )
    return pd.Series(paths.reshape(-1), dtype=float)


def _options_section(
    asset_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    risk_free_rate: float,
    config: QuantTerminalConfig,
) -> dict[str, Any]:
    maturity = float(config.options.get("maturity_years", 1.0))
    dividend_yield = float(config.options.get("dividend_yield", 0.0))
    steps = int(config.options.get("binomial_steps", 100))
    rows = {}
    for symbol in asset_prices.columns:
        spot = float(asset_prices[str(symbol)].iloc[-1])
        strike = spot
        volatility = float(log_returns[str(symbol)].std(ddof=1) * np.sqrt(252))
        volatility = max(volatility, 1e-6)
        call = black_scholes_price(
            spot,
            strike,
            risk_free_rate,
            volatility,
            maturity,
            "call",
            dividend_yield,
        )
        put = black_scholes_price(
            spot,
            strike,
            risk_free_rate,
            volatility,
            maturity,
            "put",
            dividend_yield,
        )
        rows[str(symbol)] = {
            "spot": spot,
            "strike": strike,
            "volatility": volatility,
            "black_scholes_call": call,
            "black_scholes_put": put,
            "call_diagnostics": black_scholes_diagnostics(
                spot,
                strike,
                risk_free_rate,
                volatility,
                maturity,
                "call",
                dividend_yield,
            ),
            "put_diagnostics": black_scholes_diagnostics(
                spot,
                strike,
                risk_free_rate,
                volatility,
                maturity,
                "put",
                dividend_yield,
            ),
            "binomial_call": binomial_crr_price(
                spot, strike, risk_free_rate, volatility, maturity, steps, "call", dividend_yield
            ),
            "binomial_put": binomial_crr_price(
                spot, strike, risk_free_rate, volatility, maturity, steps, "put", dividend_yield
            ),
            "call_greeks": black_scholes_greeks(
                spot,
                strike,
                risk_free_rate,
                volatility,
                maturity,
                "call",
                dividend_yield,
            ),
            "put_call_parity_gap": put_call_parity_gap(
                call,
                put,
                spot,
                strike,
                risk_free_rate,
                maturity,
                dividend_yield,
            ),
            "payoff_profile": payoff_profile(spot, strike, "call"),
            "protective_put_payoff": protective_put_payoff(spot, strike, put),
            "scenario_table": option_scenario_table(
                spot,
                risk_free_rate,
                volatility,
                maturity,
                dividend_yield,
                steps,
            ),
            "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
            "real_market_status": "DATA_REQUIRED_FOR_REAL_MARKET_VALUATION",
        }
    return rows


def _fixed_income_kwargs(config: QuantTerminalConfig) -> dict[str, Any]:
    defaults = {
        "face_value": 1000.0,
        "coupon_rate": 0.04,
        "yield_to_maturity": 0.045,
        "maturity_years": 5.0,
        "frequency": 2,
        "yield_bump": 0.0001,
    }
    defaults.update(config.fixed_income)
    return defaults


def _rates_section(config: QuantTerminalConfig) -> dict[str, Any]:
    defaults = {
        "notional": 1_000_000.0,
        "fixed_rate": 0.04,
        "floating_forward_rate": 0.038,
        "discount_rate": 0.035,
        "maturity_years": 5.0,
        "payments_per_year": 2,
        "sofr_futures_price": 95.25,
    }
    defaults.update(config.rates_derivatives)
    sofr_price = float(defaults.pop("sofr_futures_price"))
    return {
        "swap": plain_vanilla_swap_summary(**defaults),
        "sofr_futures": sofr_futures_implied_rate(sofr_price),
    }


def _hedging_section(
    asset_returns: pd.DataFrame,
    benchmark_prices: pd.Series,
    config: QuantTerminalConfig,
) -> dict[str, Any]:
    benchmark_returns = daily_simple_returns(benchmark_prices)
    ratio = minimum_variance_hedge_ratio(asset_returns.iloc[:, 0], benchmark_returns)
    contract = hedge_contract_count(
        exposure_value=float(config.hedging.get("exposure_value", 1_000_000.0)),
        futures_contract_value=float(config.hedging.get("futures_contract_value", 200_000.0)),
        hedge_ratio=float(ratio["hedge_ratio"]),
    )
    return {"minimum_variance": ratio, "contract_estimate": contract}


def _exposure_section(config: QuantTerminalConfig) -> dict[str, Any]:
    exposure_config = config.exposure
    paths = simulate_exposure_paths(
        n_paths=int(exposure_config.get("n_paths", 1000)),
        n_steps=int(exposure_config.get("n_steps", 20)),
        initial_exposure=float(exposure_config.get("initial_exposure", 0.0)),
        drift=float(exposure_config.get("drift", 0.0)),
        volatility=float(exposure_config.get("volatility", 100_000.0)),
        seed=int(exposure_config.get("seed", 42)),
    )
    return exposure_profile(paths, confidence=0.95)


def _sanitize_message(exc: Exception) -> str:
    message = str(exc)[:240]
    for marker in ("apikey", "api_key", "apiKey", "token", "secret", "key"):
        message = message.replace(marker, "credential_param")
    return message


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
