"""Professional quant terminal report assembly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
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
            "frontier_csv": "reports/generated/portfolio_optimization/3stocks_frontier.csv"
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

    output_path = Path(output_dir) / "3stocks_10y_report.json"
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
    output_path = Path(output_dir) / "3stocks_frontier.csv"
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
        metrics_summary = _stock_metric_summary(
            return_series, equity_curve, single_assets[str(symbol)]
        )
        backtesting_results = _single_stock_backtest_section(return_series)
        options_results = options.get(str(symbol), {})
        ml_results = run_walk_forward_forecast(
            price_series,
            volume=volume_series,
            benchmark_returns=benchmark_returns,
            min_train_size=756,
            max_test_observations=63,
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
            "drawdown_series": _series_rows(dd_series, "drawdown"),
            "rolling_volatility": _series_rows(rolling["rolling_volatility"], "rolling_volatility"),
            "rolling_sharpe": _series_rows(rolling["rolling_sharpe"], "rolling_sharpe"),
            "rolling_beta": _series_rows(rolling["rolling_beta"], "rolling_beta"),
            "metrics": metrics_summary,
            "var": var_summary,
            "monte_carlo": mc,
            "ml_forecasting": ml_results,
            "decision_signal": decision_signal,
            "backtesting_results": backtesting_results,
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
    return {
        "rolling_volatility": rolling_vol.dropna(),
        "rolling_sharpe": rolling_sharpe.dropna(),
        "rolling_beta": rolling_beta.replace([np.inf, -np.inf], np.nan).dropna(),
    }


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
