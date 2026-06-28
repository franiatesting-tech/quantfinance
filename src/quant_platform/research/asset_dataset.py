"""Dataset and configuration helpers for the professional quant terminal."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.config.simple_yaml import load_simple_yaml
from quant_platform.data.providers.base import OHLCVRequest
from quant_platform.data.schemas import MarketType


class QuantTerminalConfigError(ValueError):
    """Raised when terminal configuration is invalid."""


@dataclass(frozen=True)
class QuantTerminalConfig:
    """Validated local config for the 3-stock professional terminal."""

    selected_stocks: tuple[str, ...]
    fallback_stocks: tuple[str, ...]
    benchmark_symbol: str
    risk_free_symbol: str
    frequency: str = "1d"
    lookback_years: int = 10
    base_currency: str = "USD"
    risk_free_rate_annual: float = 0.0
    monte_carlo: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    backtesting: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    fixed_income: dict[str, Any] = field(default_factory=dict)
    rates_derivatives: dict[str, Any] = field(default_factory=dict)
    hedging: dict[str, Any] = field(default_factory=dict)
    exposure: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.selected_stocks) < 2:
            raise QuantTerminalConfigError("selected_stocks must contain at least 2 symbols.")
        if len(set(self.selected_stocks)) != len(self.selected_stocks):
            raise QuantTerminalConfigError("selected_stocks must be unique.")
        if self.frequency != "1d":
            raise QuantTerminalConfigError("Only daily frequency '1d' is supported.")
        if self.lookback_years < 1:
            raise QuantTerminalConfigError("lookback_years must be >= 1.")
        if not self.benchmark_symbol:
            raise QuantTerminalConfigError("benchmark symbol must not be empty.")
        if not self.risk_free_symbol:
            raise QuantTerminalConfigError("risk-free symbol must not be empty.")
        if not self.base_currency:
            raise QuantTerminalConfigError("base_currency must not be empty.")
        if not np.isfinite(self.risk_free_rate_annual):
            raise QuantTerminalConfigError("risk_free_rate_annual must be finite.")

    @property
    def market_symbols(self) -> tuple[str, ...]:
        """Symbols needed for stock, benchmark, and risk-free proxy data."""

        symbols = [*self.selected_stocks, self.benchmark_symbol, self.risk_free_symbol]
        return tuple(dict.fromkeys(symbols))


def load_quant_terminal_config(path: str | Path) -> QuantTerminalConfig:
    """Load and validate the terminal YAML subset config."""

    raw = load_simple_yaml(path)
    benchmark = _mapping(raw.get("benchmark"), "benchmark")
    risk_free = _mapping(raw.get("risk_free_proxy"), "risk_free_proxy")
    return QuantTerminalConfig(
        selected_stocks=_string_tuple(raw.get("selected_stocks"), "selected_stocks"),
        fallback_stocks=_string_tuple(raw.get("fallback_stocks", []), "fallback_stocks"),
        benchmark_symbol=str(benchmark.get("symbol", "SPY")),
        risk_free_symbol=str(risk_free.get("symbol", "^IRX")),
        frequency=str(raw.get("frequency", "1d")),
        lookback_years=int(raw.get("lookback_years", 10)),
        base_currency=str(raw.get("base_currency", "USD")),
        risk_free_rate_annual=float(raw.get("risk_free_rate_annual", 0.0)),
        monte_carlo=_mapping(raw.get("monte_carlo", {}), "monte_carlo"),
        optimization=_mapping(raw.get("optimization", {}), "optimization"),
        backtesting=_mapping(raw.get("backtesting", {}), "backtesting"),
        options=_mapping(raw.get("options", {}), "options"),
        fixed_income=_mapping(raw.get("fixed_income", {}), "fixed_income"),
        rates_derivatives=_mapping(raw.get("rates_derivatives", {}), "rates_derivatives"),
        hedging=_mapping(raw.get("hedging", {}), "hedging"),
        exposure=_mapping(raw.get("exposure", {}), "exposure"),
    )


def build_ohlcv_request(
    config: QuantTerminalConfig, end: pd.Timestamp | None = None
) -> OHLCVRequest:
    """Build a read-only yfinance-compatible OHLCV request from config."""

    clean_end = pd.Timestamp.now(tz="UTC").normalize() if end is None else pd.Timestamp(end)
    start = clean_end - pd.DateOffset(years=config.lookback_years)
    return OHLCVRequest(
        symbols=config.market_symbols,
        start=start.date().isoformat(),
        end=(clean_end + pd.Timedelta(days=1)).date().isoformat(),
        frequency=config.frequency,
        market_type=MarketType.EQUITY,
        source="yfinance",
        currency=config.base_currency,
    )


def make_synthetic_ohlcv(
    config: QuantTerminalConfig,
    end: pd.Timestamp | None = None,
    seed: int = 21,
) -> pd.DataFrame:
    """Create deterministic OHLCV fallback data for offline tests and demos.

    The generated data is explicitly synthetic and must not be labeled as real market
    data by callers.

    Calibrated for 15-30% max drawdowns with GBM parameters: drift=0.038-0.055% daily,
    vol=0.8-1.2% daily for stocks; drift=0.03% daily, vol=0.9% daily for benchmark.
    Default seed 21 produces worst-case drawdowns of 19-24% across all stocks.
    """

    clean_end = pd.Timestamp.now(tz="UTC").normalize() if end is None else pd.Timestamp(end)
    periods = int(config.lookback_years * 252)
    index = pd.bdate_range(end=clean_end, periods=periods, tz="UTC")
    rng = np.random.default_rng(seed)
    symbols = (*config.selected_stocks, config.benchmark_symbol)
    frames = []
    n_stocks = len(config.selected_stocks)
    drifts = np.linspace(0.00038, 0.00055, n_stocks)
    vols = np.linspace(0.008, 0.012, n_stocks)
    benchmark_drift = 0.0003
    benchmark_vol = 0.009
    for position, symbol in enumerate(symbols):
        is_bm = symbol == config.benchmark_symbol
        daily_drift = benchmark_drift if is_bm else drifts[position]
        daily_vol = benchmark_vol if is_bm else vols[position]
        innovations = rng.normal(daily_drift, daily_vol, size=len(index))
        prices = 100.0 * np.exp(np.cumsum(innovations))
        open_prices = prices * (1.0 + rng.normal(0.0, 0.002, size=len(index)))
        close = prices
        high = np.maximum(open_prices, close) * (1.0 + rng.uniform(0.0, 0.01, size=len(index)))
        low = np.minimum(open_prices, close) * (1.0 - rng.uniform(0.0, 0.01, size=len(index)))
        frames.append(
            pd.DataFrame(
                {
                    "asset_id": symbol,
                    "timestamp": index,
                    "available_at": index + pd.Timedelta(days=1),
                    "open": open_prices,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(1_000_000, 20_000_000, size=len(index)),
                    "market_type": MarketType.EQUITY.value,
                    "currency": config.base_currency,
                    "source": "synthetic_quant_terminal",
                    "venue": "LOCAL_SYNTHETIC",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise QuantTerminalConfigError(f"{name} must be a list of symbols.")
    symbols = tuple(str(symbol).strip() for symbol in value if str(symbol).strip())
    if not symbols:
        raise QuantTerminalConfigError(f"{name} must not be empty.")
    return symbols


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QuantTerminalConfigError(f"{name} must be a mapping.")
    return dict(value)
