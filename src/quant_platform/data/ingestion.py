"""Read-only real-data ingestion orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.config.simple_yaml import load_simple_yaml
from quant_platform.data.providers.base import MarketDataProvider, OHLCVRequest, OHLCVResponse
from quant_platform.data.providers.binance_public_provider import BinancePublicSpotProvider
from quant_platform.data.providers.yfinance_provider import YFinanceDailyProvider
from quant_platform.data.registry import RegisteredDataset, register_dataset
from quant_platform.data.schemas import DatasetMetadata, Frequency, MarketType


class IngestionError(ValueError):
    """Raised when read-only ingestion cannot be configured safely."""


def load_universe_config(path: str | Path) -> dict[str, Any]:
    """Load the ETFs + crypto majors universe config."""

    loaded = load_simple_yaml(path)
    if loaded.get("frequency") != Frequency.DAILY.value:
        raise IngestionError("Only daily universe configs are supported in Iteration 005.")
    return loaded


def _flatten_symbol_groups(groups: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for value in groups.values():
        if isinstance(value, list):
            symbols.extend(str(symbol) for symbol in value)
    return symbols


def _limit_symbols(symbols: list[str], limit: int | None) -> tuple[str, ...]:
    if limit is None:
        return tuple(symbols)
    if limit < 0:
        raise IngestionError("symbol limits must be >= 0.")
    return tuple(symbols[:limit])


def download_equity_universe_daily(
    universe_config: dict[str, Any],
    start: str,
    end: str,
    provider: MarketDataProvider | None = None,
    limit_symbols: int | None = None,
) -> OHLCVResponse:
    """Download read-only daily equity/ETF bars from the configured universe."""

    symbols = _limit_symbols(
        _flatten_symbol_groups(universe_config.get("equity_universe", {})),
        limit_symbols,
    )
    request = OHLCVRequest(
        symbols=symbols,
        start=start,
        end=end,
        frequency=Frequency.DAILY.value,
        market_type=MarketType.EQUITY,
        source="yfinance",
        currency=str(universe_config.get("base_currency", "USD")),
    )
    return (provider or YFinanceDailyProvider()).download_ohlcv(request)


def download_crypto_universe_daily(
    universe_config: dict[str, Any],
    start: str,
    end: str,
    provider: MarketDataProvider | None = None,
    limit_symbols: int | None = None,
) -> OHLCVResponse:
    """Download read-only daily crypto spot bars from Binance public klines."""

    crypto_groups = universe_config.get("crypto_universe", {})
    symbols = _limit_symbols(_flatten_symbol_groups(crypto_groups), limit_symbols)
    request = OHLCVRequest(
        symbols=symbols,
        start=start,
        end=end,
        frequency=Frequency.DAILY.value,
        market_type=MarketType.CRYPTO,
        source="binance_public",
        currency="USDT",
    )
    return (provider or BinancePublicSpotProvider()).download_ohlcv(request)


def download_combined_daily_universe(
    universe_config: dict[str, Any],
    start: str,
    end: str,
    equity_provider: MarketDataProvider | None = None,
    crypto_provider: MarketDataProvider | None = None,
    limit_equity: int | None = None,
    limit_crypto: int | None = None,
) -> OHLCVResponse:
    """Download equity and crypto universes, recording partial failures."""

    equity = download_equity_universe_daily(
        universe_config,
        start=start,
        end=end,
        provider=equity_provider,
        limit_symbols=limit_equity,
    )
    crypto = download_crypto_universe_daily(
        universe_config,
        start=start,
        end=end,
        provider=crypto_provider,
        limit_symbols=limit_crypto,
    )
    frames = [response.data for response in (equity, crypto) if not response.data.empty]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    failed = {**equity.failed_symbols, **crypto.failed_symbols}
    return OHLCVResponse(
        data=data,
        successful_symbols=equity.successful_symbols + crypto.successful_symbols,
        failed_symbols=failed,
        metadata={"equity": equity.metadata, "crypto": crypto.metadata},
    )


def register_real_dataset(
    response: OHLCVResponse,
    registry_dir: str | Path,
    dataset_id: str,
    version: str,
    market_type: MarketType = MarketType.EQUITY,
) -> RegisteredDataset:
    """Register a read-only downloaded dataset in the local registry."""

    if response.data.empty:
        raise IngestionError("Cannot register an empty dataset response.")
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        version=version,
        source="read_only_real_data",
        market_type=market_type,
        frequency=Frequency.DAILY,
        created_at=datetime.now(tz=UTC),
        description=(
            f"successful={list(response.successful_symbols)}; "
            f"failed={response.failed_symbols}"
        ),
    )
    return register_dataset(response.data, metadata, registry_dir=registry_dir, overwrite=False)
