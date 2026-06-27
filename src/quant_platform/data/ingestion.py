"""Read-only real-data ingestion orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.config.settings import PlatformSettings, load_settings_from_env
from quant_platform.config.simple_yaml import load_simple_yaml
from quant_platform.data.providers.base import MarketDataProvider, OHLCVRequest, OHLCVResponse
from quant_platform.data.providers.binance_public_provider import BinancePublicSpotProvider
from quant_platform.data.providers.factory import create_market_data_provider
from quant_platform.data.providers.yfinance_provider import YFinanceDailyProvider
from quant_platform.data.registry import RegisteredDataset, register_dataset
from quant_platform.data.schemas import DatasetMetadata, Frequency, MarketType
from quant_platform.reporting.data_quality_report import (
    build_data_quality_report,
    write_data_quality_report,
)


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
    equity_provider_name: str | None = None,
    crypto_provider_name: str | None = None,
    fallback_provider_name: str | None = None,
    allow_fallback: bool = True,
    settings: PlatformSettings | None = None,
    nasdaq_dataset_code: str | None = None,
    limit_equity: int | None = None,
    limit_crypto: int | None = None,
) -> OHLCVResponse:
    """Download equity and crypto universes, recording partial failures."""

    active_settings = settings or load_settings_from_env()
    equity = _download_market_with_optional_fallback(
        market_type=MarketType.EQUITY,
        universe_config=universe_config,
        start=start,
        end=end,
        provider=equity_provider,
        provider_name=equity_provider_name,
        fallback_provider_name=fallback_provider_name,
        allow_fallback=allow_fallback,
        settings=active_settings,
        nasdaq_dataset_code=nasdaq_dataset_code,
        limit_symbols=limit_equity,
    )
    crypto = _download_market_with_optional_fallback(
        market_type=MarketType.CRYPTO,
        universe_config=universe_config,
        start=start,
        end=end,
        provider=crypto_provider,
        provider_name=crypto_provider_name,
        fallback_provider_name=fallback_provider_name,
        allow_fallback=allow_fallback,
        settings=active_settings,
        nasdaq_dataset_code=nasdaq_dataset_code,
        limit_symbols=limit_crypto,
    )
    frames = [response.data for response in (equity, crypto) if not response.data.empty]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    failed = {**equity.failed_symbols, **crypto.failed_symbols}
    return OHLCVResponse(
        data=data,
        successful_symbols=equity.successful_symbols + crypto.successful_symbols,
        failed_symbols=failed,
        metadata={
            "equity": equity.metadata,
            "crypto": crypto.metadata,
            "successful_symbols": list(equity.successful_symbols + crypto.successful_symbols),
            "failed_symbols": failed,
            "start": start,
            "end": end,
            "frequency": Frequency.DAILY.value,
        },
    )


def _download_market_with_optional_fallback(
    market_type: MarketType,
    universe_config: dict[str, Any],
    start: str,
    end: str,
    provider: MarketDataProvider | None,
    provider_name: str | None,
    fallback_provider_name: str | None,
    allow_fallback: bool,
    settings: PlatformSettings,
    nasdaq_dataset_code: str | None,
    limit_symbols: int | None,
) -> OHLCVResponse:
    primary_name = provider_name or _default_provider_name(market_type)
    try:
        active_provider = provider or create_market_data_provider(
            primary_name,
            settings,
            dataset_code=nasdaq_dataset_code,
        )
        response = _download_market(
            market_type,
            universe_config,
            start,
            end,
            active_provider,
            limit_symbols,
        )
        if response.data.empty and response.failed_symbols:
            raise IngestionError(f"provider {primary_name} returned no usable rows")
        return _with_provider_metadata(response, primary_name, None)
    except Exception as exc:  # noqa: BLE001 - controlled fallback boundary.
        if not allow_fallback or not fallback_provider_name:
            raise IngestionError(f"Provider {primary_name} failed: {exc}") from exc
        if not _fallback_supports_market(fallback_provider_name, market_type):
            raise IngestionError(
                f"Provider {primary_name} failed and fallback {fallback_provider_name} "
                f"does not support {market_type.value}."
            ) from exc
        fallback = create_market_data_provider(fallback_provider_name, settings)
        fallback_response = _download_market(
            market_type,
            universe_config,
            start,
            end,
            fallback,
            limit_symbols,
        )
        return _with_provider_metadata(fallback_response, fallback_provider_name, str(exc))


def _download_market(
    market_type: MarketType,
    universe_config: dict[str, Any],
    start: str,
    end: str,
    provider: MarketDataProvider,
    limit_symbols: int | None,
) -> OHLCVResponse:
    if market_type == MarketType.EQUITY:
        return download_equity_universe_daily(
            universe_config,
            start=start,
            end=end,
            provider=provider,
            limit_symbols=limit_symbols,
        )
    return download_crypto_universe_daily(
        universe_config,
        start=start,
        end=end,
        provider=provider,
        limit_symbols=limit_symbols,
    )


def _default_provider_name(market_type: MarketType) -> str:
    if market_type == MarketType.EQUITY:
        return "yfinance"
    return "binance_public"


def _fallback_supports_market(provider_name: str, market_type: MarketType) -> bool:
    name = provider_name.strip().lower()
    if market_type == MarketType.EQUITY:
        return name in {"yfinance", "alpha_vantage", "polygon", "nasdaq_data_link"}
    return name in {"binance_public", "cryptocompare"}


def _with_provider_metadata(
    response: OHLCVResponse,
    provider_name: str,
    fallback_reason: str | None,
) -> OHLCVResponse:
    provider_used_by_symbol = {symbol: provider_name for symbol in response.successful_symbols}
    metadata = {
        **response.metadata,
        "selected_provider": provider_name,
        "provider_used_by_symbol": provider_used_by_symbol,
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason
    return OHLCVResponse(
        data=response.data,
        successful_symbols=response.successful_symbols,
        failed_symbols=response.failed_symbols,
        metadata=metadata,
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
        detail = (
            "all symbols failed or no rows were returned" if response.failed_symbols else "empty"
        )
        raise IngestionError(f"Cannot register dataset: {detail}.")
    provider_metadata = {
        **response.metadata,
        "successful_symbols": list(response.successful_symbols),
        "failed_symbols": response.failed_symbols,
    }
    quality_report = build_data_quality_report(response.data, provider_metadata)
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
    registered = register_dataset(
        response.data,
        metadata,
        registry_dir=registry_dir,
        overwrite=False,
    )
    report_path = write_data_quality_report(
        quality_report,
        registered.manifest_path.parent / "data_quality_report.json",
    )
    _append_quality_metadata_to_manifest(registered.manifest_path, quality_report, report_path.name)
    return registered


def _append_quality_metadata_to_manifest(
    manifest_path: Path,
    quality_report: dict[str, Any],
    quality_report_file: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quality_report_file"] = quality_report_file
    manifest["coverage_metadata"] = {
        "symbols_total": quality_report["symbols_total"],
        "symbols_successful": quality_report["symbols_successful"],
        "symbols_failed": quality_report["symbols_failed"],
        "date_range": quality_report["date_range"],
        "failed_checks": quality_report["failed_checks"],
        "warnings": quality_report["warnings"],
        "suitable_for_backtest_demo": quality_report["suitable_for_backtest_demo"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
