"""Factory and status registry for read-only market data providers."""

from __future__ import annotations

from typing import Any

from quant_platform.config.settings import PlatformSettings
from quant_platform.data.providers.alpha_vantage_provider import AlphaVantageDailyProvider
from quant_platform.data.providers.base import MarketDataProvider
from quant_platform.data.providers.binance_public_provider import BinancePublicSpotProvider
from quant_platform.data.providers.cryptocompare_provider import CryptoCompareDailyProvider
from quant_platform.data.providers.http import HttpClientConfig, ReadOnlyHttpClient
from quant_platform.data.providers.nasdaq_data_link_provider import NasdaqDataLinkProvider
from quant_platform.data.providers.polygon_provider import PolygonDailyProvider
from quant_platform.data.providers.yfinance_provider import YFinanceDailyProvider
from quant_platform.data.schemas import MarketType


class ProviderFactoryError(ValueError):
    """Raised when a provider cannot be constructed safely."""


PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "yfinance": {
        "market_types": [MarketType.EQUITY.value],
        "requires_api_key": False,
        "enabled_attr": "yfinance_enabled",
        "configured_attr": None,
    },
    "binance_public": {
        "market_types": [MarketType.CRYPTO.value],
        "requires_api_key": False,
        "enabled_attr": "binance_public_enabled",
        "configured_attr": None,
    },
    "alpha_vantage": {
        "market_types": [MarketType.EQUITY.value],
        "requires_api_key": True,
        "enabled_attr": "alpha_vantage_enabled",
        "configured_attr": "alpha_vantage_configured",
    },
    "polygon": {
        "market_types": [MarketType.EQUITY.value],
        "requires_api_key": True,
        "enabled_attr": "polygon_enabled",
        "configured_attr": "polygon_configured",
    },
    "nasdaq_data_link": {
        "market_types": [MarketType.EQUITY.value, "macro", "alternative"],
        "requires_api_key": True,
        "enabled_attr": "nasdaq_data_link_enabled",
        "configured_attr": "nasdaq_data_link_configured",
    },
    "cryptocompare": {
        "market_types": [MarketType.CRYPTO.value],
        "requires_api_key": True,
        "enabled_attr": "cryptocompare_enabled",
        "configured_attr": "cryptocompare_configured",
    },
}


def create_market_data_provider(
    name: str,
    settings: PlatformSettings,
    dataset_code: str | None = None,
) -> MarketDataProvider:
    """Create a read-only provider by name without exposing credentials."""

    clean_name = name.strip().lower()
    http_client = _http_client(settings)
    credentials = settings.provider_credentials
    if clean_name == "yfinance":
        return YFinanceDailyProvider()
    if clean_name == "binance_public":
        return BinancePublicSpotProvider(
            timeout=settings.provider_runtime.provider_http_timeout_seconds
        )
    if clean_name == "alpha_vantage":
        return AlphaVantageDailyProvider(credentials.alpha_vantage_api_key, http_client=http_client)
    if clean_name == "polygon":
        return PolygonDailyProvider(credentials.polygon_api_key, http_client=http_client)
    if clean_name == "nasdaq_data_link":
        return NasdaqDataLinkProvider(
            credentials.nasdaq_data_link_api_key,
            dataset_code=dataset_code,
            http_client=http_client,
        )
    if clean_name == "cryptocompare":
        return CryptoCompareDailyProvider(
            credentials.cryptocompare_api_key,
            http_client=http_client,
        )
    raise ProviderFactoryError(f"Unknown provider: {name}")


def list_available_providers(settings: PlatformSettings) -> list[dict[str, object]]:
    """Return safe provider availability metadata with no secret values."""

    rows = []
    for name, capability in PROVIDER_CAPABILITIES.items():
        enabled = bool(getattr(settings.data_providers, str(capability["enabled_attr"])))
        configured_attr = capability["configured_attr"]
        configured = True
        if configured_attr is not None:
            configured = bool(getattr(settings.provider_credentials, str(configured_attr)))
        requires_key = bool(capability["requires_api_key"])
        if not enabled:
            status = "disabled"
        elif requires_key and not configured:
            status = "missing_api_key"
        else:
            status = "available"
        rows.append(
            {
                "name": name,
                "enabled": enabled,
                "configured": configured,
                "market_types": list(capability["market_types"]),
                "requires_api_key": requires_key,
                "status": status,
            }
        )
    return rows


def _http_client(settings: PlatformSettings) -> ReadOnlyHttpClient:
    runtime = settings.provider_runtime
    return ReadOnlyHttpClient(
        HttpClientConfig(
            timeout_seconds=runtime.provider_http_timeout_seconds,
            max_retries=runtime.provider_max_retries,
            retry_backoff_seconds=runtime.provider_retry_backoff_seconds,
            cache_enabled=runtime.provider_cache_enabled,
        )
    )
