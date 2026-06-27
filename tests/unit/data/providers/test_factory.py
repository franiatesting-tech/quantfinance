from __future__ import annotations

from quant_platform.config.settings import load_settings_from_env
from quant_platform.data.providers.factory import (
    create_market_data_provider,
    list_available_providers,
)


def test_list_available_providers_never_exposes_keys() -> None:
    settings = load_settings_from_env(
        {
            "POLYGON_ENABLED": "true",
            "POLYGON_API_KEY": "polygon-secret",
            "ALPHA_VANTAGE_ENABLED": "true",
        }
    )

    providers = list_available_providers(settings)
    payload = str(providers)

    assert "polygon-secret" not in payload
    assert {row["name"] for row in providers} >= {"yfinance", "polygon", "cryptocompare"}
    polygon = next(row for row in providers if row["name"] == "polygon")
    alpha = next(row for row in providers if row["name"] == "alpha_vantage")
    assert polygon["status"] == "available"
    assert alpha["status"] == "missing_api_key"


def test_list_available_providers_distinguishes_configured_disabled_keyed_provider() -> None:
    settings = load_settings_from_env(
        {
            "POLYGON_ENABLED": "false",
            "POLYGON_API_KEY": "polygon-secret",
        }
    )

    providers = list_available_providers(settings)
    polygon = next(row for row in providers if row["name"] == "polygon")

    assert polygon["configured"] is True
    assert polygon["enabled"] is False
    assert polygon["status"] == "configured_but_disabled"


def test_create_public_providers() -> None:
    settings = load_settings_from_env({})

    assert (
        create_market_data_provider("yfinance", settings).__class__.__name__
        == "YFinanceDailyProvider"
    )
    assert (
        create_market_data_provider("binance_public", settings).__class__.__name__
        == "BinancePublicSpotProvider"
    )


def test_create_keyed_provider_requires_key() -> None:
    settings = load_settings_from_env({"POLYGON_ENABLED": "true"})

    try:
        create_market_data_provider("polygon", settings)
    except ValueError as exc:
        assert "API key is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing key should fail")
