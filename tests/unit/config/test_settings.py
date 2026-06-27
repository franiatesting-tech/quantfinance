from __future__ import annotations

import pytest

from quant_platform.config.settings import (
    SettingsError,
    load_settings_from_env,
    public_settings_dict,
)


def test_load_settings_defaults_are_safe() -> None:
    settings = load_settings_from_env({})

    assert settings.environment.name == "local"
    assert not settings.safety.allow_live_trading
    assert not settings.safety.allow_paper_trading
    assert settings.safety.require_explicit_live_trading_confirmation
    assert settings.safety.max_allowed_leverage == pytest.approx(1.0)
    assert settings.safety.base_currency == "USD"
    assert settings.safety.initial_capital == pytest.approx(10_000.0)
    assert settings.research.default_frequency == "1d"
    assert settings.research.default_equity_provider == "yfinance"
    assert settings.research.default_crypto_provider == "binance_public"
    assert settings.research.conservative_target_max_drawdown == pytest.approx(0.15)
    assert settings.research.aggressive_target_max_drawdown == pytest.approx(0.30)
    assert not settings.data_providers.alpha_vantage_enabled
    assert settings.provider_runtime.provider_http_timeout_seconds == pytest.approx(30.0)
    assert settings.provider_runtime.provider_max_retries == 3
    assert settings.provider_runtime.provider_retry_backoff_seconds == pytest.approx(2.0)
    assert settings.provider_runtime.provider_cache_enabled


def test_load_settings_parses_valid_booleans() -> None:
    settings = load_settings_from_env(
        {
            "QUANT_PLATFORM_ALLOW_PAPER_TRADING": "yes",
            "YFINANCE_ENABLED": "TRUE",
            "STOOQ_ENABLED": "0",
        }
    )

    assert settings.safety.allow_paper_trading
    assert settings.data_providers.yfinance_enabled
    assert not settings.data_providers.stooq_enabled


def test_live_trading_is_blocked_without_double_confirmation() -> None:
    with pytest.raises(SettingsError, match="I_UNDERSTAND_LIVE_TRADING_RISK"):
        load_settings_from_env({"QUANT_PLATFORM_ALLOW_LIVE_TRADING": "true"})


def test_live_trading_requires_explicit_risk_acknowledgement() -> None:
    settings = load_settings_from_env(
        {
            "QUANT_PLATFORM_ALLOW_LIVE_TRADING": "true",
            "I_UNDERSTAND_LIVE_TRADING_RISK": "true",
        }
    )

    assert settings.safety.allow_live_trading
    assert settings.safety.live_trading_risk_acknowledged


def test_negative_max_leverage_fails() -> None:
    with pytest.raises(SettingsError, match="MAX_ALLOWED_LEVERAGE"):
        load_settings_from_env({"MAX_ALLOWED_LEVERAGE": "-1"})


def test_empty_base_currency_fails() -> None:
    with pytest.raises(SettingsError, match="BASE_CURRENCY"):
        load_settings_from_env({"BASE_CURRENCY": " "})


def test_api_keys_are_optional_and_not_required() -> None:
    settings = load_settings_from_env({"COINGECKO_ENABLED": "true"})

    assert settings.data_providers.coingecko_enabled
    assert settings.provider_credentials.alpha_vantage_api_key == ""
    assert settings.provider_credentials.polygon_api_key == ""
    assert settings.provider_credentials.nasdaq_data_link_api_key == ""
    assert settings.provider_credentials.cryptocompare_api_key == ""
    assert not settings.provider_credentials.alpha_vantage_configured


def test_invalid_initial_capital_fails() -> None:
    with pytest.raises(SettingsError, match="INITIAL_CAPITAL"):
        load_settings_from_env({"INITIAL_CAPITAL": "0"})


def test_invalid_drawdown_targets_fail() -> None:
    with pytest.raises(SettingsError, match="CONSERVATIVE_TARGET_MAX_DRAWDOWN"):
        load_settings_from_env({"CONSERVATIVE_TARGET_MAX_DRAWDOWN": "-0.1"})
    with pytest.raises(SettingsError, match="AGGRESSIVE_TARGET_MAX_DRAWDOWN"):
        load_settings_from_env({"AGGRESSIVE_TARGET_MAX_DRAWDOWN": "1.0"})


def test_conservative_drawdown_must_be_lower_than_aggressive() -> None:
    with pytest.raises(SettingsError, match="CONSERVATIVE_TARGET_MAX_DRAWDOWN"):
        load_settings_from_env(
            {
                "CONSERVATIVE_TARGET_MAX_DRAWDOWN": "0.30",
                "AGGRESSIVE_TARGET_MAX_DRAWDOWN": "0.30",
            }
        )


def test_empty_default_provider_fails() -> None:
    with pytest.raises(SettingsError, match="DEFAULT_EQUITY_PROVIDER"):
        load_settings_from_env({"DEFAULT_EQUITY_PROVIDER": " "})


def test_missing_dotenv_file_does_not_fail(tmp_path) -> None:  # noqa: ANN001
    settings = load_settings_from_env({}, dotenv_path=tmp_path / "missing.env")

    assert settings.environment.name == "local"


def test_present_keys_set_configured_true_without_public_secret_output() -> None:
    settings = load_settings_from_env(
        {
            "ALPHA_VANTAGE_API_KEY": "alpha-secret",
            "POLYGON_API_KEY": "polygon-secret",
            "NASDAQ_DATA_LINK_API_KEY": "nasdaq-secret",
            "CRYPTOCOMPARE_API_KEY": "crypto-secret",
        }
    )

    assert settings.provider_credentials.alpha_vantage_configured
    assert settings.provider_credentials.polygon_configured
    assert settings.provider_credentials.nasdaq_data_link_configured
    assert settings.provider_credentials.cryptocompare_configured
    public_payload = str(public_settings_dict(settings))
    repr_payload = repr(settings.provider_credentials)
    assert "alpha-secret" not in public_payload
    assert "polygon-secret" not in public_payload
    assert "nasdaq-secret" not in public_payload
    assert "crypto-secret" not in public_payload
    assert "alpha-secret" not in repr_payload


def test_empty_keys_set_configured_false() -> None:
    settings = load_settings_from_env(
        {
            "ALPHA_VANTAGE_API_KEY": " ",
            "POLYGON_API_KEY": "",
        }
    )

    assert not settings.provider_credentials.alpha_vantage_configured
    assert not settings.provider_credentials.polygon_configured


def test_provider_enabled_without_key_is_incomplete_but_safe() -> None:
    settings = load_settings_from_env({"POLYGON_ENABLED": "true", "POLYGON_API_KEY": ""})

    assert settings.data_providers.polygon_enabled
    assert not settings.provider_credentials.polygon_configured


def test_provider_runtime_invalid_values_fail() -> None:
    with pytest.raises(SettingsError, match="PROVIDER_HTTP_TIMEOUT_SECONDS"):
        load_settings_from_env({"PROVIDER_HTTP_TIMEOUT_SECONDS": "0"})
    with pytest.raises(SettingsError, match="PROVIDER_MAX_RETRIES"):
        load_settings_from_env({"PROVIDER_MAX_RETRIES": "-1"})
    with pytest.raises(SettingsError, match="PROVIDER_RETRY_BACKOFF_SECONDS"):
        load_settings_from_env({"PROVIDER_RETRY_BACKOFF_SECONDS": "-0.1"})


def test_provider_cache_flag_parses_boolean() -> None:
    settings = load_settings_from_env({"PROVIDER_CACHE_ENABLED": "false"})

    assert not settings.provider_runtime.provider_cache_enabled
