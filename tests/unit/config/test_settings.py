from __future__ import annotations

import pytest

from quant_platform.config.settings import SettingsError, load_settings_from_env


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
    assert settings.data_providers.alpha_vantage_api_key == ""
    assert settings.data_providers.polygon_api_key == ""
    assert settings.data_providers.nasdaq_data_link_api_key == ""
    assert settings.data_providers.cryptocompare_api_key == ""


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
