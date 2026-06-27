"""Environment-backed platform settings with safety defaults.

This module intentionally uses only the Python standard library. It does not load
`.env` files directly; `.env.example` is a contract, and callers may populate
`os.environ` by their preferred deployment mechanism in future iterations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


class SettingsError(ValueError):
    """Raised when environment settings are unsafe or invalid."""


TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


@dataclass(frozen=True)
class PlatformEnvironment:
    """Runtime environment label."""

    name: str = "local"

    def __post_init__(self) -> None:
        clean_name = self.name.strip()
        if not clean_name:
            raise SettingsError("QUANT_PLATFORM_ENV must not be empty.")
        object.__setattr__(self, "name", clean_name)


@dataclass(frozen=True)
class SafetySettings:
    """Safety switches that must default to non-trading behavior."""

    allow_live_trading: bool = False
    allow_paper_trading: bool = False
    require_explicit_live_trading_confirmation: bool = True
    live_trading_risk_acknowledged: bool = False
    max_allowed_leverage: float = 1.0
    base_currency: str = "USD"
    initial_capital: float = 10_000.0

    def __post_init__(self) -> None:
        if self.allow_live_trading:
            if (
                self.require_explicit_live_trading_confirmation
                and not self.live_trading_risk_acknowledged
            ):
                raise SettingsError(
                    "Live trading requires I_UNDERSTAND_LIVE_TRADING_RISK=true."
                )
        if self.max_allowed_leverage <= 0:
            raise SettingsError("MAX_ALLOWED_LEVERAGE must be > 0.")
        clean_currency = self.base_currency.strip().upper()
        if not clean_currency:
            raise SettingsError("BASE_CURRENCY must not be empty.")
        object.__setattr__(self, "base_currency", clean_currency)
        if self.initial_capital <= 0:
            raise SettingsError("INITIAL_CAPITAL must be > 0.")


@dataclass(frozen=True)
class DataProviderSettings:
    """Read-only provider toggles and optional key placeholders."""

    yfinance_enabled: bool = False
    stooq_enabled: bool = False
    coingecko_enabled: bool = False
    binance_public_enabled: bool = False
    ccxt_enabled: bool = False
    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""
    nasdaq_data_link_api_key: str = ""
    cryptocompare_api_key: str = ""


@dataclass(frozen=True)
class ResearchSettings:
    """Research defaults chosen for the first real-data simulation layer."""

    default_frequency: str = "1d"
    default_equity_provider: str = "yfinance"
    default_crypto_provider: str = "binance_public"
    conservative_target_max_drawdown: float = 0.15
    aggressive_target_max_drawdown: float = 0.30

    def __post_init__(self) -> None:
        clean_frequency = self.default_frequency.strip()
        clean_equity_provider = self.default_equity_provider.strip()
        clean_crypto_provider = self.default_crypto_provider.strip()
        if not clean_frequency:
            raise SettingsError("DEFAULT_FREQUENCY must not be empty.")
        if not clean_equity_provider:
            raise SettingsError("DEFAULT_EQUITY_PROVIDER must not be empty.")
        if not clean_crypto_provider:
            raise SettingsError("DEFAULT_CRYPTO_PROVIDER must not be empty.")
        if not 0 < self.conservative_target_max_drawdown < 1:
            raise SettingsError("CONSERVATIVE_TARGET_MAX_DRAWDOWN must be in (0, 1).")
        if not 0 < self.aggressive_target_max_drawdown < 1:
            raise SettingsError("AGGRESSIVE_TARGET_MAX_DRAWDOWN must be in (0, 1).")
        if self.conservative_target_max_drawdown >= self.aggressive_target_max_drawdown:
            raise SettingsError(
                "CONSERVATIVE_TARGET_MAX_DRAWDOWN must be lower than "
                "AGGRESSIVE_TARGET_MAX_DRAWDOWN."
            )
        object.__setattr__(self, "default_frequency", clean_frequency)
        object.__setattr__(self, "default_equity_provider", clean_equity_provider)
        object.__setattr__(self, "default_crypto_provider", clean_crypto_provider)


@dataclass(frozen=True)
class PlatformSettings:
    """Complete platform settings snapshot."""

    environment: PlatformEnvironment
    safety: SafetySettings
    data_providers: DataProviderSettings
    research: ResearchSettings


def _env_value(environ: Mapping[str, str], name: str, default: str) -> str:
    return environ.get(name, default)


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    clean_value = raw_value.strip().lower()
    if clean_value in TRUE_VALUES:
        return True
    if clean_value in FALSE_VALUES:
        return False
    raise SettingsError(f"{name} must be a boolean value.")


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be numeric.") from exc


def load_settings_from_env(environ: Mapping[str, str] | None = None) -> PlatformSettings:
    """Load platform settings from environment variables with safe defaults."""

    source = os.environ if environ is None else environ
    environment = PlatformEnvironment(
        name=_env_value(source, "QUANT_PLATFORM_ENV", "local"),
    )
    safety = SafetySettings(
        allow_live_trading=_env_bool(source, "QUANT_PLATFORM_ALLOW_LIVE_TRADING", False),
        allow_paper_trading=_env_bool(source, "QUANT_PLATFORM_ALLOW_PAPER_TRADING", False),
        require_explicit_live_trading_confirmation=_env_bool(
            source,
            "REQUIRE_EXPLICIT_LIVE_TRADING_CONFIRMATION",
            True,
        ),
        live_trading_risk_acknowledged=_env_bool(
            source,
            "I_UNDERSTAND_LIVE_TRADING_RISK",
            False,
        ),
        max_allowed_leverage=_env_float(source, "MAX_ALLOWED_LEVERAGE", 1.0),
        base_currency=_env_value(source, "BASE_CURRENCY", "USD"),
        initial_capital=_env_float(source, "INITIAL_CAPITAL", 10_000.0),
    )
    data_providers = DataProviderSettings(
        yfinance_enabled=_env_bool(source, "YFINANCE_ENABLED", True),
        stooq_enabled=_env_bool(source, "STOOQ_ENABLED", False),
        coingecko_enabled=_env_bool(source, "COINGECKO_ENABLED", True),
        binance_public_enabled=_env_bool(source, "BINANCE_PUBLIC_ENABLED", True),
        ccxt_enabled=_env_bool(source, "CCXT_ENABLED", False),
        alpha_vantage_api_key=_env_value(source, "ALPHA_VANTAGE_API_KEY", ""),
        polygon_api_key=_env_value(source, "POLYGON_API_KEY", ""),
        nasdaq_data_link_api_key=_env_value(source, "NASDAQ_DATA_LINK_API_KEY", ""),
        cryptocompare_api_key=_env_value(source, "CRYPTOCOMPARE_API_KEY", ""),
    )
    research = ResearchSettings(
        default_frequency=_env_value(source, "DEFAULT_FREQUENCY", "1d"),
        default_equity_provider=_env_value(source, "DEFAULT_EQUITY_PROVIDER", "yfinance"),
        default_crypto_provider=_env_value(source, "DEFAULT_CRYPTO_PROVIDER", "binance_public"),
        conservative_target_max_drawdown=_env_float(
            source,
            "CONSERVATIVE_TARGET_MAX_DRAWDOWN",
            0.15,
        ),
        aggressive_target_max_drawdown=_env_float(
            source,
            "AGGRESSIVE_TARGET_MAX_DRAWDOWN",
            0.30,
        ),
    )
    return PlatformSettings(
        environment=environment,
        safety=safety,
        data_providers=data_providers,
        research=research,
    )
