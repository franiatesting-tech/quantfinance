"""Environment-backed platform settings with safety defaults.

The module may load a local `.env` file through `python-dotenv`, but API key values
are never printed, logged, or exposed through the public settings snapshot.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv


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
    """Read-only provider toggles."""

    yfinance_enabled: bool = True
    stooq_enabled: bool = False
    coingecko_enabled: bool = True
    binance_public_enabled: bool = True
    ccxt_enabled: bool = False
    alpha_vantage_enabled: bool = False
    polygon_enabled: bool = False
    nasdaq_data_link_enabled: bool = False
    cryptocompare_enabled: bool = False


@dataclass(frozen=True)
class ProviderCredentialSettings:
    """API key values and boolean configured flags.

    Key fields use `repr=False` so accidental dataclass repr output cannot leak
    secrets. Use `public_settings_dict` for CLI output.
    """

    alpha_vantage_api_key: str = field(default="", repr=False)
    polygon_api_key: str = field(default="", repr=False)
    nasdaq_data_link_api_key: str = field(default="", repr=False)
    cryptocompare_api_key: str = field(default="", repr=False)
    alpha_vantage_configured: bool = False
    polygon_configured: bool = False
    nasdaq_data_link_configured: bool = False
    cryptocompare_configured: bool = False

    def __post_init__(self) -> None:
        clean_values = {
            "alpha_vantage_api_key": self.alpha_vantage_api_key.strip(),
            "polygon_api_key": self.polygon_api_key.strip(),
            "nasdaq_data_link_api_key": self.nasdaq_data_link_api_key.strip(),
            "cryptocompare_api_key": self.cryptocompare_api_key.strip(),
        }
        for name, value in clean_values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "alpha_vantage_configured",
            bool(clean_values["alpha_vantage_api_key"]),
        )
        object.__setattr__(self, "polygon_configured", bool(clean_values["polygon_api_key"]))
        object.__setattr__(
            self,
            "nasdaq_data_link_configured",
            bool(clean_values["nasdaq_data_link_api_key"]),
        )
        object.__setattr__(
            self,
            "cryptocompare_configured",
            bool(clean_values["cryptocompare_api_key"]),
        )


@dataclass(frozen=True)
class ProviderRuntimeSettings:
    """Runtime controls for read-only provider HTTP calls and cache policy."""

    provider_http_timeout_seconds: float = 30.0
    provider_max_retries: int = 3
    provider_retry_backoff_seconds: float = 2.0
    provider_cache_enabled: bool = True

    def __post_init__(self) -> None:
        if self.provider_http_timeout_seconds <= 0:
            raise SettingsError("PROVIDER_HTTP_TIMEOUT_SECONDS must be > 0.")
        if self.provider_max_retries < 0:
            raise SettingsError("PROVIDER_MAX_RETRIES must be >= 0.")
        if self.provider_retry_backoff_seconds < 0:
            raise SettingsError("PROVIDER_RETRY_BACKOFF_SECONDS must be >= 0.")


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
    provider_credentials: ProviderCredentialSettings
    provider_runtime: ProviderRuntimeSettings
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


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw_value = environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer.") from exc


def load_settings_from_env(
    environ: Mapping[str, str] | None = None,
    dotenv_path: str | Path = ".env",
    load_dotenv_file: bool = True,
) -> PlatformSettings:
    """Load platform settings from environment variables with safe defaults."""

    if environ is None and load_dotenv_file:
        load_dotenv(dotenv_path=dotenv_path, override=False)
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
        alpha_vantage_enabled=_env_bool(source, "ALPHA_VANTAGE_ENABLED", False),
        polygon_enabled=_env_bool(source, "POLYGON_ENABLED", False),
        nasdaq_data_link_enabled=_env_bool(source, "NASDAQ_DATA_LINK_ENABLED", False),
        cryptocompare_enabled=_env_bool(source, "CRYPTOCOMPARE_ENABLED", False),
    )
    provider_credentials = ProviderCredentialSettings(
        alpha_vantage_api_key=_env_value(source, "ALPHA_VANTAGE_API_KEY", ""),
        polygon_api_key=_env_value(source, "POLYGON_API_KEY", ""),
        nasdaq_data_link_api_key=_env_value(source, "NASDAQ_DATA_LINK_API_KEY", ""),
        cryptocompare_api_key=_env_value(source, "CRYPTOCOMPARE_API_KEY", ""),
    )
    provider_runtime = ProviderRuntimeSettings(
        provider_http_timeout_seconds=_env_float(source, "PROVIDER_HTTP_TIMEOUT_SECONDS", 30.0),
        provider_max_retries=_env_int(source, "PROVIDER_MAX_RETRIES", 3),
        provider_retry_backoff_seconds=_env_float(source, "PROVIDER_RETRY_BACKOFF_SECONDS", 2.0),
        provider_cache_enabled=_env_bool(source, "PROVIDER_CACHE_ENABLED", True),
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
        provider_credentials=provider_credentials,
        provider_runtime=provider_runtime,
        research=research,
    )


def public_settings_dict(settings: PlatformSettings) -> dict[str, object]:
    """Return a JSON-safe settings snapshot with no secret values."""

    payload = asdict(settings)
    payload["provider_credentials"] = {
        "alpha_vantage_configured": settings.provider_credentials.alpha_vantage_configured,
        "polygon_configured": settings.provider_credentials.polygon_configured,
        "nasdaq_data_link_configured": settings.provider_credentials.nasdaq_data_link_configured,
        "cryptocompare_configured": settings.provider_credentials.cryptocompare_configured,
    }
    return payload
