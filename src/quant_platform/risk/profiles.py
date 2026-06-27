"""Risk profile loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_platform.backtesting.costs import TransactionCostConfig
from quant_platform.config.simple_yaml import load_simple_yaml
from quant_platform.data.schemas import MarketType


class RiskProfileError(ValueError):
    """Raised when a risk profile is missing required fields or invalid."""


REQUIRED_FIELDS = (
    "target_max_drawdown",
    "max_single_asset_weight",
    "max_gross_leverage",
    "max_turnover",
    "commission_bps_equity",
    "spread_bps_equity",
    "slippage_bps_equity",
    "commission_bps_crypto",
    "spread_bps_crypto",
    "slippage_bps_crypto",
)


def validate_risk_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Validate one risk profile dictionary and return it unchanged."""

    missing = [field for field in REQUIRED_FIELDS if field not in profile]
    if missing:
        raise RiskProfileError(f"Risk profile missing fields: {missing}")
    if not 0 < float(profile["target_max_drawdown"]) < 1:
        raise RiskProfileError("target_max_drawdown must be in (0, 1).")
    if not 0 < float(profile["max_single_asset_weight"]) <= 1:
        raise RiskProfileError("max_single_asset_weight must be in (0, 1].")
    if not 0 < float(profile["max_gross_leverage"]) <= 1:
        raise RiskProfileError("max_gross_leverage must be in (0, 1] for current scope.")
    if float(profile["max_turnover"]) < 0:
        raise RiskProfileError("max_turnover must be >= 0.")
    for field in REQUIRED_FIELDS:
        if field.startswith(("commission_bps", "spread_bps", "slippage_bps")):
            if float(profile[field]) < 0:
                raise RiskProfileError(f"{field} must be >= 0.")
    return profile


def load_risk_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate risk profiles from a local YAML config."""

    loaded = load_simple_yaml(path)
    profiles = loaded.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise RiskProfileError("profiles mapping is required.")
    for profile in profiles.values():
        if not isinstance(profile, dict):
            raise RiskProfileError("Each profile must be a mapping.")
        validate_risk_profile(profile)
    return profiles


def get_risk_profile(profiles: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """Return a named profile or fail clearly."""

    try:
        return profiles[name]
    except KeyError as exc:
        raise RiskProfileError(f"Unknown risk profile: {name}") from exc


def build_cost_config_for_market(
    profile: dict[str, Any],
    market_type: str | MarketType,
) -> TransactionCostConfig:
    """Build transaction cost settings for equity or crypto from bps fields."""

    validate_risk_profile(profile)
    clean_market_type = MarketType(market_type)
    if clean_market_type == MarketType.EQUITY:
        suffix = "equity"
    elif clean_market_type == MarketType.CRYPTO:
        suffix = "crypto"
    else:
        raise RiskProfileError(f"Unsupported market_type: {market_type}")
    return TransactionCostConfig.from_bps(
        commission_bps=float(profile[f"commission_bps_{suffix}"]),
        spread_bps=float(profile[f"spread_bps_{suffix}"]),
        slippage_bps=float(profile[f"slippage_bps_{suffix}"]),
    )
