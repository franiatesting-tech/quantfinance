from __future__ import annotations

import pytest

from quant_platform.data.schemas import MarketType
from quant_platform.risk.profiles import (
    RiskProfileError,
    build_cost_config_for_market,
    get_risk_profile,
    load_risk_profiles,
    validate_risk_profile,
)


def valid_profile() -> dict[str, float]:
    return {
        "target_max_drawdown": 0.15,
        "max_single_asset_weight": 0.20,
        "max_gross_leverage": 1.0,
        "max_turnover": 0.30,
        "commission_bps_equity": 1.0,
        "spread_bps_equity": 2.0,
        "slippage_bps_equity": 2.0,
        "commission_bps_crypto": 10.0,
        "spread_bps_crypto": 5.0,
        "slippage_bps_crypto": 10.0,
    }


def test_load_risk_profiles_loads_conservative_and_aggressive() -> None:
    profiles = load_risk_profiles("configs/risk_profiles.yaml")

    assert "conservative" in profiles
    assert "aggressive" in profiles
    assert get_risk_profile(profiles, "conservative")["target_max_drawdown"] == pytest.approx(0.15)
    assert get_risk_profile(profiles, "aggressive")["target_max_drawdown"] == pytest.approx(0.30)


def test_validate_risk_profile_rejects_drawdown_outside_range() -> None:
    profile = valid_profile()
    profile["target_max_drawdown"] = 1.0

    with pytest.raises(RiskProfileError, match="target_max_drawdown"):
        validate_risk_profile(profile)


def test_validate_risk_profile_rejects_leverage_above_one_for_now() -> None:
    profile = valid_profile()
    profile["max_gross_leverage"] = 1.2

    with pytest.raises(RiskProfileError, match="max_gross_leverage"):
        validate_risk_profile(profile)


def test_build_cost_config_for_equity_converts_bps() -> None:
    config = build_cost_config_for_market(valid_profile(), MarketType.EQUITY)

    assert config.commission_rate == pytest.approx(0.0001)
    assert config.spread_rate == pytest.approx(0.0002)
    assert config.slippage_rate == pytest.approx(0.0002)


def test_build_cost_config_for_crypto_converts_bps() -> None:
    config = build_cost_config_for_market(valid_profile(), MarketType.CRYPTO)

    assert config.commission_rate == pytest.approx(0.001)
    assert config.spread_rate == pytest.approx(0.0005)
    assert config.slippage_rate == pytest.approx(0.001)


def test_build_cost_config_for_unknown_market_type_fails() -> None:
    with pytest.raises(ValueError):
        build_cost_config_for_market(valid_profile(), "unknown")
