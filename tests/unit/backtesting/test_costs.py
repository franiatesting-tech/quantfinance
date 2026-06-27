from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.backtesting.costs import (
    TransactionCostConfig,
    TransactionCostError,
    apply_transaction_costs,
    bps_to_rate,
    portfolio_turnover,
    rate_to_bps,
    transaction_costs_from_turnover,
)


def test_portfolio_turnover_compares_against_previous_weights() -> None:
    weights = pd.DataFrame({"A": [0.5, 0.7], "B": [0.5, 0.3]})

    result = portfolio_turnover(weights)

    assert result.tolist() == pytest.approx([1.0, 0.4])


def test_transaction_costs_from_turnover_use_half_spread() -> None:
    config = TransactionCostConfig(commission_rate=0.001, spread_rate=0.002, slippage_rate=0.0005)

    costs = transaction_costs_from_turnover(pd.Series([0.4]), config)

    assert costs.iloc[0] == pytest.approx(0.001)


def test_apply_transaction_costs_subtracts_costs_and_funding() -> None:
    gross_returns = pd.Series([0.01], index=pd.Index(["2024-01-01"]))
    turnover = pd.Series([0.4], index=gross_returns.index)
    config = TransactionCostConfig(
        commission_rate=0.001,
        spread_rate=0.002,
        slippage_rate=0.0005,
        funding_rate=0.0002,
    )

    net_returns = apply_transaction_costs(gross_returns, turnover, config)

    assert net_returns.iloc[0] == pytest.approx(0.0088)


def test_transaction_cost_config_rejects_negative_rates() -> None:
    with pytest.raises(TransactionCostError, match="commission_rate"):
        TransactionCostConfig(commission_rate=-0.001)


def test_bps_to_rate_converts_basis_points() -> None:
    assert bps_to_rate(10.0) == pytest.approx(0.001)


def test_rate_to_bps_converts_decimal_rate() -> None:
    assert rate_to_bps(0.001) == pytest.approx(10.0)


def test_transaction_cost_config_from_bps_builds_decimal_rates() -> None:
    config = TransactionCostConfig.from_bps(
        commission_bps=10.0,
        spread_bps=20.0,
        slippage_bps=5.0,
    )

    assert config.commission_rate == pytest.approx(0.001)
    assert config.spread_rate == pytest.approx(0.002)
    assert config.slippage_rate == pytest.approx(0.0005)
    assert config.per_turnover_rate == pytest.approx(0.0025)


def test_negative_bps_fails() -> None:
    with pytest.raises(TransactionCostError, match="bps"):
        bps_to_rate(-1.0)


def test_negative_rate_to_bps_fails() -> None:
    with pytest.raises(TransactionCostError, match="rate"):
        rate_to_bps(-0.001)
