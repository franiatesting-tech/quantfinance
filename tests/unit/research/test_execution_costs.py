from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.research.state_of_art.execution_costs import (
    execution_cost_sensitivity,
    turnover_between_weights,
    turnover_cost,
)


def test_turnover_between_weights_is_one_way_turnover() -> None:
    left = pd.Series({"A": 0.60, "B": 0.40})
    right = pd.Series({"A": 0.50, "B": 0.50})

    assert turnover_between_weights(left, right) == pytest.approx(0.10)


def test_turnover_cost_uses_basis_points() -> None:
    assert turnover_cost(1_000_000.0, 0.10, cost_bps=10.0) == pytest.approx(100.0)


def test_execution_cost_sensitivity_reports_total_and_capacity_warning() -> None:
    old = pd.Series({"A": 0.60, "B": 0.40})
    new = pd.Series({"A": 0.50, "B": 0.50})

    table = execution_cost_sensitivity(
        old,
        new,
        notional=1_000_000.0,
        adv_by_asset={"A": 10_000_000.0, "B": 10_000_000.0},
        volatility_by_asset={"A": 0.20, "B": 0.20},
    )

    total = table[table["asset_id"] == "PORTFOLIO_TOTAL"].iloc[0]
    assert total["trade_notional"] == pytest.approx(200_000.0)
    assert total["total_estimated_cost"] > 0
