from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.research.state_of_art.tail_risk_backtesting import (
    backtest_var,
    christoffersen_independence,
    kupiec_unconditional_coverage,
    var_exceptions,
)


def test_var_exceptions_use_positive_loss_convention() -> None:
    returns = pd.Series([0.01, -0.02, -0.10, 0.03])

    exceptions = var_exceptions(returns, 0.05)

    assert exceptions.tolist() == [False, False, True, False]


def test_kupiec_passes_when_exception_rate_matches_expected_rate() -> None:
    exceptions = pd.Series([True] * 5 + [False] * 95)

    result = kupiec_unconditional_coverage(exceptions, alpha=0.95)

    assert result["lr_uc"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["status"] == "PASS"


def test_christoffersen_returns_transition_counts() -> None:
    exceptions = pd.Series([False, True, False, False, True, False, False])

    result = christoffersen_independence(exceptions)

    assert result["n01"] == 2
    assert result["n10"] == 2
    assert result["status"] in {"PASS", "REJECT"}


def test_backtest_var_reports_traffic_light() -> None:
    returns = pd.Series([-0.10] * 5 + [0.001] * 95)

    result = backtest_var(returns, 0.05, alpha=0.95)

    assert result["exceptions"] == 5
    assert result["traffic_light"] == "GREEN_EXCEPTION_RATE_WITHIN_TOLERANCE"
    assert result["loss_convention"].startswith("L_t = -R_t")
