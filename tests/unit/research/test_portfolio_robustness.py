from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.research.state_of_art.portfolio_robustness import (
    portfolio_robustness_summary,
    weight_stability,
)


def test_weight_stability_reports_l1_and_max_shift() -> None:
    left = pd.Series({"A": 0.60, "B": 0.40})
    right = pd.Series({"A": 0.50, "B": 0.50})

    result = weight_stability(left, right)

    assert result["l1_shift"] == pytest.approx(0.20)
    assert result["max_abs_shift"] == pytest.approx(0.10)


def test_portfolio_robustness_summary_contains_objectives() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.00, 0.01, 0.03],
            "B": [0.00, 0.01, -0.02, 0.01, 0.02, 0.01],
            "C": [0.01, -0.01, 0.00, 0.01, -0.02, 0.02],
        }
    )

    table = portfolio_robustness_summary(returns, grid_step=0.5)

    assert set(table["objective"]) == {"min_variance", "max_sharpe"}
    assert (table["turnover_if_rebalanced"] >= 0).all()
