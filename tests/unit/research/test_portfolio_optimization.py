from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.research.portfolio import portfolio_returns
from quant_platform.research.portfolio_optimization import (
    long_only_weight_grid,
    optimize_long_only_portfolio,
)


def test_long_only_grid_weights_sum_to_one() -> None:
    grid = long_only_weight_grid(("A", "B", "C"), step=0.5)

    assert len(grid) == 6
    assert grid.sum(axis=1).tolist() == pytest.approx([1.0] * 6)
    assert (grid >= 0).all().all()


def test_optimize_long_only_portfolio_returns_min_variance_and_max_sharpe() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.01, 0.01, 0.01],
            "B": [0.02, -0.01, 0.02, -0.01],
            "C": [0.0, 0.0, 0.01, 0.0],
        }
    )

    result = optimize_long_only_portfolio(returns, step=0.5)
    max_sharpe_returns = portfolio_returns(returns, result["max_sharpe"]["weights"])

    assert result["method"] == "long_only_grid_search"
    assert result["portfolio_count"] == 6
    assert sum(result["min_variance"]["weights"].values()) == pytest.approx(1.0)
    assert sum(result["max_sharpe"]["weights"].values()) == pytest.approx(1.0)
    assert len(max_sharpe_returns) == len(returns)
