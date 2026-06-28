from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.monte_carlo import simulate_portfolio_normal_returns


def test_portfolio_monte_carlo_shape_and_seed() -> None:
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, -0.01, 0.0], "B": [0.0, 0.01, 0.02, -0.01]}
    )
    weights = {"A": 0.6, "B": 0.4}

    first = simulate_portfolio_normal_returns(returns, weights, 6, 4, seed=3)
    second = simulate_portfolio_normal_returns(returns, weights, 6, 4, seed=3)

    assert first.shape == (4, 6)
    np.testing.assert_allclose(first, second)
