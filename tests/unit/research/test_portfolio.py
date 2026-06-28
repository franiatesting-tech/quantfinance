from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.portfolio import portfolio_returns


def test_portfolio_returns_use_weighted_rebalanced_returns() -> None:
    returns = pd.DataFrame({"A": [0.10, -0.05], "B": [0.0, 0.02]})

    result = portfolio_returns(returns, {"A": 0.25, "B": 0.75})

    np.testing.assert_allclose(result.to_numpy(), [0.025, 0.0025])
