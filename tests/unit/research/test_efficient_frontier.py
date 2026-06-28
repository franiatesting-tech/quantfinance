from __future__ import annotations

import numpy as np

from quant_platform.research.efficient_frontier import capital_allocation_line


def test_capital_allocation_line_starts_at_risk_free_rate() -> None:
    tangent = {"annualized_volatility": 0.2, "annualized_return": 0.1}

    rows = capital_allocation_line(tangent, risk_free_rate_annual=0.02, points=3)

    assert rows[0] == {"annualized_volatility": 0.0, "annualized_return": 0.02}
    assert np.isclose(rows[-1]["annualized_volatility"], 0.3)
