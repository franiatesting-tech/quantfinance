from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.state_of_art.garch_volatility import fit_garch_11


def test_fit_garch_11_returns_stationary_forecast() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(0.0002, 0.012, size=320)
    returns = pd.Series(values, index=pd.date_range("2020-01-01", periods=320, freq="B"))

    result = fit_garch_11(returns)

    assert result["model"] == "GARCH(1,1)"
    assert 0.0 <= result["alpha"] < 1.0
    assert 0.0 <= result["beta"] < 1.0
    assert result["persistence"] < 0.999
    assert result["forecast_volatility_annual"] > 0
    assert result["volatility_series"]
