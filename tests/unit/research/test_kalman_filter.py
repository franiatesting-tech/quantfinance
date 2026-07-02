from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.state_of_art.kalman_filter import kalman_dynamic_regression


def test_kalman_dynamic_regression_tracks_beta() -> None:
    rng = np.random.default_rng(13)
    index = pd.date_range("2021-01-01", periods=160, freq="B")
    x = pd.Series(np.linspace(1.0, 2.5, len(index)), index=index)
    y = 0.4 + 1.7 * x + pd.Series(rng.normal(0.0, 0.02, len(index)), index=index)

    result = kalman_dynamic_regression(y, x, transition_covariance=1e-6)

    assert result["observations"] == len(index)
    assert abs(result["latest_beta"] - 1.7) < 0.2
    assert result["state_rows"]
