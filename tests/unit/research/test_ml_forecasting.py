from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.ml_forecasting import run_walk_forward_forecast


def test_walk_forward_forecast_is_out_of_sample_and_reports_baseline_status() -> None:
    index = pd.date_range("2020-01-01", periods=140, freq="D", tz="UTC")
    returns = pd.Series(np.tile([0.002, -0.001, 0.001, -0.0005], 35), index=index)
    prices = 100.0 * (1.0 + returns).cumprod()

    result = run_walk_forward_forecast(
        prices,
        min_train_size=40,
        max_test_observations=20,
    )

    assert result["validation"] == "walk_forward_no_shuffle_no_leakage"
    assert result["test_observations"] <= 20
    assert result["status"] in {
        "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE",
        "MODEL_OUTPERFORMS_NAIVE_UNDER_TEST_ASSUMPTIONS",
    }
    assert "rmse" in result
    assert "directional_accuracy" in result
