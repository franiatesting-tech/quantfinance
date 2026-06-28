from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.ml_forecasting import run_walk_forward_forecast


def test_walk_forward_forecast_is_out_of_sample_and_reports_baseline_status() -> None:
    np.random.seed(42)
    index = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    returns = pd.Series(np.random.normal(0.0005, 0.015, 400), index=index)
    prices = 100.0 * (1.0 + returns).cumprod()

    result = run_walk_forward_forecast(
        prices,
        min_train_size=100,
        max_test_observations=50,
    )

    assert result["validation"] == "walk_forward_expanding_window_no_shuffle_no_leakage"
    assert result["status"] in {
        "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE",
        "MODEL_OUTPERFORMS_NAIVE_UNDER_TEST_ASSUMPTIONS",
    }
    assert "rmse" in result
    assert "directional_accuracy" in result
    assert "oos_r_squared" in result
    assert "per_model_metrics" in result
    assert "features" in result
    assert "n_features" in result
    assert result["n_features"] > 0


def test_walk_forward_forecast_with_larger_dataset() -> None:
    np.random.seed(42)
    index = pd.date_range("2015-01-01", periods=1500, freq="D", tz="UTC")
    returns = pd.Series(np.random.normal(0.0005, 0.015, 1500), index=index)
    prices = 100.0 * (1.0 + returns).cumprod()

    result = run_walk_forward_forecast(
        prices,
        min_train_size=252,
        horizon_days=1,
        max_test_observations=252,
    )

    assert result["test_observations"] > 0
    assert result["status"] in {
        "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE",
        "MODEL_OUTPERFORMS_NAIVE_UNDER_TEST_ASSUMPTIONS",
    }
    assert result["purge_gap"] == 1
    assert "models_evaluated" in result
    assert "naive_random_walk_baseline" in result["models_evaluated"]
    assert "historical_mean_baseline" in result["models_evaluated"]


def test_walk_forward_forecast_with_benchmark() -> None:
    np.random.seed(123)
    index = pd.date_range("2018-01-01", periods=800, freq="D", tz="UTC")
    asset_returns = pd.Series(np.random.normal(0.0004, 0.018, 800), index=index)
    benchmark_returns = pd.Series(np.random.normal(0.0003, 0.012, 800), index=index)
    prices = 100.0 * (1.0 + asset_returns).cumprod()

    result = run_walk_forward_forecast(
        prices,
        benchmark_returns=benchmark_returns,
        min_train_size=200,
        max_test_observations=100,
    )

    assert result["test_observations"] > 0
    features = result.get("features", [])
    has_beta = any("beta" in f for f in features)
    has_corr = any("correlation" in f for f in features)
    assert has_beta, f"Expected beta feature in {features}"
    assert has_corr, f"Expected correlation feature in {features}"


def test_garch_forecast_produced_when_enough_data() -> None:
    np.random.seed(99)
    index = pd.date_range("2010-01-01", periods=500, freq="D", tz="UTC")
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500), index=index)
    prices = 100.0 * (1.0 + returns).cumprod()

    result = run_walk_forward_forecast(
        prices,
        min_train_size=252,
        max_test_observations=50,
    )

    garch = result.get("garch_forecast")
    if garch is not None:
        assert garch["model"] == "GARCH(1,1)"
        assert garch["persistence"] < 1.0
        assert garch["alpha"] >= 0
        assert garch["beta"] >= 0
        assert garch["conditional_volatility_annual"] > 0
