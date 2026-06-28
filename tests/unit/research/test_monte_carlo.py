from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.monte_carlo import (
    simulate_bootstrap_returns,
    simulate_gbm_prices,
    summarize_simulated_paths,
)


def test_bootstrap_returns_are_reproducible() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.0])

    first = simulate_bootstrap_returns(returns, horizon_days=5, n_paths=3, seed=7)
    second = simulate_bootstrap_returns(returns, horizon_days=5, n_paths=3, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 5)


def test_gbm_accepts_unbounded_log_returns_and_keeps_prices_positive() -> None:
    log_returns = pd.Series([-1.2, 0.01, 0.02, -0.03, 0.0])

    paths = simulate_gbm_prices(100.0, log_returns, horizon_days=4, n_paths=2, seed=11)

    assert paths.shape == (2, 4)
    assert np.isfinite(paths).all()
    assert (paths > 0).all()


def test_simulated_path_percentiles_are_ordered() -> None:
    paths = np.array([[0.01, 0.02], [-0.01, 0.03], [0.0, -0.02]])

    summary = summarize_simulated_paths(paths)
    first_step = summary["fan_chart"][0]

    assert first_step["p5"] <= first_step["p25"] <= first_step["p50"]
    assert first_step["p50"] <= first_step["p75"] <= first_step["p95"]
