from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.features.returns import (
    ReturnCalculationError,
    align_features_and_forward_returns,
    cumulative_returns,
    excess_returns,
    log_returns,
    simple_returns,
)


def test_simple_returns_for_manual_prices() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])

    result = simple_returns(prices)

    expected = pd.Series([np.nan, 0.10, 0.10])
    pd.testing.assert_series_equal(result, expected)


def test_log_returns_for_manual_prices() -> None:
    prices = pd.Series([100.0, 110.0, 121.0])

    result = log_returns(prices)

    expected = pd.Series([np.nan, np.log(110.0 / 100.0), np.log(121.0 / 110.0)])
    pd.testing.assert_series_equal(result, expected)


def test_cumulative_returns_for_manual_returns() -> None:
    returns = pd.Series([0.10, -0.10])

    result = cumulative_returns(returns, initial_value=1.0)

    expected = pd.Series([1.10, 0.99])
    pd.testing.assert_series_equal(result, expected)


def test_excess_returns_subtracts_scalar_periodic_rate() -> None:
    returns = pd.Series([0.02, 0.01])

    result = excess_returns(returns, 0.005)

    expected = pd.Series([0.015, 0.005])
    pd.testing.assert_series_equal(result, expected)


def test_log_returns_reject_zero_or_negative_prices() -> None:
    prices = pd.Series([100.0, 0.0, -1.0])

    with pytest.raises(ReturnCalculationError, match="strictly positive"):
        log_returns(prices)


def test_align_features_and_forward_returns_uses_future_target_without_future_features() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0]}, index=index)
    realized_returns = pd.Series([0.01, 0.02, 0.03], index=index)

    aligned_features, forward_returns = align_features_and_forward_returns(
        features,
        realized_returns,
        horizon=1,
    )

    expected_features = features.iloc[:2]
    expected_forward_returns = pd.Series([0.02, 0.03], index=index[:2])
    pd.testing.assert_frame_equal(aligned_features, expected_features)
    pd.testing.assert_series_equal(forward_returns, expected_forward_returns)


def test_align_features_and_forward_returns_rejects_invalid_horizon() -> None:
    features = pd.Series([1.0, 2.0])
    returns = pd.Series([0.1, 0.2])

    with pytest.raises(ReturnCalculationError, match="horizon"):
        align_features_and_forward_returns(features, returns, horizon=0)
