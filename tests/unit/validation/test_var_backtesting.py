from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.validation.var_backtesting import (
    VaRBacktestingError,
    binomial_exception_probability,
    exception_count,
    exception_rate,
    expected_exception_rate,
    var_exceptions,
)


def test_var_exceptions_for_known_losses_and_var() -> None:
    losses = pd.Series([0.01, 0.03, 0.02, 0.05])
    var_forecast = pd.Series([0.02, 0.02, 0.02, 0.04])

    exceptions = var_exceptions(losses, var_forecast)

    assert exceptions.tolist() == [0, 1, 0, 1]


def test_exception_count_and_rate() -> None:
    exceptions = pd.Series([0, 1, 0, 1])

    assert exception_count(exceptions) == 2
    assert exception_rate(exceptions) == pytest.approx(0.5)


def test_expected_exception_rate_for_95_confidence() -> None:
    assert expected_exception_rate(0.95) == pytest.approx(0.05)


def test_binomial_exception_probability_exact_value() -> None:
    assert binomial_exception_probability(n=4, k=2, p=0.5) == pytest.approx(0.375)


def test_var_exceptions_reject_misaligned_sizes() -> None:
    with pytest.raises(VaRBacktestingError, match="same length"):
        var_exceptions([0.01, 0.02], [0.01])


def test_var_exceptions_reject_negative_values() -> None:
    with pytest.raises(VaRBacktestingError, match="non-negative"):
        var_exceptions([0.01, -0.02], [0.01, 0.01])
