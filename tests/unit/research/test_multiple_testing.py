from __future__ import annotations

import pytest

from quant_platform.research.state_of_art.multiple_testing import (
    deflated_sharpe_ratio_approximation,
    probabilistic_sharpe_ratio,
)


def test_probabilistic_sharpe_ratio_is_high_for_positive_sharpe() -> None:
    psr = probabilistic_sharpe_ratio(1.0, benchmark_sharpe=0.0, observations=252)

    assert psr > 0.99


def test_deflated_sharpe_ratio_applies_best_of_n_haircut() -> None:
    result = deflated_sharpe_ratio_approximation(
        1.0,
        observations=252,
        number_of_trials=10,
    )

    assert result["implementation_status"] == "PARTIAL_IMPLEMENTATION_DSR_APPROXIMATION"
    assert result["sharpe_haircut"] > 0
    assert result["deflated_sharpe"] == pytest.approx(1.0 - result["sharpe_haircut"])
    assert result["selection_bias_warning"] is True
