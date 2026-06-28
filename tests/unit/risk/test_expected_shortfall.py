from __future__ import annotations

import pytest

from quant_platform.risk.expected_shortfall import historical_expected_shortfall
from quant_platform.risk.var import RiskMetricError


def test_historical_expected_shortfall_uses_mean_tail_loss() -> None:
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert historical_expected_shortfall(losses, alpha=0.8) == pytest.approx(5.0)


def test_historical_expected_shortfall_includes_losses_at_var_threshold() -> None:
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert historical_expected_shortfall(losses, alpha=0.5) == pytest.approx(4.0)


def test_historical_expected_shortfall_accepts_negative_losses() -> None:
    """Negative losses (gains) are valid under the standard convention."""
    result = historical_expected_shortfall([1.0, -0.5, 2.0], alpha=0.95)
    assert isinstance(result, float)


def test_historical_expected_shortfall_rejects_empty_losses() -> None:
    with pytest.raises(RiskMetricError, match="must not be empty"):
        historical_expected_shortfall([], alpha=0.95)
