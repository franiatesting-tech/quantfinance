from __future__ import annotations

import pytest

from quant_platform.risk.var import RiskMetricError, historical_var


def test_historical_var_uses_alpha_quantile_of_positive_losses() -> None:
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert historical_var(losses, alpha=0.8) == pytest.approx(4.2)


def test_historical_var_rejects_negative_losses() -> None:
    with pytest.raises(RiskMetricError, match="positive losses"):
        historical_var([1.0, -1.0, 2.0], alpha=0.95)


def test_historical_var_rejects_invalid_alpha() -> None:
    with pytest.raises(RiskMetricError, match="alpha"):
        historical_var([1.0, 2.0], alpha=1.0)


def test_historical_var_rejects_nan_or_inf() -> None:
    with pytest.raises(RiskMetricError, match="NaN"):
        historical_var([1.0, float("inf")], alpha=0.95)
