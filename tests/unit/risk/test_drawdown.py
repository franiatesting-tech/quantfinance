from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.risk.drawdown import DrawdownError, drawdown, max_drawdown


def test_drawdown_for_manual_equity_curve() -> None:
    equity_curve = pd.Series([100.0, 120.0, 90.0, 150.0])

    result = drawdown(equity_curve)

    expected = pd.Series([0.0, 0.0, -0.25, 0.0])
    pd.testing.assert_series_equal(result, expected)


def test_max_drawdown_for_manual_equity_curve() -> None:
    equity_curve = pd.Series([100.0, 120.0, 90.0, 150.0])

    assert max_drawdown(equity_curve) == pytest.approx(-0.25)


def test_drawdown_rejects_non_positive_equity() -> None:
    equity_curve = pd.Series([100.0, 0.0, 90.0])

    with pytest.raises(DrawdownError, match="strictly positive"):
        drawdown(equity_curve)


def test_drawdown_rejects_nan_equity() -> None:
    equity_curve = pd.Series([100.0, float("nan")])

    with pytest.raises(DrawdownError, match="NaN"):
        drawdown(equity_curve)
