"""Drawdown calculations for equity curves."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

PandasObject: TypeAlias = pd.Series | pd.DataFrame


class DrawdownError(ValueError):
    """Raised when an equity curve violates drawdown input conventions."""


def _as_positive_equity_curve(equity_curve: PandasObject) -> PandasObject:
    if not isinstance(equity_curve, pd.Series | pd.DataFrame):
        raise TypeError("equity_curve must be a pandas Series or DataFrame.")
    if equity_curve.empty:
        raise DrawdownError("equity_curve must not be empty.")
    try:
        values = equity_curve.astype(float)
    except (TypeError, ValueError) as exc:
        raise DrawdownError("equity_curve must be numeric.") from exc
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise DrawdownError("equity_curve must not contain NaN or infinite values.")
    if (values <= 0).to_numpy().any():
        raise DrawdownError("equity_curve values must be strictly positive.")
    return values


def drawdown(equity_curve: PandasObject) -> PandasObject:
    """Compute drawdown `DD_t = V_t / max_{s<=t}(V_s) - 1`.

    Input is an equity curve, not raw asset prices unless those prices represent a
    buy-and-hold equity curve.
    """

    clean_equity = _as_positive_equity_curve(equity_curve)
    running_peak = clean_equity.cummax()
    return clean_equity / running_peak - 1.0


def max_drawdown(equity_curve: PandasObject) -> float | pd.Series:
    """Return the minimum drawdown over time.

    For a Series, the result is a scalar. For a DataFrame, the result is one value per
    column.
    """

    return drawdown(equity_curve).min()
