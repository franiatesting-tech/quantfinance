"""Historical Value at Risk for positive losses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series


class RiskMetricError(ValueError):
    """Raised when risk metric inputs violate sign or data conventions."""


def _validate_alpha(alpha: float) -> float:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise RiskMetricError("alpha must be a finite confidence level in (0, 1).")
    return float(alpha)


def _validate_losses(losses: ArrayLike) -> np.ndarray:
    """Validate that losses are numeric, finite, non-empty.

    Convention: losses = -returns (standard convention, can be negative).
    A negative loss means a gain at the given confidence level.
    """
    try:
        values = np.asarray(losses, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RiskMetricError("losses must be numeric.") from exc
    if values.size == 0:
        raise RiskMetricError("losses must not be empty.")
    if not np.isfinite(values).all():
        raise RiskMetricError("losses must not contain NaN or infinite values.")
    return values


def historical_var(losses: ArrayLike, alpha: float = 0.95) -> float:
    """Compute historical VaR as the `alpha` quantile of positive losses.

    Convention: input `losses` are positive losses `L_t = -r_{p,t}`. The output is a
    positive number. `alpha` is the confidence level, not the tail probability.
    """

    clean_alpha = _validate_alpha(alpha)
    clean_losses = _validate_losses(losses)
    return float(np.quantile(clean_losses, clean_alpha))
