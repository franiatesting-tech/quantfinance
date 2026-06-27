"""Historical Expected Shortfall for positive losses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_platform.risk.var import (
    RiskMetricError,
    _validate_alpha,
    _validate_losses,
    historical_var,
)

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def historical_expected_shortfall(losses: ArrayLike, alpha: float = 0.95) -> float:
    """Compute empirical ES as mean tail loss beyond historical VaR.

    Convention: input `losses` are positive losses `L_t = -r_{p,t}`. The implemented
    finite-sample estimator is `mean(losses | losses >= VaR_alpha(losses))`, consistent
    with the roadmap's initial historical ES requirement and Acerbi-Tasche's tail-loss
    interpretation for sample worst cases. Advanced discontinuity treatment remains a
    validation task for later iterations.
    """

    clean_alpha = _validate_alpha(alpha)
    clean_losses = _validate_losses(losses)
    var_value = historical_var(clean_losses, clean_alpha)
    tail_losses = clean_losses[clean_losses >= var_value]
    if tail_losses.size == 0:
        raise RiskMetricError("Expected Shortfall tail is empty after VaR filtering.")
    return float(tail_losses.mean())
