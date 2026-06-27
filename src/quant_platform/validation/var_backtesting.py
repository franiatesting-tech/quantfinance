"""Basic VaR exception backtesting for positive losses."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series


class VaRBacktestingError(ValueError):
    """Raised when VaR backtesting inputs are invalid."""


def _as_non_negative_series(values: ArrayLike, name: str) -> pd.Series:
    try:
        series = (
            values.astype(float)
            if isinstance(values, pd.Series)
            else pd.Series(values, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise VaRBacktestingError(f"{name} must be numeric.") from exc
    if series.empty:
        raise VaRBacktestingError(f"{name} must not be empty.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise VaRBacktestingError(f"{name} must not contain NaN or infinite values.")
    if (series < 0).any():
        raise VaRBacktestingError(f"{name} must contain non-negative positive-loss values.")
    return series


def _as_exception_series(exceptions: ArrayLike) -> pd.Series:
    try:
        series = (
            exceptions.astype(int)
            if isinstance(exceptions, pd.Series)
            else pd.Series(exceptions, dtype=int)
        )
    except (TypeError, ValueError) as exc:
        raise VaRBacktestingError("exceptions must be numeric 0/1 values.") from exc
    if series.empty:
        raise VaRBacktestingError("exceptions must not be empty.")
    if not set(series.unique()).issubset({0, 1}):
        raise VaRBacktestingError("exceptions must contain only 0/1 values.")
    return series


def _validate_alpha(alpha: float) -> float:
    clean_alpha = float(alpha)
    if not np.isfinite(clean_alpha) or not 0 < clean_alpha < 1:
        raise VaRBacktestingError("alpha must be a finite confidence level in (0, 1).")
    return clean_alpha


def var_exceptions(losses: ArrayLike, var_forecast: ArrayLike) -> pd.Series:
    """Return `1` when positive loss `L_t` exceeds same-horizon `VaR_t`."""

    clean_losses = _as_non_negative_series(losses, "losses")
    clean_var = _as_non_negative_series(var_forecast, "var_forecast")
    if len(clean_losses) != len(clean_var):
        raise VaRBacktestingError("losses and var_forecast must have the same length.")
    if isinstance(losses, pd.Series) and isinstance(var_forecast, pd.Series):
        if not clean_losses.index.equals(clean_var.index):
            raise VaRBacktestingError("losses and var_forecast indices must match.")
    clean_var.index = clean_losses.index
    return (clean_losses > clean_var).astype(int)


def exception_count(exceptions: ArrayLike) -> int:
    """Count VaR exceptions."""

    return int(_as_exception_series(exceptions).sum())


def exception_rate(exceptions: ArrayLike) -> float:
    """Return observed exception frequency."""

    clean_exceptions = _as_exception_series(exceptions)
    return float(clean_exceptions.mean())


def expected_exception_rate(alpha: float) -> float:
    """Return expected exception rate `1 - alpha`."""

    return 1.0 - _validate_alpha(alpha)


def binomial_exception_probability(n: int, k: int, p: float) -> float:
    """Return exact binomial probability of observing `k` exceptions in `n` trials."""

    clean_n = int(n)
    clean_k = int(k)
    clean_p = float(p)
    if clean_n < 0:
        raise VaRBacktestingError("n must be >= 0.")
    if clean_k < 0 or clean_k > clean_n:
        raise VaRBacktestingError("k must satisfy 0 <= k <= n.")
    if not np.isfinite(clean_p) or not 0 <= clean_p <= 1:
        raise VaRBacktestingError("p must be a finite probability in [0, 1].")
    probability = math.comb(clean_n, clean_k) * clean_p**clean_k
    probability *= (1.0 - clean_p) ** (clean_n - clean_k)
    return float(probability)


def binomial_two_sided_p_value(n: int, k: int, p: float) -> float:
    """Return a conservative exact two-sided binomial p-value.

    The p-value sums probabilities of outcomes whose exact probability is less
    than or equal to the observed outcome probability. This avoids SciPy while
    remaining deterministic for small validation samples.
    """

    observed_probability = binomial_exception_probability(n, k, p)
    total = 0.0
    for outcome in range(int(n) + 1):
        probability = binomial_exception_probability(n, outcome, p)
        if probability <= observed_probability + 1e-15:
            total += probability
    return float(min(total, 1.0))
