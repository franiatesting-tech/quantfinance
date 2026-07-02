"""GARCH(1,1) volatility diagnostics for research-only risk studies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class GARCHVolatilityError(ValueError):
    """Raised when GARCH diagnostics cannot be computed safely."""


def fit_garch_11(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    max_series_rows: int = 252,
) -> dict[str, Any]:
    """Fit a Gaussian GARCH(1,1) model with constrained maximum likelihood."""

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - exercised only without scipy installed.
        raise GARCHVolatilityError("scipy is required for GARCH optimization.") from exc
    clean = _as_returns(returns)
    if len(clean) < 100:
        raise GARCHVolatilityError("returns must contain at least 100 observations.")
    values = clean.to_numpy(dtype=float)
    mean_return = float(np.mean(values))
    eps = values - mean_return
    sample_var = float(np.var(eps, ddof=1))
    if sample_var <= 0 or not np.isfinite(sample_var):
        raise GARCHVolatilityError("returns must have positive variance.")

    def neg_loglik(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
            return 1e20
        sigma2 = _conditional_variance(eps, omega, alpha, beta, sample_var)
        if not np.isfinite(sigma2).all() or (sigma2 <= 0.0).any():
            return 1e20
        return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + eps**2 / sigma2))

    x0 = np.array([sample_var * 0.05, 0.08, 0.88], dtype=float)
    bounds = [(1e-12, sample_var * 10.0), (1e-8, 0.998), (1e-8, 0.998)]
    result = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise GARCHVolatilityError(f"GARCH optimizer failed: {result.message}")
    omega, alpha, beta = [float(value) for value in result.x]
    if alpha + beta >= 0.999:
        raise GARCHVolatilityError("GARCH persistence is non-stationary.")
    sigma2 = _conditional_variance(eps, omega, alpha, beta, sample_var)
    forecast_var = float(omega + alpha * eps[-1] ** 2 + beta * sigma2[-1])
    persistence = alpha + beta
    half_life = float(math.log(0.5) / math.log(persistence)) if 0.0 < persistence < 1.0 else None
    series_rows = [
        {
            "timestamp": pd.Timestamp(idx).isoformat(),
            "conditional_volatility_daily": float(math.sqrt(var)),
            "conditional_volatility_annual": float(math.sqrt(var) * math.sqrt(periods_per_year)),
        }
        for idx, var in zip(clean.index[-max_series_rows:], sigma2[-max_series_rows:])
    ]
    return {
        "model": "GARCH(1,1)",
        "observations": int(len(clean)),
        "mean_return_daily": mean_return,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": float(persistence),
        "half_life_days": half_life,
        "unconditional_volatility_daily": float(math.sqrt(omega / (1.0 - persistence))),
        "forecast_volatility_daily": float(math.sqrt(forecast_var)),
        "forecast_volatility_annual": float(math.sqrt(forecast_var) * math.sqrt(periods_per_year)),
        "log_likelihood": float(-result.fun),
        "volatility_series": series_rows,
        "model_status": "GARCH_11_GAUSSIAN_RESEARCH_ESTIMATE",
    }


def _conditional_variance(
    eps: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    initial_var: float,
) -> np.ndarray:
    sigma2 = np.empty(len(eps), dtype=float)
    sigma2[0] = max(initial_var, 1e-12)
    for index in range(1, len(eps)):
        sigma2[index] = omega + alpha * eps[index - 1] ** 2 + beta * sigma2[index - 1]
    return sigma2


def _as_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise GARCHVolatilityError("returns must be a non-empty Series.")
    clean = returns.astype(float).dropna()
    if clean.empty or not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean <= -1.0).any():
        raise GARCHVolatilityError("returns must be finite and above -100%.")
    return clean
