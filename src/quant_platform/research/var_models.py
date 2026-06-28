"""VaR and Expected Shortfall models for the terminal."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from quant_platform.risk.expected_shortfall import historical_expected_shortfall
from quant_platform.risk.var import historical_var


class VaRModelError(ValueError):
    """Raised when VaR model inputs are invalid."""


def compute_var_summary(
    returns: pd.Series,
    alpha: float = 0.95,
    simulated_returns: pd.Series | np.ndarray | None = None,
) -> dict[str, object]:
    """Compute historical, normal parametric, and optional MC VaR/ES."""

    clean_returns = _as_return_series(returns)
    losses = -clean_returns
    summary: dict[str, object] = {
        "alpha": float(alpha),
        "loss_sign_convention": "L_t = -r_t (standard, no clip)",
        "historical": {
            "var": historical_var(losses, alpha),
            "expected_shortfall": historical_expected_shortfall(losses, alpha),
        },
        "parametric_normal": parametric_normal_var_es(clean_returns, alpha),
    }
    if simulated_returns is not None:
        clean_simulated = _as_return_series(pd.Series(simulated_returns, dtype=float))
        simulated_losses = -clean_simulated
        summary["monte_carlo"] = {
            "var": historical_var(simulated_losses, alpha),
            "expected_shortfall": historical_expected_shortfall(simulated_losses, alpha),
            "model_status": "PARAMETRIC_OR_BOOTSTRAP_SIMULATION",
        }
    return summary


def compute_var_comparison_table(
    returns: pd.Series,
    alphas: tuple[float, ...] = (0.95, 0.99),
    simulated_returns: pd.Series | np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """Return long-form VaR/ES rows for historical, normal, and MC methods."""

    rows: list[dict[str, float | str]] = []
    for alpha in alphas:
        summary = compute_var_summary(returns, alpha=alpha, simulated_returns=simulated_returns)
        for method in ("historical", "parametric_normal", "monte_carlo"):
            payload = summary.get(method)
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "alpha": float(alpha),
                    "method": method,
                    "var": float(payload["var"]),
                    "expected_shortfall": float(payload["expected_shortfall"]),
                    "loss_sign_convention": "L_t = -R_t",
                }
            )
    return rows


def parametric_normal_var_es(returns: pd.Series, alpha: float = 0.95) -> dict[str, float | str]:
    """Compute normal VaR/ES for losses implied by normally distributed returns."""

    if not 0 < alpha < 1:
        raise VaRModelError("alpha must be in (0, 1).")
    clean_returns = _as_return_series(returns)
    mean = float(clean_returns.mean())
    sigma = float(clean_returns.std(ddof=1))
    if sigma < 0 or not np.isfinite(sigma):
        raise VaRModelError("return volatility must be finite.")
    normal = NormalDist()
    z = normal.inv_cdf(alpha)
    pdf = float(np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi))
    var_value = -mean + sigma * z
    es_value = -mean + sigma * pdf / (1.0 - alpha)
    return {
        "var": float(var_value),
        "expected_shortfall": float(es_value),
        "mean_return": mean,
        "volatility": sigma,
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
    }


def _as_return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise VaRModelError("returns must be a non-empty Series.")
    clean = returns.astype(float).dropna()
    if clean.empty:
        raise VaRModelError("returns must contain finite observations.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean < -1).any():
        raise VaRModelError("returns must be finite and above -100%.")
    return clean
