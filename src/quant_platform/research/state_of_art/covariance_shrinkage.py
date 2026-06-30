"""Covariance shrinkage diagnostics for portfolio robustness."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.research.portfolio_optimization import long_only_weight_grid


def covariance_shrinkage_comparison(
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.0,
    grid_step: float = 0.05,
) -> pd.DataFrame:
    """Compare sample covariance against shrinkage covariance and optimized weights."""

    clean = _as_return_frame(returns)
    sample_cov = clean.cov() * periods_per_year
    shrink_cov, shrink_meta = ledoit_wolf_or_diagonal_covariance(
        clean,
        periods_per_year=periods_per_year,
    )
    mu = clean.mean() * periods_per_year
    rows = []
    for name, cov, meta in (
        ("sample_covariance", sample_cov, {"method": "sample_covariance"}),
        ("shrinkage_covariance", shrink_cov, shrink_meta),
    ):
        min_var = optimize_weights(
            mu,
            cov,
            objective="min_variance",
            risk_free_rate_annual=risk_free_rate_annual,
            grid_step=grid_step,
        )
        max_sharpe = optimize_weights(
            mu,
            cov,
            objective="max_sharpe",
            risk_free_rate_annual=risk_free_rate_annual,
            grid_step=grid_step,
        )
        for objective, weights in (
            ("min_variance", min_var),
            ("max_sharpe", max_sharpe),
        ):
            stats = portfolio_stats(mu, cov, weights, risk_free_rate_annual)
            rows.append(
                {
                    "covariance_estimator": name,
                    "optimizer_objective": objective,
                    "estimator_method": meta["method"],
                    "shrinkage_delta": meta.get("shrinkage_delta"),
                    "annualized_return_estimate": stats["return"],
                    "annualized_volatility_estimate": stats["volatility"],
                    "sharpe_estimate": stats["sharpe"],
                    "max_weight": float(weights.max()),
                    "effective_number_of_holdings": effective_number_of_holdings(weights),
                    "weights_json": weights.to_json(),
                }
            )
    return pd.DataFrame(rows)


def diagonal_shrinkage_covariance(
    returns_or_covariance: pd.DataFrame,
    *,
    delta: float = 0.50,
    periods_per_year: int = 252,
    input_is_covariance: bool = False,
) -> pd.DataFrame:
    """Shrink a covariance matrix toward its diagonal target."""

    if not 0.0 <= delta <= 1.0 or not np.isfinite(delta):
        raise ValueError("delta must be finite and in [0, 1].")
    matrix = returns_or_covariance.astype(float)
    if input_is_covariance:
        cov = matrix.copy()
    else:
        cov = _as_return_frame(matrix).cov() * periods_per_year
    diagonal = pd.DataFrame(
        np.diag(np.diag(cov.to_numpy(dtype=float))),
        index=cov.index,
        columns=cov.columns,
    )
    return (1.0 - delta) * cov + delta * diagonal


def ledoit_wolf_or_diagonal_covariance(
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    fallback_delta: float = 0.50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use sklearn Ledoit-Wolf when available, otherwise deterministic diagonal shrinkage."""

    clean = _as_return_frame(returns)
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore[import-not-found]

        estimator = LedoitWolf().fit(clean.to_numpy(dtype=float))
        covariance = pd.DataFrame(
            estimator.covariance_ * periods_per_year,
            index=clean.columns,
            columns=clean.columns,
        )
        return covariance, {
            "method": "sklearn_ledoit_wolf",
            "shrinkage_delta": float(estimator.shrinkage_),
        }
    except Exception:
        covariance = diagonal_shrinkage_covariance(
            clean,
            delta=fallback_delta,
            periods_per_year=periods_per_year,
        )
        return covariance, {
            "method": "fallback_diagonal_shrinkage",
            "shrinkage_delta": float(fallback_delta),
        }


def optimize_weights(
    annualized_returns: pd.Series,
    annualized_covariance: pd.DataFrame,
    *,
    objective: str,
    risk_free_rate_annual: float = 0.0,
    grid_step: float = 0.05,
) -> pd.Series:
    """Optimize long-only grid weights for min variance or max Sharpe."""

    assets = tuple(str(asset) for asset in annualized_covariance.index)
    grid = long_only_weight_grid(assets, step=grid_step)
    sigma = annualized_covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
    mu = annualized_returns.reindex(assets).to_numpy(dtype=float)
    best_score = float("inf") if objective == "min_variance" else -float("inf")
    best: pd.Series | None = None
    for _, row in grid.iterrows():
        weights = row.reindex(assets).astype(float)
        vector = weights.to_numpy(dtype=float)
        variance = float(vector @ sigma @ vector)
        volatility = math.sqrt(max(variance, 0.0))
        expected_return = float(vector @ mu)
        if objective == "min_variance":
            score = variance
            better = score < best_score
        elif objective == "max_sharpe":
            score = (
                (expected_return - risk_free_rate_annual) / volatility
                if volatility
                else -float("inf")
            )
            better = score > best_score
        else:
            raise ValueError("objective must be 'min_variance' or 'max_sharpe'.")
        if np.isfinite(score) and better:
            best_score = score
            best = weights
    if best is None:
        raise ValueError("No feasible finite portfolio weights found.")
    return best / float(best.sum())


def portfolio_stats(
    annualized_returns: pd.Series,
    annualized_covariance: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, float | None]:
    """Return estimated mean, volatility and Sharpe for annualized inputs."""

    aligned_weights = weights.reindex(annualized_covariance.index).astype(float)
    mu = annualized_returns.reindex(annualized_covariance.index).to_numpy(dtype=float)
    sigma = annualized_covariance.to_numpy(dtype=float)
    vector = aligned_weights.to_numpy(dtype=float)
    expected_return = float(vector @ mu)
    variance = float(vector @ sigma @ vector)
    volatility = math.sqrt(max(variance, 0.0))
    sharpe = (expected_return - risk_free_rate_annual) / volatility if volatility else None
    return {"return": expected_return, "volatility": volatility, "sharpe": sharpe}


def effective_number_of_holdings(weights: pd.Series) -> float | None:
    values = weights.astype(float).to_numpy(dtype=float)
    hhi = float(np.sum(values**2))
    return float(1.0 / hhi) if hhi > 0 else None


def _as_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty:
        raise ValueError("returns must contain finite aligned observations.")
    if clean.columns.has_duplicates:
        raise ValueError("returns columns must be unique.")
    if (clean <= -1.0).to_numpy().any():
        raise ValueError("simple returns must be greater than -100%.")
    return clean
