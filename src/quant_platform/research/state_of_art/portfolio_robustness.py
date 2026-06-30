"""Portfolio robustness diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_platform.research.state_of_art.covariance_shrinkage import (
    ledoit_wolf_or_diagonal_covariance,
    optimize_weights,
)
from quant_platform.research.state_of_art.execution_costs import turnover_between_weights


def portfolio_robustness_summary(
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    grid_step: float = 0.05,
    max_weight_limit: float = 0.35,
) -> pd.DataFrame:
    """Summarize weight stability under sample vs shrinkage covariance."""

    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or len(clean.columns) < 2:
        return pd.DataFrame()
    mu = clean.mean() * periods_per_year
    sample_cov = clean.cov() * periods_per_year
    shrink_cov, shrink_meta = ledoit_wolf_or_diagonal_covariance(
        clean,
        periods_per_year=periods_per_year,
    )
    rows = []
    for objective in ("min_variance", "max_sharpe"):
        sample_weights = optimize_weights(mu, sample_cov, objective=objective, grid_step=grid_step)
        shrink_weights = optimize_weights(mu, shrink_cov, objective=objective, grid_step=grid_step)
        stability = weight_stability(sample_weights, shrink_weights)
        rows.append(
            {
                "objective": objective,
                "shrinkage_method": shrink_meta["method"],
                "weight_l1_shift": stability["l1_shift"],
                "max_abs_weight_shift": stability["max_abs_shift"],
                "turnover_if_rebalanced": turnover_between_weights(sample_weights, shrink_weights),
                "sample_max_weight": float(sample_weights.max()),
                "shrinkage_max_weight": float(shrink_weights.max()),
                "sample_effective_holdings": effective_number_of_holdings(sample_weights),
                "shrinkage_effective_holdings": effective_number_of_holdings(shrink_weights),
                "max_weight_limit": max_weight_limit,
                "max_weight_breached": bool(shrink_weights.max() > max_weight_limit),
                "sample_weights_json": json.dumps(sample_weights.to_dict(), sort_keys=True),
                "shrinkage_weights_json": json.dumps(shrink_weights.to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def weight_stability(left: pd.Series, right: pd.Series) -> dict[str, float]:
    """Compare two weight vectors after aligning assets."""

    assets = sorted(set(left.index.astype(str)).union(set(right.index.astype(str))))
    lvec = left.reindex(assets).fillna(0.0).astype(float)
    rvec = right.reindex(assets).fillna(0.0).astype(float)
    diff = (lvec - rvec).abs()
    return {
        "l1_shift": float(diff.sum()),
        "max_abs_shift": float(diff.max()) if not diff.empty else 0.0,
    }


def effective_number_of_holdings(weights: pd.Series) -> float | None:
    values = weights.astype(float).to_numpy(dtype=float)
    hhi = float(np.sum(values**2))
    return float(1.0 / hhi) if hhi > 0 else None
