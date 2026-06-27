"""Long-only grid-search portfolio optimization for three-stock portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.research.returns import annual_rate_to_periodic


class PortfolioOptimizationError(ValueError):
    """Raised when optimization inputs are invalid."""


def long_only_weight_grid(
    asset_ids: tuple[str, ...] | list[str],
    step: float = 0.05,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """Build a deterministic long-only weight grid that sums to one."""

    assets = tuple(str(asset) for asset in asset_ids)
    if not assets or len(set(assets)) != len(assets):
        raise PortfolioOptimizationError("asset_ids must be non-empty and unique.")
    if not np.isfinite(step) or step <= 0 or step > 1:
        raise PortfolioOptimizationError("step must be in (0, 1].")
    if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
        raise PortfolioOptimizationError("max_weight must be in (0, 1].")
    slots = int(round(1.0 / step))
    if not np.isclose(slots * step, 1.0):
        raise PortfolioOptimizationError("step must divide 1.0 exactly, e.g. 0.05.")
    max_slots = int(np.floor(max_weight / step + 1e-12))
    rows = []
    for allocation in _integer_allocations(len(assets), slots, max_slots):
        rows.append(
            {asset: weight_slots / slots for asset, weight_slots in zip(assets, allocation)}
        )
    return pd.DataFrame(rows, columns=assets, dtype=float)


def optimize_long_only_portfolio(
    returns: pd.DataFrame,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
    step: float = 0.05,
    max_weight: float = 1.0,
) -> dict[str, object]:
    """Approximate min-variance and max-Sharpe portfolios by long-only grid search."""

    clean_returns = _as_return_frame(returns)
    weights = long_only_weight_grid(
        tuple(str(column) for column in clean_returns.columns), step, max_weight
    )
    periodic_rf = annual_rate_to_periodic(risk_free_rate_annual, periods_per_year)
    rows = []
    for _, weight_row in weights.iterrows():
        port_returns = clean_returns.mul(weight_row, axis=1).sum(axis=1)
        ann_return = float(port_returns.mean() * periods_per_year)
        ann_vol = float(port_returns.std(ddof=1) * np.sqrt(periods_per_year))
        if ann_vol == 0 or not np.isfinite(ann_vol):
            sharpe = float("nan")
        else:
            sharpe = float((port_returns.mean() - periodic_rf) / port_returns.std(ddof=1))
            sharpe *= float(np.sqrt(periods_per_year))
        rows.append(
            {
                "weights": {
                    str(asset): float(weight_row[asset]) for asset in clean_returns.columns
                },
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
            }
        )
    frontier = sorted(
        rows, key=lambda row: (row["annualized_volatility"], row["annualized_return"])
    )
    min_variance = min(frontier, key=lambda row: row["annualized_volatility"])
    finite_sharpe = [row for row in frontier if np.isfinite(float(row["sharpe_ratio"]))]
    max_sharpe = (
        max(finite_sharpe, key=lambda row: row["sharpe_ratio"]) if finite_sharpe else frontier[0]
    )
    return {
        "method": "long_only_grid_search",
        "grid_step": float(step),
        "max_weight": float(max_weight),
        "portfolio_count": int(len(frontier)),
        "min_variance": min_variance,
        "max_sharpe": max_sharpe,
        "frontier": frontier,
        "limitations": [
            "Grid-search frontier is approximate and constrained to long-only weights.",
            "Expected returns and covariance are estimated from historical daily returns.",
        ],
    }


def _integer_allocations(asset_count: int, slots: int, max_slots: int) -> list[tuple[int, ...]]:
    if asset_count == 1:
        return [(slots,)] if slots <= max_slots else []
    allocations = []
    for first in range(min(max_slots, slots) + 1):
        for tail in _integer_allocations(asset_count - 1, slots - first, max_slots):
            allocations.append((first, *tail))
    return allocations


def _as_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise PortfolioOptimizationError("returns must be a non-empty DataFrame.")
    clean = returns.astype(float)
    if clean.columns.has_duplicates:
        raise PortfolioOptimizationError("returns columns must be unique.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean < -1).to_numpy().any():
        raise PortfolioOptimizationError("returns must be finite and above -100%.")
    return clean
