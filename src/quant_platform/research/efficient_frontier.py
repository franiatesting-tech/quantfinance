"""Efficient-frontier presentation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def capital_allocation_line(
    max_sharpe_portfolio: dict[str, Any],
    risk_free_rate_annual: float = 0.0,
    points: int = 25,
) -> list[dict[str, float]]:
    """Build Capital Allocation Line points from the max-Sharpe portfolio."""

    if points < 2:
        raise ValueError("points must be >= 2.")
    tangent_vol = float(max_sharpe_portfolio["annualized_volatility"])
    tangent_return = float(max_sharpe_portfolio["annualized_return"])
    if tangent_vol <= 0 or not np.isfinite(tangent_vol):
        return []
    slope = (tangent_return - risk_free_rate_annual) / tangent_vol
    vol_grid = np.linspace(0.0, tangent_vol * 1.5, points)
    return [
        {
            "annualized_volatility": float(vol),
            "annualized_return": float(risk_free_rate_annual + slope * vol),
        }
        for vol in vol_grid
    ]


def frontier_scatter_rows(optimization_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten optimizer frontier rows for charts and CSV export."""

    rows = []
    for row in optimization_result.get("frontier", []):
        weights = row.get("weights", {})
        flat = {
            "annualized_return": row.get("annualized_return"),
            "annualized_volatility": row.get("annualized_volatility"),
            "sharpe_ratio": row.get("sharpe_ratio"),
        }
        if isinstance(weights, dict):
            flat.update({f"weight_{key}": value for key, value in weights.items()})
        rows.append(flat)
    return rows
