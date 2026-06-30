"""Execution-cost sensitivity diagnostics for research-only portfolios."""

from __future__ import annotations

import math

import pandas as pd


def turnover_between_weights(left: pd.Series, right: pd.Series) -> float:
    """One-way portfolio turnover between two long-only weight vectors."""

    assets = sorted(set(left.index.astype(str)).union(set(right.index.astype(str))))
    lvec = left.reindex(assets).fillna(0.0).astype(float)
    rvec = right.reindex(assets).fillna(0.0).astype(float)
    return float(0.5 * (lvec - rvec).abs().sum())


def turnover_cost(notional: float, turnover: float, *, cost_bps: float) -> float:
    """Linear transaction cost from notional, one-way turnover and cost in bps."""

    if notional < 0 or turnover < 0 or cost_bps < 0:
        raise ValueError("notional, turnover and cost_bps must be non-negative.")
    return float(notional * turnover * cost_bps / 10_000.0)


def execution_cost_sensitivity(
    old_weights: pd.Series,
    new_weights: pd.Series,
    *,
    notional: float = 1_000_000.0,
    spread_bps: float = 5.0,
    impact_coefficient: float = 0.10,
    adv_by_asset: dict[str, float] | None = None,
    volatility_by_asset: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate spread and square-root market-impact costs by asset."""

    assets = sorted(set(old_weights.index.astype(str)).union(set(new_weights.index.astype(str))))
    old = old_weights.reindex(assets).fillna(0.0).astype(float)
    new = new_weights.reindex(assets).fillna(0.0).astype(float)
    rows = []
    total_trade = 0.0
    total_cost = 0.0
    for asset in assets:
        trade_notional = abs(float(new[asset] - old[asset])) * notional
        total_trade += trade_notional
        adv = float((adv_by_asset or {}).get(asset, 0.0) or 0.0)
        volatility = float((volatility_by_asset or {}).get(asset, 0.25) or 0.25)
        spread_cost = trade_notional * spread_bps / 10_000.0
        participation = trade_notional / adv if adv > 0 else None
        impact_cost = 0.0
        if participation is not None:
            impact_cost = trade_notional * impact_coefficient * volatility
            impact_cost *= math.sqrt(max(participation, 0.0))
        total_asset_cost = spread_cost + impact_cost
        total_cost += total_asset_cost
        rows.append(
            {
                "asset_id": asset,
                "old_weight": float(old[asset]),
                "new_weight": float(new[asset]),
                "trade_notional": trade_notional,
                "spread_bps": spread_bps,
                "spread_cost": spread_cost,
                "adv_notional": adv if adv > 0 else None,
                "participation_rate": participation,
                "impact_cost": impact_cost,
                "total_estimated_cost": total_asset_cost,
                "capacity_warning": _capacity_warning(participation),
                "literature": "AlmgrenChriss2001; BertsimasLo1998; Gatheral2010",
            }
        )
    rows.append(
        {
            "asset_id": "PORTFOLIO_TOTAL",
            "old_weight": None,
            "new_weight": None,
            "trade_notional": total_trade,
            "spread_bps": spread_bps,
            "spread_cost": sum(float(row["spread_cost"]) for row in rows),
            "adv_notional": None,
            "participation_rate": None,
            "impact_cost": sum(float(row["impact_cost"]) for row in rows),
            "total_estimated_cost": total_cost,
            "capacity_warning": "RESEARCH_PROXY_NOT_BROKER_CALIBRATED",
            "literature": "AlmgrenChriss2001; BertsimasLo1998; Gatheral2010",
        }
    )
    return pd.DataFrame(rows)


def _capacity_warning(participation: float | None) -> str:
    if participation is None:
        return "ADV_DATA_REQUIRED"
    if participation > 0.10:
        return "HIGH_PARTICIPATION_CAPACITY_REVIEW"
    if participation > 0.02:
        return "MODERATE_PARTICIPATION_REVIEW"
    return "LOW_PARTICIPATION_PROXY"
