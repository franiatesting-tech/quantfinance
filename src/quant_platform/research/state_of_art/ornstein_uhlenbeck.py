"""Ornstein-Uhlenbeck pairs-trading research diagnostics."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


class OrnsteinUhlenbeckError(ValueError):
    """Raised when OU diagnostics cannot be computed safely."""


def estimate_ou_from_spread(
    spread: pd.Series,
    *,
    periods_per_year: int = 252,
) -> dict[str, float | str | None]:
    """Estimate discrete OU parameters from a stationary spread proxy.

    The discrete regression is x_t = c + phi x_{t-1} + eps_t. For 0 < phi < 1,
    the continuous-time OU speed is kappa = -log(phi) per day.
    """

    clean = _as_series(spread, "spread")
    if len(clean) < 60:
        raise OrnsteinUhlenbeckError("spread must contain at least 60 observations.")
    y = clean.iloc[1:].to_numpy(dtype=float)
    x = clean.iloc[:-1].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - (intercept + phi * x)
    residual_std = float(np.std(residuals, ddof=2))
    theta = float(intercept / (1.0 - phi)) if not np.isclose(1.0 - phi, 0.0) else None
    if 0.0 < phi < 1.0:
        kappa = float(-math.log(phi))
        half_life = float(math.log(2.0) / kappa)
        sigma = float(residual_std * math.sqrt(2.0 * kappa / (1.0 - phi**2)))
    else:
        kappa = None
        half_life = None
        sigma = None
    return {
        "model": "discrete_ar1_ou_proxy",
        "intercept": float(intercept),
        "phi": float(phi),
        "theta": theta,
        "kappa_daily": kappa,
        "half_life_days": half_life,
        "sigma_daily": sigma,
        "residual_std_daily": residual_std,
        "annualized_spread_volatility": residual_std * math.sqrt(periods_per_year),
        "mean_reverting": bool(0.0 < phi < 1.0),
        "model_status": "OU_AR1_RESEARCH_ESTIMATE",
    }


def screen_ou_pairs(
    prices: pd.DataFrame,
    *,
    max_pairs: int = 20,
    periods_per_year: int = 252,
) -> list[dict[str, Any]]:
    """Screen all price pairs by OLS hedge ratio and OU mean-reversion diagnostics."""

    clean = _as_price_frame(prices)
    rows: list[dict[str, Any]] = []
    for left, right in combinations(clean.columns, 2):
        pair = np.log(clean[[left, right]]).dropna(how="any")
        if len(pair) < 252:
            continue
        y = pair[left].to_numpy(dtype=float)
        x = pair[right].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        spread = pd.Series(y - (alpha + beta * x), index=pair.index)
        ou = estimate_ou_from_spread(spread, periods_per_year=periods_per_year)
        corr = float(pair[left].corr(pair[right]))
        half_life = ou["half_life_days"]
        score = _pair_score(corr, half_life, bool(ou["mean_reverting"]))
        rows.append(
            {
                "asset_y": str(left),
                "asset_x": str(right),
                "hedge_intercept": float(alpha),
                "hedge_beta": float(beta),
                "log_price_correlation": corr,
                "spread_mean": float(spread.mean()),
                "spread_std": float(spread.std(ddof=1)),
                "ou_phi": float(ou["phi"]),
                "ou_half_life_days": half_life,
                "ou_mean_reverting": bool(ou["mean_reverting"]),
                "selection_score": score,
            }
        )
    return sorted(rows, key=lambda row: float(row["selection_score"]), reverse=True)[:max_pairs]


def pairs_zscore_backtest(
    prices: pd.DataFrame,
    pair: dict[str, Any],
    *,
    lookback: int = 63,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Backtest an educational market-neutral OU z-score rule with one-day lag."""

    clean = _as_price_frame(prices)
    y_name = str(pair["asset_y"])
    x_name = str(pair["asset_x"])
    if y_name not in clean or x_name not in clean:
        raise OrnsteinUhlenbeckError("pair assets must exist in prices.")
    beta = float(pair["hedge_beta"])
    alpha = float(pair.get("hedge_intercept", 0.0))
    logs = np.log(clean[[y_name, x_name]]).dropna(how="any")
    spread = logs[y_name] - (alpha + beta * logs[x_name])
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std(ddof=1)
    zscore = (spread - rolling_mean) / rolling_std
    position = pd.Series(0.0, index=zscore.index)
    current = 0.0
    for idx, z in zscore.dropna().items():
        if current == 0.0:
            if z > entry_z:
                current = -1.0
            elif z < -entry_z:
                current = 1.0
        elif abs(z) < exit_z:
            current = 0.0
        position.loc[idx] = current
    simple_returns = clean[[y_name, x_name]].pct_change().reindex(position.index)
    spread_returns = simple_returns[y_name] - beta * simple_returns[x_name]
    strategy_returns = position.shift(1).fillna(0.0) * spread_returns
    strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
    trades = int((position.diff().abs() > 0).sum())
    ann_return = (
        float(strategy_returns.mean() * periods_per_year) if not strategy_returns.empty else 0.0
    )
    ann_vol = (
        float(strategy_returns.std(ddof=1) * math.sqrt(periods_per_year))
        if len(strategy_returns) > 1
        else 0.0
    )
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    equity = (1.0 + strategy_returns).cumprod()
    max_dd = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else 0.0
    rows = [
        {
            "timestamp": pd.Timestamp(idx).isoformat(),
            "spread": float(spread.loc[idx]),
            "zscore": float(zscore.loc[idx]) if np.isfinite(zscore.loc[idx]) else None,
            "position": float(position.loc[idx]),
        }
        for idx in zscore.dropna().index[-252:]
    ]
    return {
        "asset_y": y_name,
        "asset_x": x_name,
        "hedge_beta": beta,
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "lookback": int(lookback),
        "observations": int(len(strategy_returns)),
        "trades": trades,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "latest_zscore": float(zscore.dropna().iloc[-1]) if not zscore.dropna().empty else None,
        "series": rows,
        "model_status": "RESEARCH_ONLY_BACKTEST_WITH_ONE_DAY_LAG",
    }


def _as_series(value: pd.Series, name: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise OrnsteinUhlenbeckError(f"{name} must be a non-empty Series.")
    clean = value.astype(float).dropna()
    if clean.empty or not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise OrnsteinUhlenbeckError(f"{name} must contain finite observations.")
    return clean


def _as_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty or len(prices.columns) < 2:
        raise OrnsteinUhlenbeckError("prices must be a non-empty DataFrame with >=2 columns.")
    clean = prices.astype(float).dropna(how="any")
    if clean.empty or (clean <= 0).to_numpy().any() or not np.isfinite(clean.to_numpy()).all():
        raise OrnsteinUhlenbeckError("prices must be positive and finite.")
    return clean


def _pair_score(correlation: float, half_life: object, mean_reverting: bool) -> float:
    if not mean_reverting or half_life is None or not np.isfinite(float(half_life)):
        return -1.0
    hl = max(float(half_life), 1.0)
    return float(abs(correlation) / (1.0 + abs(hl - 20.0) / 20.0))
