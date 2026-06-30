"""Factor-model scaffolding that fails closed when factor returns are unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FACTOR_MODEL_SPECS: dict[str, tuple[str, ...]] = {
    "capm_market": ("MKT",),
    "fama_french_3": ("MKT", "SMB", "HML"),
    "carhart_4": ("MKT", "SMB", "HML", "MOM"),
    "fama_french_5": ("MKT", "SMB", "HML", "RMW", "CMA"),
}


def factor_model_gap_table(asset_ids: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Return a fail-closed table describing required factor data per asset/model."""

    rows = []
    for asset_id in asset_ids:
        for model, factors in FACTOR_MODEL_SPECS.items():
            rows.append(
                {
                    "asset_id": str(asset_id),
                    "factor_model": model,
                    "required_factors": "; ".join(factors),
                    "status": "FACTOR_DATA_REQUIRED",
                    "interpretation": (
                        "Do not claim factor-adjusted alpha until point-in-time factor "
                        "returns are loaded and aligned."
                    ),
                    "literature": "FamaFrench1993; Carhart1997; FamaFrench2015",
                }
            )
    return pd.DataFrame(rows)


def load_factor_returns_csv(path: str | Path) -> pd.DataFrame:
    """Load future external factor returns from a CSV with a date/timestamp column."""

    frame = pd.read_csv(path)
    date_column = "timestamp" if "timestamp" in frame.columns else "date"
    if date_column not in frame.columns:
        raise ValueError("factor CSV must contain 'timestamp' or 'date'.")
    frame[date_column] = pd.to_datetime(frame[date_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    return frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")


def fit_factor_model(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame | None,
    *,
    model: str = "fama_french_3",
    risk_free_rate_daily: float = 0.0,
) -> dict[str, Any]:
    """Fit an OLS factor model if required factor data are available."""

    if model not in FACTOR_MODEL_SPECS:
        raise ValueError(f"Unknown factor model: {model}")
    required = FACTOR_MODEL_SPECS[model]
    if factor_returns is None or factor_returns.empty:
        return _factor_data_required(model, required)
    missing = [factor for factor in required if factor not in factor_returns.columns]
    if missing:
        result = _factor_data_required(model, required)
        result["missing_factors"] = "; ".join(missing)
        return result
    aligned = pd.concat(
        [asset_returns.astype(float).rename("asset"), factor_returns.loc[:, required]],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(aligned) <= len(required) + 2:
        return {
            "factor_model": model,
            "status": "INSUFFICIENT_FACTOR_OBSERVATIONS",
            "observations": int(len(aligned)),
        }
    y = aligned["asset"].to_numpy(dtype=float) - risk_free_rate_daily
    x = aligned.loc[:, required].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(aligned)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    total_var = float(np.sum((y - y.mean()) ** 2))
    residual_var = float(np.sum(residual**2))
    r_squared = 1.0 - residual_var / total_var if total_var > 0 else None
    return {
        "factor_model": model,
        "status": "FACTOR_MODEL_ESTIMATED",
        "observations": int(len(aligned)),
        "daily_alpha": float(coefficients[0]),
        "annualized_alpha": float(coefficients[0] * 252.0),
        "r_squared": r_squared,
        "betas": {factor: float(value) for factor, value in zip(required, coefficients[1:])},
        "literature": "FamaFrench1993; Carhart1997; FamaFrench2015",
    }


def _factor_data_required(model: str, required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "factor_model": model,
        "status": "FACTOR_DATA_REQUIRED",
        "required_factors": "; ".join(required),
        "annualized_alpha": None,
        "r_squared": None,
        "literature": "FamaFrench1993; Carhart1997; FamaFrench2015",
    }
