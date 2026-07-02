"""Kalman-filter state estimation for dynamic hedge ratios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class KalmanFilterError(ValueError):
    """Raised when Kalman inputs are invalid."""


def kalman_dynamic_regression(
    y: pd.Series,
    x: pd.Series,
    *,
    transition_covariance: float = 1e-5,
    observation_covariance: float | None = None,
) -> dict[str, Any]:
    """Estimate y_t = alpha_t + beta_t x_t + noise_t with a random-walk state."""

    frame = pd.concat([_as_series(y, "y"), _as_series(x, "x")], axis=1).dropna(how="any")
    frame.columns = ["y", "x"]
    if len(frame) < 30:
        raise KalmanFilterError("at least 30 paired observations are required.")
    if transition_covariance <= 0 or not np.isfinite(transition_covariance):
        raise KalmanFilterError("transition_covariance must be finite and > 0.")
    initial_design = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(dtype=float)])
    initial_state = np.linalg.lstsq(initial_design, frame["y"].to_numpy(dtype=float), rcond=None)[0]
    residuals = frame["y"].to_numpy(dtype=float) - initial_design @ initial_state
    obs_var = (
        float(np.var(residuals, ddof=2))
        if observation_covariance is None
        else observation_covariance
    )
    if obs_var <= 0 or not np.isfinite(obs_var):
        obs_var = 1e-6
    state = np.array(initial_state, dtype=float)
    covariance = np.eye(2) * 0.1
    q = np.eye(2) * float(transition_covariance)
    rows = []
    for idx, obs in frame.iterrows():
        covariance = covariance + q
        h = np.array([1.0, float(obs["x"])])
        predicted_y = float(h @ state)
        innovation = float(obs["y"] - predicted_y)
        innovation_var = float(h @ covariance @ h.T + obs_var)
        gain = (covariance @ h.T) / innovation_var
        state = state + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, h)) @ covariance
        rows.append(
            {
                "timestamp": pd.Timestamp(idx).isoformat(),
                "alpha": float(state[0]),
                "beta": float(state[1]),
                "predicted_y": predicted_y,
                "innovation": innovation,
                "innovation_std": float(np.sqrt(max(innovation_var, 0.0))),
            }
        )
    beta_values = np.array([row["beta"] for row in rows], dtype=float)
    innovation_values = np.array([row["innovation"] for row in rows], dtype=float)
    return {
        "model": "kalman_dynamic_linear_regression",
        "observations": int(len(rows)),
        "latest_alpha": float(rows[-1]["alpha"]),
        "latest_beta": float(rows[-1]["beta"]),
        "average_beta": float(np.mean(beta_values)),
        "beta_volatility": float(np.std(beta_values, ddof=1)) if len(beta_values) > 1 else 0.0,
        "innovation_std": float(np.std(innovation_values, ddof=1))
        if len(innovation_values) > 1
        else 0.0,
        "transition_covariance": float(transition_covariance),
        "observation_covariance": float(obs_var),
        "state_rows": rows[-252:],
        "model_status": "RESEARCH_ONLY_DYNAMIC_HEDGE_RATIO",
    }


def _as_series(value: pd.Series, name: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise KalmanFilterError(f"{name} must be a non-empty Series.")
    clean = value.astype(float).dropna()
    if clean.empty or not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise KalmanFilterError(f"{name} must contain finite observations.")
    return clean
