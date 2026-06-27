"""Monte Carlo simulation models for stock and portfolio analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MonteCarloError(ValueError):
    """Raised when Monte Carlo inputs are invalid."""


def simulate_bootstrap_returns(
    returns: pd.Series,
    horizon_days: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate simple-return paths by historical bootstrap."""

    clean = _as_return_series(returns)
    rng = np.random.default_rng(seed)
    return rng.choice(clean.to_numpy(dtype=float), size=(n_paths, horizon_days), replace=True)


def simulate_block_bootstrap_returns(
    returns: pd.Series,
    horizon_days: int,
    n_paths: int,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Simulate simple-return paths by block bootstrap."""

    clean = _as_return_series(returns)
    if block_size < 1:
        raise MonteCarloError("block_size must be >= 1.")
    values = clean.to_numpy(dtype=float)
    if len(values) < block_size:
        raise MonteCarloError("returns length must be at least block_size.")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(values) - block_size + 1)
    paths = np.empty((n_paths, horizon_days), dtype=float)
    for path_index in range(n_paths):
        collected = []
        while len(collected) < horizon_days:
            start = int(rng.choice(starts))
            collected.extend(values[start : start + block_size])
        paths[path_index, :] = collected[:horizon_days]
    return paths


def simulate_normal_returns(
    returns: pd.Series,
    horizon_days: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate normally distributed simple-return paths from historical moments."""

    clean = _as_return_series(returns)
    rng = np.random.default_rng(seed)
    return rng.normal(float(clean.mean()), float(clean.std(ddof=1)), size=(n_paths, horizon_days))


def simulate_gbm_prices(
    start_price: float,
    log_returns: pd.Series,
    horizon_days: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate GBM price paths from historical log-return drift and volatility."""

    if not np.isfinite(start_price) or start_price <= 0:
        raise MonteCarloError("start_price must be finite and > 0.")
    clean = _as_return_series(log_returns)
    rng = np.random.default_rng(seed)
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=1))
    shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=(n_paths, horizon_days))
    return start_price * np.exp(np.cumsum(shocks, axis=1))


def simulate_portfolio_normal_returns(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    horizon_days: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate correlated normal asset returns and collapse to portfolio returns."""

    clean = _as_return_frame(returns)
    clean_weights = _weights(weights, clean.columns)
    mean = clean.mean().to_numpy(dtype=float)
    covariance = _nearest_psd(clean.cov().to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, covariance, size=(n_paths, horizon_days))
    return np.einsum("pth,h->pt", draws, clean_weights.to_numpy(dtype=float))


def summarize_simulated_paths(paths: np.ndarray, start_value: float = 1.0) -> dict[str, object]:
    """Summarize simulated return paths as terminal wealth and percentile fan data."""

    clean = _as_path_array(paths)
    values = start_value * np.cumprod(1.0 + clean, axis=1)
    terminal = values[:, -1]
    percentiles = [5, 25, 50, 75, 95]
    fan = np.percentile(values, percentiles, axis=0)
    return {
        "path_count": int(clean.shape[0]),
        "horizon_days": int(clean.shape[1]),
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_p05": float(np.percentile(terminal, 5)),
        "terminal_p95": float(np.percentile(terminal, 95)),
        "fan_chart": [
            {
                "step": int(step + 1),
                **{
                    f"p{percentile}": float(fan[index, step])
                    for index, percentile in enumerate(percentiles)
                },
            }
            for step in range(clean.shape[1])
        ],
        "model_status": "PARAMETRIC_OR_BOOTSTRAP_SIMULATION",
    }


def _as_return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series) or returns.empty:
        raise MonteCarloError("returns must be a non-empty Series.")
    clean = returns.astype(float).dropna()
    if clean.empty or not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean < -1).any():
        raise MonteCarloError("returns must be finite and above -100%.")
    return clean


def _as_return_frame(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise MonteCarloError("returns must be a non-empty DataFrame.")
    clean = returns.astype(float).dropna(how="any")
    if (
        clean.empty
        or not np.isfinite(clean.to_numpy(dtype=float)).all()
        or (clean < -1).to_numpy().any()
    ):
        raise MonteCarloError("returns must be finite and above -100%.")
    return clean


def _weights(weights: dict[str, float] | pd.Series, columns: pd.Index) -> pd.Series:
    clean = (
        weights.astype(float) if isinstance(weights, pd.Series) else pd.Series(weights, dtype=float)
    )
    clean = clean.reindex(columns)
    if clean.isna().any() or (clean < 0).any():
        raise MonteCarloError("weights must align and be long-only.")
    total = float(clean.sum())
    if total <= 0:
        raise MonteCarloError("weights must sum to a positive value.")
    return clean / total


def _as_path_array(paths: np.ndarray) -> np.ndarray:
    clean = np.asarray(paths, dtype=float)
    if clean.ndim != 2 or clean.size == 0:
        raise MonteCarloError("paths must be a non-empty 2D array.")
    if not np.isfinite(clean).all() or (clean < -1).any():
        raise MonteCarloError("paths must be finite simple returns above -100%.")
    return clean


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 1e-12, None)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T
