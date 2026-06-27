"""Parametric counterparty exposure profile analytics."""

from __future__ import annotations

import numpy as np


class ExposureError(ValueError):
    """Raised when exposure inputs are invalid."""


def simulate_exposure_paths(
    n_paths: int,
    n_steps: int,
    initial_exposure: float = 0.0,
    drift: float = 0.0,
    volatility: float = 100_000.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate educational mark-to-market exposure paths by arithmetic random walk."""

    if n_paths < 1 or n_steps < 1:
        raise ExposureError("n_paths and n_steps must be >= 1.")
    values = (initial_exposure, drift, volatility)
    if any(not np.isfinite(value) for value in values) or volatility < 0:
        raise ExposureError("exposure parameters must be finite and volatility >= 0.")
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, volatility, size=(n_paths, n_steps))
    return initial_exposure + np.cumsum(shocks, axis=1)


def exposure_profile(paths: np.ndarray, confidence: float = 0.95) -> dict[str, object]:
    """Compute EE, EPE, ENE, and PFE from simulated exposure paths."""

    if not 0 < confidence < 1:
        raise ExposureError("confidence must be in (0, 1).")
    clean = np.asarray(paths, dtype=float)
    if clean.ndim != 2 or clean.size == 0 or not np.isfinite(clean).all():
        raise ExposureError("paths must be a finite non-empty 2D array.")
    positive = np.maximum(clean, 0.0)
    negative = np.maximum(-clean, 0.0)
    expected_exposure = positive.mean(axis=0)
    expected_negative_exposure = negative.mean(axis=0)
    pfe = np.quantile(positive, confidence, axis=0)
    return {
        "confidence": float(confidence),
        "path_count": int(clean.shape[0]),
        "step_count": int(clean.shape[1]),
        "epe": float(expected_exposure.mean()),
        "ene": float(expected_negative_exposure.mean()),
        "max_pfe": float(pfe.max()),
        "profile": [
            {
                "step": int(index + 1),
                "expected_exposure": float(expected_exposure[index]),
                "expected_negative_exposure": float(expected_negative_exposure[index]),
                "pfe": float(pfe[index]),
            }
            for index in range(clean.shape[1])
        ],
        "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
        "real_market_status": "DATA_REQUIRED_FOR_REAL_MARKET_VALUATION",
    }
