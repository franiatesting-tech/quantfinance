"""Multiple-testing and Sharpe overfitting diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey-Lopez de Prado probabilistic Sharpe ratio approximation."""

    if observations < 2:
        raise ValueError("observations must be >= 2.")
    denominator = 1.0 - skewness * observed_sharpe
    denominator += ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    denominator = max(denominator, 1e-12)
    statistic = (observed_sharpe - benchmark_sharpe) * math.sqrt(observations - 1.0)
    statistic /= math.sqrt(denominator)
    return _normal_cdf(statistic)


def deflated_sharpe_ratio_approximation(
    observed_sharpe: float,
    *,
    observations: int,
    number_of_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, Any]:
    """Approximate DSR by testing Sharpe against an expected best-of-N haircut."""

    if number_of_trials < 1:
        raise ValueError("number_of_trials must be >= 1.")
    if observations < 2:
        raise ValueError("observations must be >= 2.")
    expected_best_noise_sharpe = math.sqrt(2.0 * math.log(max(number_of_trials, 1)))
    expected_best_noise_sharpe /= math.sqrt(observations - 1.0)
    psr = probabilistic_sharpe_ratio(
        observed_sharpe,
        benchmark_sharpe=expected_best_noise_sharpe,
        observations=observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return {
        "implementation_status": "PARTIAL_IMPLEMENTATION_DSR_APPROXIMATION",
        "observed_sharpe": float(observed_sharpe),
        "number_of_trials": int(number_of_trials),
        "observations": int(observations),
        "sharpe_haircut": float(expected_best_noise_sharpe),
        "deflated_sharpe": float(observed_sharpe - expected_best_noise_sharpe),
        "probability_sharpe_exceeds_haircut": psr,
        "selection_bias_warning": bool(number_of_trials > 1),
        "literature": "BaileyLopezPrado2014; White2000; Hansen2005",
    }


def multiple_testing_adjustment_table(
    rows: list[dict[str, Any]],
    *,
    number_of_trials: int | None = None,
) -> pd.DataFrame:
    """Create DSR/PSR diagnostics for metric rows containing Sharpe-like values."""

    records = []
    trials = number_of_trials or max(len(rows), 1)
    for row in rows:
        observed_sharpe = _safe_float(row.get("sharpe_ratio"))
        observations = int(row.get("observations") or 0)
        if observed_sharpe is None or observations < 2:
            continue
        result = deflated_sharpe_ratio_approximation(
            observed_sharpe,
            observations=observations,
            number_of_trials=trials,
            skewness=float(row.get("skewness") or 0.0),
            kurtosis=float(row.get("kurtosis") or 3.0),
        )
        result["asset_id"] = row.get("asset_id") or row.get("series")
        records.append(result)
    return pd.DataFrame(records)


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
