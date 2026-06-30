"""Model confidence diagnostics for financial ML outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

MODEL_CONFIDENCE_ACCEPTED = "MODEL_CONFIDENCE_ACCEPTED"
MODEL_CONFIDENCE_WEAK = "MODEL_CONFIDENCE_WEAK"
MODEL_CONFIDENCE_REJECTED = "MODEL_CONFIDENCE_REJECTED"
MODEL_CONFIDENCE_REQUIRES_MORE_TESTS = "MODEL_CONFIDENCE_REQUIRES_MORE_TESTS"


def evaluate_model_confidence(
    *,
    oos_r_squared: float | None,
    rmse: float | None,
    baseline_rmse: float | None,
    information_coefficient: float | None,
    directional_accuracy: float | None,
    baseline_directional_accuracy: float | None,
    min_directional_edge: float = 0.02,
) -> dict[str, Any]:
    """Evaluate whether an ML forecast beats a naive baseline robustly enough."""

    inputs = [
        oos_r_squared,
        rmse,
        baseline_rmse,
        information_coefficient,
        directional_accuracy,
        baseline_directional_accuracy,
    ]
    if any(value is None for value in inputs):
        return {
            "model_confidence_status": MODEL_CONFIDENCE_REQUIRES_MORE_TESTS,
            "passed_checks": 0,
            "failed_checks": "missing_required_metric",
        }
    checks = {
        "oos_r_squared_positive": float(oos_r_squared) > 0.0,
        "rmse_improves_naive": float(rmse) < float(baseline_rmse),
        "information_coefficient_positive": float(information_coefficient) > 0.0,
        "directional_accuracy_ge_52pct": float(directional_accuracy) >= 0.52,
        "directional_accuracy_edge_ge_min": (
            float(directional_accuracy) - float(baseline_directional_accuracy)
        )
        >= min_directional_edge,
    }
    passed = [name for name, value in checks.items() if value]
    failed = [name for name, value in checks.items() if not value]
    if len(passed) == len(checks):
        status = MODEL_CONFIDENCE_ACCEPTED
    elif checks["rmse_improves_naive"] and checks["information_coefficient_positive"]:
        status = MODEL_CONFIDENCE_WEAK
    else:
        status = MODEL_CONFIDENCE_REJECTED
    return {
        "model_confidence_status": status,
        "passed_checks": len(passed),
        "failed_checks": "; ".join(failed),
        "oos_r_squared_positive": checks["oos_r_squared_positive"],
        "rmse_improves_naive": checks["rmse_improves_naive"],
        "information_coefficient_positive": checks["information_coefficient_positive"],
        "directional_accuracy_edge": float(directional_accuracy)
        - float(baseline_directional_accuracy),
        "literature": "GuKellyXiu2020; LopezDePrado2018; HansenLundeNason2011",
    }


def model_confidence_table(ml_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create confidence rows from institutional ML audit rows."""

    rows = []
    for row in ml_rows:
        result = evaluate_model_confidence(
            oos_r_squared=_maybe_float(row.get("oos_r_squared")),
            rmse=_maybe_float(row.get("rmse")),
            baseline_rmse=_maybe_float(row.get("baseline_rmse")),
            information_coefficient=_maybe_float(row.get("information_coefficient")),
            directional_accuracy=_maybe_float(row.get("directional_accuracy")),
            baseline_directional_accuracy=_maybe_float(
                row.get("baseline_directional_accuracy")
            ),
        )
        result["asset_id"] = row.get("asset_id")
        result["strict_edge_validated"] = bool(row.get("strict_edge_validated"))
        rows.append(result)
    return pd.DataFrame(rows)


def _maybe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
