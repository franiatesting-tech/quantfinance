from __future__ import annotations

from quant_platform.research.state_of_art.model_confidence import (
    MODEL_CONFIDENCE_ACCEPTED,
    MODEL_CONFIDENCE_REJECTED,
    MODEL_CONFIDENCE_REQUIRES_MORE_TESTS,
    evaluate_model_confidence,
)


def test_model_confidence_accepts_model_that_beats_baseline() -> None:
    result = evaluate_model_confidence(
        oos_r_squared=0.05,
        rmse=0.90,
        baseline_rmse=1.00,
        information_coefficient=0.10,
        directional_accuracy=0.56,
        baseline_directional_accuracy=0.50,
    )

    assert result["model_confidence_status"] == MODEL_CONFIDENCE_ACCEPTED
    assert result["failed_checks"] == ""


def test_model_confidence_rejects_model_that_does_not_beat_baseline() -> None:
    result = evaluate_model_confidence(
        oos_r_squared=-0.01,
        rmse=1.01,
        baseline_rmse=1.00,
        information_coefficient=-0.10,
        directional_accuracy=0.49,
        baseline_directional_accuracy=0.50,
    )

    assert result["model_confidence_status"] == MODEL_CONFIDENCE_REJECTED


def test_model_confidence_requires_more_tests_when_metric_missing() -> None:
    result = evaluate_model_confidence(
        oos_r_squared=None,
        rmse=1.0,
        baseline_rmse=1.0,
        information_coefficient=0.1,
        directional_accuracy=0.55,
        baseline_directional_accuracy=0.50,
    )

    assert result["model_confidence_status"] == MODEL_CONFIDENCE_REQUIRES_MORE_TESTS
