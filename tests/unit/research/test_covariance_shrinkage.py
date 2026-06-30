from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.research.state_of_art.covariance_shrinkage import (
    covariance_shrinkage_comparison,
    diagonal_shrinkage_covariance,
    ledoit_wolf_or_diagonal_covariance,
)


def test_diagonal_shrinkage_removes_off_diagonal_at_delta_one() -> None:
    covariance = pd.DataFrame([[4.0, 2.0], [2.0, 9.0]], columns=["A", "B"], index=["A", "B"])

    shrunk = diagonal_shrinkage_covariance(covariance, delta=1.0, input_is_covariance=True)

    assert shrunk.loc["A", "A"] == pytest.approx(4.0)
    assert shrunk.loc["B", "B"] == pytest.approx(9.0)
    assert shrunk.loc["A", "B"] == pytest.approx(0.0)


def test_ledoit_wolf_or_diagonal_covariance_returns_psd_square_matrix() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.00, 0.01],
            "B": [0.00, 0.01, -0.02, 0.01, 0.02],
        }
    )

    covariance, metadata = ledoit_wolf_or_diagonal_covariance(returns)

    assert covariance.shape == (2, 2)
    assert metadata["method"] in {"sklearn_ledoit_wolf", "fallback_diagonal_shrinkage"}
    assert np.linalg.eigvalsh(covariance.to_numpy()).min() >= -1e-12


def test_covariance_shrinkage_comparison_contains_two_estimators_and_objectives() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.00, 0.01, 0.03],
            "B": [0.00, 0.01, -0.02, 0.01, 0.02, 0.01],
            "C": [0.01, -0.01, 0.00, 0.01, -0.02, 0.02],
        }
    )

    table = covariance_shrinkage_comparison(returns, grid_step=0.5)

    assert set(table["covariance_estimator"]) == {"sample_covariance", "shrinkage_covariance"}
    assert set(table["optimizer_objective"]) == {"min_variance", "max_sharpe"}
    assert (table["effective_number_of_holdings"] > 0).all()
