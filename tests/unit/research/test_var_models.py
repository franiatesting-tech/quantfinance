from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from quant_platform.research.var_models import (
    compute_var_comparison_table,
    parametric_normal_var_es,
)


def test_parametric_var_uses_positive_loss_convention_without_clamping() -> None:
    returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    alpha = 0.95
    z = NormalDist().inv_cdf(alpha)
    expected = -returns.mean() + returns.std(ddof=1) * z

    result = parametric_normal_var_es(returns, alpha)

    assert np.isclose(result["var"], expected)


def test_var_comparison_table_returns_95_and_99_rows() -> None:
    returns = pd.Series([-0.03, -0.01, 0.0, 0.02, 0.04, -0.02])

    rows = compute_var_comparison_table(returns, simulated_returns=returns)

    assert {row["alpha"] for row in rows} == {0.95, 0.99}
    assert {"historical", "parametric_normal", "monte_carlo"}.issubset(
        {row["method"] for row in rows}
    )
    assert all(row["loss_sign_convention"] == "L_t = -R_t" for row in rows)
