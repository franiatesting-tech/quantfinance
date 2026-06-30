from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.research.state_of_art.factor_models import (
    factor_model_gap_table,
    fit_factor_model,
)


def test_factor_model_gap_table_fails_closed_without_factor_data() -> None:
    table = factor_model_gap_table(["A", "B"])

    assert len(table) == 8
    assert set(table["status"]) == {"FACTOR_DATA_REQUIRED"}


def test_fit_factor_model_reports_factor_data_required_without_factors() -> None:
    result = fit_factor_model(pd.Series([0.01, 0.02]), None)

    assert result["status"] == "FACTOR_DATA_REQUIRED"


def test_fit_factor_model_estimates_capm_alpha_and_beta() -> None:
    index = pd.date_range("2020-01-01", periods=6, tz="UTC")
    factors = pd.DataFrame({"MKT": [0.01, 0.02, -0.01, 0.00, 0.01, 0.03]}, index=index)
    asset = pd.Series(0.001 + 2.0 * factors["MKT"], index=index)

    result = fit_factor_model(asset, factors, model="capm_market")

    assert result["status"] == "FACTOR_MODEL_ESTIMATED"
    assert result["daily_alpha"] == pytest.approx(0.001)
    assert result["betas"]["MKT"] == pytest.approx(2.0)
