from __future__ import annotations

import numpy as np
import pytest

from quant_platform.research.fixed_income import FixedIncomeError, bond_price, bond_summary


def test_zero_coupon_bond_price_and_duration() -> None:
    price = bond_price(100.0, 0.0, 0.05, 2.0, frequency=1)
    summary = bond_summary(100.0, 0.0, 0.05, 2.0, frequency=1)

    assert np.isclose(price, 100.0 / 1.05**2)
    assert np.isclose(summary["macaulay_duration_years"], 2.0)
    assert summary["dv01"] > 0


def test_bond_summary_rejects_zero_yield_bump() -> None:
    with pytest.raises(FixedIncomeError):
        bond_summary(100.0, 0.03, 0.04, 2.0, yield_bump=0.0)
