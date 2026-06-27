from __future__ import annotations

from quant_platform.ui.formulas import all_formulas


def test_all_formulas_have_required_fields() -> None:
    required = {
        "name",
        "latex",
        "plain_explanation",
        "intuition",
        "inputs",
        "outputs",
        "used_in",
        "common_misinterpretation",
        "example_short",
    }

    formulas = all_formulas()

    assert len(formulas) == 12
    for formula in formulas:
        assert required <= set(formula)
        assert all(formula[field] for field in required)
        assert isinstance(formula["inputs"], list)


def test_formulas_do_not_contain_financial_recommendations() -> None:
    blocked = ("buy now", "sell now", "recommend buying", "financial advice")
    text = " ".join(str(formula).lower() for formula in all_formulas())

    assert not any(term in text for term in blocked)
