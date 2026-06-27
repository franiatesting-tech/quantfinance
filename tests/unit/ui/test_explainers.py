from __future__ import annotations

from quant_platform.ui.explainers import all_explainers


def test_all_explainers_have_required_fields() -> None:
    required = {
        "title",
        "summary",
        "why_it_matters",
        "analogy",
        "technical_note",
        "risk_warning",
    }

    explainers = all_explainers()

    assert len(explainers) == 14
    for explainer in explainers:
        assert required <= set(explainer)
        assert all(explainer[field] for field in required)


def test_explainers_cover_read_only_and_backtest_limits() -> None:
    text = " ".join(str(explainer).lower() for explainer in all_explainers())

    assert "read-only" in text
    assert "no es una prediccion" in text
    assert "no es asesoramiento financiero" in text
