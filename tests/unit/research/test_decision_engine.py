from __future__ import annotations

from quant_platform.research.decision_engine import build_research_decision_signal


def test_research_signal_is_favorable_without_advisory_language() -> None:
    result = build_research_decision_signal(
        metrics={
            "observations": 1_000,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.4,
            "max_drawdown": -0.2,
            "cagr": 0.12,
            "annualized_volatility": 0.18,
            "beta_to_benchmark": 1.0,
            "jensen_alpha": 0.02,
            "treynor_ratio": 0.1,
        },
        var_results={"historical": {"var": 0.03, "expected_shortfall": 0.05}},
        monte_carlo={"parametric_normal": {"terminal_median": 0.06, "probability_of_loss": 0.3}},
        ml_forecasting={
            "status": "MODEL_OUTPERFORMS_NAIVE_UNDER_TEST_ASSUMPTIONS",
            "directional_accuracy": 0.55,
            "baseline_directional_accuracy": 0.50,
        },
        backtesting={"buy_and_hold": {"metrics": {"sharpe_ratio": 0.8, "final_equity": 12_000}}},
    )

    joined = " ".join(str(value) for value in result.values()).lower()
    assert result["signal"] == "FAVORABLE"
    assert result["not_investment_advice"] is True
    assert "buy" not in joined
    assert "sell" not in joined


def test_research_signal_requires_minimum_observations() -> None:
    result = build_research_decision_signal(metrics={"observations": 100})

    assert result["signal"] == "INSUFFICIENT_DATA"
    assert "MIN_OBSERVATIONS_NOT_MET" in result["limits_triggered"]
