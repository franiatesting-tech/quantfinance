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
            "status": "MODEL_EDGE_PASSED_STRICT_DIAGNOSTIC_GATES",
            "directional_accuracy": 0.56,
            "baseline_directional_accuracy": 0.50,
            "directional_accuracy_edge_vs_naive": 0.06,
            "oos_r_squared": 0.03,
            "information_coefficient": 0.04,
            "rmse": 0.010,
            "baseline_rmse": 0.012,
            "approval_gates": {
                "rmse_improves_naive": True,
                "mae_improves_naive": True,
                "directional_accuracy_edge_ge_2pct": True,
                "directional_accuracy_ge_52pct": True,
                "oos_r_squared_positive": True,
                "information_coefficient_positive": True,
                "strategy_sharpe_positive": True,
            },
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


def test_research_signal_caps_historical_profile_when_predictive_edge_fails() -> None:
    result = build_research_decision_signal(
        metrics={
            "observations": 1_500,
            "sharpe_ratio": 1.4,
            "sortino_ratio": 1.7,
            "max_drawdown": -0.2,
            "cagr": 0.18,
            "annualized_volatility": 0.2,
            "beta_to_benchmark": 1.0,
            "jensen_alpha": 0.05,
            "treynor_ratio": 0.1,
        },
        var_results={"historical": {"var": 0.02, "expected_shortfall": 0.03}},
        monte_carlo={"parametric_normal": {"terminal_median": 0.08, "probability_of_loss": 0.2}},
        ml_forecasting={
            "status": "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE",
            "directional_accuracy": 0.56,
            "baseline_directional_accuracy": 0.50,
            "directional_accuracy_edge_vs_naive": 0.06,
            "oos_r_squared": -0.01,
            "information_coefficient": -0.02,
            "rmse": 0.010,
            "baseline_rmse": 0.012,
        },
        backtesting={"buy_and_hold": {"metrics": {"sharpe_ratio": 1.0, "final_equity": 20_000}}},
    )

    assert result["signal"] == "NEUTRAL"
    assert "PREDICTIVE_EDGE_NOT_VALIDATED" in result["limits_triggered"]
