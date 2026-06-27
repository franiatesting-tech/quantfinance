from __future__ import annotations

from quant_platform.reporting.academic_conclusions import build_stock_conclusions


def test_academic_conclusions_are_deterministic_and_non_recommendatory() -> None:
    stock = {
        "metrics": {
            "annualized_return": 0.12,
            "final_cumulative_return": 0.5,
            "annualized_volatility": 0.2,
            "max_drawdown": -0.25,
            "sharpe_ratio": 0.8,
            "sortino_ratio": 1.1,
            "beta_to_benchmark": 1.05,
            "jensen_alpha": 0.02,
            "treynor_ratio": 0.1,
            "skewness": -0.2,
            "kurtosis": 3.0,
        },
        "var": {
            "historical": {"var": 0.03, "expected_shortfall": 0.05},
            "parametric_normal": {"var": 0.04, "expected_shortfall": 0.06},
            "monte_carlo": {"var": 0.05, "expected_shortfall": 0.07},
        },
        "monte_carlo": {
            "parametric_normal": {
                "terminal_p05": -0.1,
                "terminal_median": 0.05,
                "terminal_p95": 0.2,
                "probability_of_loss": 0.35,
            }
        },
        "backtesting_results": {
            "buy_and_hold": {"metrics": {"final_equity": 12000, "max_drawdown": -0.25}}
        },
        "options_theoretical_analytics": {
            "black_scholes_call": 10,
            "black_scholes_put": 7,
            "volatility": 0.2,
        },
    }

    conclusions = build_stock_conclusions(stock)
    joined = " ".join(conclusions.values()).lower()

    assert "Historical performance" in conclusions
    assert "bajo los supuestos" in joined
    assert "comprar" not in joined
    assert "vender" not in joined
    assert "recomendamos" not in joined
    assert "garantiza" not in joined
