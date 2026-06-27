from __future__ import annotations

import pytest

from quant_platform.reporting.academic_figures import (
    FIGURE_FILENAMES,
    build_academic_stock_figures,
    write_academic_stock_figures,
)

pytest.importorskip("plotly")


def test_academic_figures_are_built_and_written(tmp_path) -> None:  # noqa: ANN001
    model = {"asset_id": "AAPL", "stock": _sample_stock()}

    figures = build_academic_stock_figures(model)
    paths = write_academic_stock_figures(model, tmp_path / "figures")

    assert set(figures) == set(FIGURE_FILENAMES)
    assert set(paths) == set(FIGURE_FILENAMES)
    assert all((tmp_path / "figures" / filename).exists() for filename in FIGURE_FILENAMES.values())
    assert all(figure.layout.title.text for figure in figures.values())


def _sample_stock() -> dict[str, object]:
    rows = [
        {
            "timestamp": "2024-01-01",
            "price": 100,
            "volume": 10,
            "return": 0.01,
            "log_return": 0.0099,
        },
        {
            "timestamp": "2024-01-02",
            "price": 101,
            "volume": 12,
            "return": -0.02,
            "log_return": -0.0202,
        },
    ]
    return {
        "asset_id": "AAPL",
        "data_used": {"provider": "synthetic", "frequency": "1d", "data_mode": "offline_synthetic"},
        "price_series": [{"timestamp": row["timestamp"], "price": row["price"]} for row in rows],
        "volume_series": [{"timestamp": row["timestamp"], "volume": row["volume"]} for row in rows],
        "simple_returns": [
            {"timestamp": row["timestamp"], "return": row["return"]} for row in rows
        ],
        "log_returns": [
            {"timestamp": row["timestamp"], "log_return": row["log_return"]} for row in rows
        ],
        "cumulative_returns": [{"timestamp": "2024-01-02", "cumulative_return": 0.02}],
        "drawdown_series": [{"timestamp": "2024-01-02", "drawdown": -0.02}],
        "rolling_volatility": [{"timestamp": "2024-01-02", "rolling_volatility": 0.2}],
        "rolling_sharpe": [{"timestamp": "2024-01-02", "rolling_sharpe": 0.5}],
        "rolling_beta": [{"timestamp": "2024-01-02", "rolling_beta": 1.0}],
        "var": {
            "historical": {"var": 0.02, "expected_shortfall": 0.03},
            "parametric_normal": {"var": 0.025, "expected_shortfall": 0.035},
            "monte_carlo": {"var": 0.03, "expected_shortfall": 0.04},
        },
        "monte_carlo": {
            "parametric_normal": {
                "paths_sample": [
                    {"path_id": 0, "step": 1, "value": 1.0},
                    {"path_id": 0, "step": 2, "value": 1.1},
                ],
                "fan_chart": [
                    {"step": 1, "p5": 0.9, "p25": 0.95, "p50": 1.0, "p75": 1.05, "p95": 1.1}
                ],
                "terminal_distribution": [-0.1, 0.05, 0.2],
            }
        },
        "backtesting_results": {
            "buy_and_hold": {"equity_curve": [{"timestamp": "2024-01-01", "equity": 10000}]}
        },
        "options_theoretical_analytics": {
            "payoff_profile": [
                {"underlying_price": 100, "payoff": 0},
                {"underlying_price": 110, "payoff": 10},
            ],
            "call_greeks": {"delta": 0.5, "gamma": 0.01, "vega": 20, "theta_annual": -5, "rho": 10},
        },
    }
