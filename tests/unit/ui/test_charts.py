from __future__ import annotations

import pytest

from quant_platform.ui import charts

pytest.importorskip("plotly")


def test_provider_status_figure_counts_statuses() -> None:
    figure = charts.provider_status_figure(
        [
            {"status": "available"},
            {"status": "configured_but_disabled"},
            {"status": "configured_but_disabled"},
        ]
    )

    assert list(figure.data[0].x) == ["available", "configured_but_disabled"]
    assert list(figure.data[0].y) == [1, 2]


def test_universe_mix_figure_uses_symbol_counts() -> None:
    figure = charts.universe_mix_figure(
        {"groups": [{"group": "core", "symbol_count": 3, "asset_class": "equity"}]}
    )

    assert list(figure.data[0].x) == ["core"]
    assert list(figure.data[0].y) == [3]


def test_backtest_metric_figure_filters_non_numeric_values() -> None:
    figure = charts.backtest_metric_figure({"final_equity": 10001.0, "sharpe_ratio": "bad"})

    assert "final_equity" in list(figure.data[0].x)
    assert "sharpe_ratio" not in list(figure.data[0].x)


def test_all_chart_builders_return_titled_figures_for_empty_inputs() -> None:
    figures = [
        charts.provider_status_figure([]),
        charts.universe_mix_figure({"groups": []}),
        charts.risk_profile_figure({"profiles": []}),
        charts.dataset_rows_by_asset_figure([]),
        charts.data_quality_warnings_figure({}),
        charts.backtest_metric_figure({}),
        charts.profile_comparison_figure({}),
        charts.equity_curve_figure(None),
        charts.drawdown_curve_figure(None),
        charts.returns_histogram_figure([]),
        charts.var_es_conceptual_figure(),
        charts.transaction_cost_breakdown_figure({}),
    ]

    for figure in figures:
        assert figure.layout.title.text
        assert "secret" not in figure.to_json().lower()


def test_time_series_and_cost_charts_accept_simple_data() -> None:
    equity = charts.equity_curve_figure(
        [{"timestamp": "2024-01-01", "equity": 10000}, {"timestamp": "2024-01-02", "equity": 10010}]
    )
    drawdown = charts.drawdown_curve_figure([{"timestamp": "2024-01-02", "drawdown": -0.01}])
    histogram = charts.returns_histogram_figure([0.01, -0.02, 0.03])
    costs = charts.transaction_cost_breakdown_figure({"commission": 0.1, "slippage": 0.2})

    assert equity.data
    assert drawdown.data
    assert histogram.data
    assert costs.data
