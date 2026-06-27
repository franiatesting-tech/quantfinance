"""Plotly chart builders for the local read-only UI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from quant_platform.ui.view_models import backtest_metric_rows

EMPTY_MESSAGE = "No local report data available yet. Run the safe CLI pipeline first."


def provider_status_figure(provider_rows: list[dict[str, Any]]):  # noqa: ANN201
    """Build a provider status bar chart."""

    go = _plotly_go()
    counts: dict[str, int] = {}
    for provider in provider_rows:
        status = str(provider.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if not counts:
        return _empty_figure("Provider Status")
    figure = go.Figure(
        data=[
            go.Bar(
                x=list(counts),
                y=list(counts.values()),
                marker_color=[_status_color(status) for status in counts],
                hovertemplate="Status=%{x}<br>Providers=%{y}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Provider Status", "Providers", "Count")
    return figure


def universe_mix_figure(universe: dict[str, Any]):  # noqa: ANN201
    """Build a universe group composition chart."""

    go = _plotly_go()
    groups = [group for group in universe.get("groups", []) if isinstance(group, dict)]
    if not groups:
        return _empty_figure("Universe By Group")
    figure = go.Figure(
        data=[
            go.Bar(
                x=[str(group.get("group", "unknown")) for group in groups],
                y=[int(group.get("symbol_count", 0)) for group in groups],
                marker_color=[
                    "#3dd6c6" if group.get("asset_class") == "crypto" else "#d6b35a"
                    for group in groups
                ],
                customdata=[str(group.get("asset_class", "unknown")) for group in groups],
                hovertemplate=(
                    "Group=%{x}<br>Asset class=%{customdata}<br>"
                    "Symbols=%{y}<extra></extra>"
                ),
            )
        ]
    )
    _apply_layout(figure, "Universe By Group", "Universe group", "Symbols")
    figure.update_layout(xaxis_tickangle=-35)
    _add_note(figure, "Research universe only; not an investment recommendation.")
    return figure


def risk_profile_figure(risk_profiles: dict[str, Any]):  # noqa: ANN201
    """Build a risk profile limits chart."""

    go = _plotly_go()
    rows = [row for row in risk_profiles.get("profiles", []) if isinstance(row, dict)]
    if not rows:
        return _empty_figure("Risk Profile Limits")
    figure = go.Figure()
    for metric, color in (
        ("target_max_drawdown", "#d6b35a"),
        ("max_single_asset_weight", "#3dd6c6"),
        ("max_turnover", "#d66a4a"),
    ):
        figure.add_trace(
            go.Bar(
                name=metric,
                x=[row.get("profile") for row in rows],
                y=[row.get(metric) for row in rows],
                marker_color=color,
                hovertemplate=f"Profile=%{{x}}<br>{metric}=%{{y:.2%}}<extra></extra>",
            )
        )
    _apply_layout(figure, "Risk Profile Limits", "Profile", "Limit")
    _add_note(figure, "Drawdown targets are evaluation thresholds, not guarantees.")
    return figure


def dataset_rows_by_asset_figure(rows: list[dict[str, Any]]):  # noqa: ANN201
    """Build a rows-by-asset figure from quality coverage rows."""

    go = _plotly_go()
    clean_rows = [row for row in rows if isinstance(row, dict)]
    if not clean_rows:
        return _empty_figure("Dataset Rows By Asset")
    figure = go.Figure(
        data=[
            go.Bar(
                x=[str(row.get("asset_id", row.get("symbol", "unknown"))) for row in clean_rows],
                y=[int(row.get("row_count", 0) or 0) for row in clean_rows],
                marker_color="#d6b35a",
                hovertemplate="Asset=%{x}<br>Rows=%{y}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Dataset Rows By Asset", "Asset", "Rows")
    figure.update_layout(xaxis_tickangle=-35)
    return figure


def data_quality_warnings_figure(report: dict[str, Any]):  # noqa: ANN201
    """Build a warning and failed-check count figure from one quality report."""

    warnings = len(report.get("warnings", []) or [])
    failed_checks = len(report.get("failed_checks", []) or [])
    gaps = len(report.get("date_gaps", []) or [])
    if warnings == 0 and failed_checks == 0 and gaps == 0:
        return _empty_figure("Data Quality Warnings", message="No warnings or failed checks found.")
    go = _plotly_go()
    figure = go.Figure(
        data=[
            go.Bar(
                x=["warnings", "failed_checks", "date_gaps"],
                y=[warnings, failed_checks, gaps],
                marker_color=["#d6b35a", "#d66a4a", "#88929c"],
                hovertemplate="Item=%{x}<br>Count=%{y}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Data Quality Warnings", "Issue type", "Count")
    _add_note(figure, "Warnings require review; they are not automatically fatal.")
    return figure


def backtest_metric_figure(report: dict[str, Any]):  # noqa: ANN201
    """Build a compact backtest metric bar chart."""

    rows = [row for row in backtest_metric_rows(report) if row["value"] is not None]
    if not rows:
        return _empty_figure("Backtest Metrics")
    go = _plotly_go()
    figure = go.Figure(
        data=[
            go.Bar(
                x=[str(row["metric"]) for row in rows],
                y=[row["value"] for row in rows],
                marker_color="#3dd6c6",
                hovertemplate="Metric=%{x}<br>Value=%{y:.6f}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Backtest Metrics", "Metric", "Value")
    figure.update_layout(xaxis_tickangle=-35)
    _add_note(figure, "Backtest metrics describe a historical simulation, not a forecast.")
    return figure


def profile_comparison_figure(report: dict[str, Any]):  # noqa: ANN201
    """Build a profile comparison chart from comparison_table rows."""

    rows = [row for row in report.get("comparison_table", []) if isinstance(row, dict)]
    profiles = sorted(str(profile) for profile in report.get("profiles", {}))
    if not rows or not profiles:
        return _empty_figure("Profile Comparison")
    go = _plotly_go()
    figure = go.Figure()
    for profile, color in zip(profiles, ("#d6b35a", "#3dd6c6", "#d66a4a"), strict=False):
        figure.add_trace(
            go.Bar(
                name=profile,
                x=[row.get("metric") for row in rows],
                y=[row.get(profile) for row in rows],
                marker_color=color,
                hovertemplate=f"Profile={profile}<br>Metric=%{{x}}<br>Value=%{{y:.6f}}<extra></extra>",
            )
        )
    _apply_layout(figure, "Profile Comparison", "Metric", "Value")
    figure.update_layout(xaxis_tickangle=-35, barmode="group")
    _add_note(
        figure,
        "Aggressive profiles can have higher risk and are not guaranteed to earn more.",
    )
    return figure


def equity_curve_figure(series: Sequence[dict[str, Any]] | dict[str, Any] | None):  # noqa: ANN201
    """Build an equity curve figure from simple time/value rows."""

    rows = _series_rows(series, value_keys=("equity", "value", "capital"))
    if not rows:
        return _empty_figure("Equity Curve")
    go = _plotly_go()
    figure = go.Figure(
        data=[
            go.Scatter(
                x=[row["timestamp"] for row in rows],
                y=[row["value"] for row in rows],
                mode="lines",
                line={"color": "#3dd6c6", "width": 3},
                hovertemplate="Date=%{x}<br>Equity=%{y:.2f}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Equity Curve", "Date", "Simulated capital")
    _add_note(figure, "Capital is fictitious and based on historical data.")
    return figure


def drawdown_curve_figure(series: Sequence[dict[str, Any]] | dict[str, Any] | None):  # noqa: ANN201
    """Build a drawdown curve figure from simple time/value rows."""

    rows = _series_rows(series, value_keys=("drawdown", "value"))
    if not rows:
        return _empty_figure("Drawdown Curve")
    go = _plotly_go()
    figure = go.Figure(
        data=[
            go.Scatter(
                x=[row["timestamp"] for row in rows],
                y=[row["value"] for row in rows],
                mode="lines",
                fill="tozeroy",
                line={"color": "#d66a4a", "width": 2},
                hovertemplate="Date=%{x}<br>Drawdown=%{y:.2%}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Drawdown Curve", "Date", "Drawdown")
    _add_note(figure, "Future drawdown can be worse than historical drawdown.")
    return figure


def returns_histogram_figure(returns: Sequence[float] | None):  # noqa: ANN201
    """Build a return-distribution histogram."""

    clean = [_clean_float(value) for value in (returns or [])]
    values = [value for value in clean if value is not None]
    if not values:
        return _empty_figure("Returns Histogram")
    go = _plotly_go()
    figure = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=min(30, max(5, len(values))),
                marker_color="#d6b35a",
                hovertemplate="Return bin=%{x}<br>Count=%{y}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Returns Histogram", "Return", "Count")
    _add_note(figure, "Historical distribution can miss rare future events.")
    return figure


def var_es_conceptual_figure(var_value: float | None = None, es_value: float | None = None):  # noqa: ANN201
    """Build a conceptual VaR/ES tail-loss chart."""

    var_clean = 0.03 if var_value is None else float(var_value)
    es_clean = max(var_clean, 0.05 if es_value is None else float(es_value))
    go = _plotly_go()
    losses = [0.0, var_clean / 2, var_clean, (var_clean + es_clean) / 2, es_clean]
    density = [20, 14, 8, 4, 2]
    figure = go.Figure(
        data=[
            go.Scatter(
                x=losses,
                y=density,
                mode="lines+markers",
                fill="tozeroy",
                line={"color": "#d66a4a"},
                hovertemplate="Loss=%{x:.2%}<br>Conceptual density=%{y}<extra></extra>",
            )
        ]
    )
    figure.add_vline(x=var_clean, line_color="#d6b35a", line_dash="dash")
    figure.add_vline(x=es_clean, line_color="#3dd6c6", line_dash="dot")
    _apply_layout(figure, "VaR / Expected Shortfall Concept", "Positive loss", "Conceptual density")
    _add_note(figure, "VaR is a threshold; ES estimates average loss beyond the threshold.")
    return figure


def transaction_cost_breakdown_figure(costs: dict[str, float] | None):  # noqa: ANN201
    """Build a transaction-cost component chart."""

    values = {key: _clean_float(value) for key, value in (costs or {}).items()}
    clean = {key: value for key, value in values.items() if value is not None}
    if not clean:
        return _empty_figure("Transaction Costs Breakdown")
    go = _plotly_go()
    figure = go.Figure(
        data=[
            go.Bar(
                x=list(clean),
                y=list(clean.values()),
                marker_color=["#d6b35a", "#3dd6c6", "#d66a4a", "#88929c"][: len(clean)],
                hovertemplate="Component=%{x}<br>Cost=%{y:.6f}<extra></extra>",
            )
        ]
    )
    _apply_layout(figure, "Transaction Costs Breakdown", "Component", "Cost")
    _add_note(figure, "Costs are simplified research assumptions, not execution guarantees.")
    return figure


def quality_coverage_figure(report: dict[str, Any]):  # noqa: ANN201
    """Build a data-quality coverage chart."""

    coverage = report.get("coverage_by_asset", [])
    return dataset_rows_by_asset_figure(coverage if isinstance(coverage, list) else [])


def _plotly_go():  # noqa: ANN202
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("Install the ui extra to use Plotly charts: .[ui]") from exc
    return go


def _empty_figure(title: str, message: str = EMPTY_MESSAGE):  # noqa: ANN202
    go = _plotly_go()
    figure = go.Figure()
    _apply_layout(figure, title, "", "")
    figure.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 15, "color": "#d6b35a"},
    )
    return figure


def _apply_layout(figure, title: str, xaxis_title: str, yaxis_title: str) -> None:  # noqa: ANN001
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#e8ecef"},
        margin={"l": 24, "r": 18, "t": 60, "b": 78},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )


def _add_note(figure, text: str) -> None:  # noqa: ANN001
    figure.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.22,
        showarrow=False,
        align="left",
        font={"size": 12, "color": "#b8c0c8"},
    )


def _status_color(status: str) -> str:
    return {
        "available": "#3dd6c6",
        "configured_but_disabled": "#d6b35a",
        "missing_api_key": "#d66a4a",
        "disabled": "#6a7480",
    }.get(status, "#88929c")


def _series_rows(
    series: Sequence[dict[str, Any]] | dict[str, Any] | None,
    value_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    if series is None:
        return []
    if isinstance(series, dict):
        raw_rows = [
            {"timestamp": key, "value": value}
            for key, value in series.items()
            if _clean_float(value) is not None
        ]
    else:
        raw_rows = [row for row in series if isinstance(row, dict)]
    rows = []
    for row in raw_rows:
        timestamp = row.get("timestamp") or row.get("date") or row.get("time")
        value = None
        for key in value_keys:
            value = _clean_float(row.get(key))
            if value is not None:
                break
        if timestamp is not None and value is not None:
            rows.append({"timestamp": str(timestamp), "value": value})
    return rows


def _clean_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
