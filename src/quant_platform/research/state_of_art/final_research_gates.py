"""Final research-only decision gates."""

from __future__ import annotations

from typing import Any

RESEARCH_ONLY_DISCLAIMER = (
    "This is not investment advice. This is a research-only quantitative signal based on "
    "historical data, assumptions and model limitations."
)

ALLOWED_SIGNALS = {
    "FAVORABLE_FOR_FURTHER_RESEARCH",
    "NEUTRAL_MONITOR",
    "CAUTION_RISK_REVIEW",
    "UNFAVORABLE_UNDER_CURRENT_EVIDENCE",
    "INSUFFICIENT_DATA",
    "BLOCKED_FOR_BROKER_PROMOTION",
}


def final_research_decision(
    *,
    data_warnings: bool,
    risk_free_proxy_available: bool,
    predictive_edge_validated: bool,
    severe_drawdown: bool,
    tail_backtest_passed: bool | None,
    model_confidence_status: str | None,
) -> dict[str, Any]:
    """Map evidence into the final allowed research-only signal vocabulary."""

    blockers = []
    if data_warnings:
        blockers.append("provider_data_warnings_present")
    if not risk_free_proxy_available:
        blockers.append("risk_free_proxy_unavailable")
    if not predictive_edge_validated:
        blockers.append("predictive_edge_not_validated")
    if severe_drawdown:
        blockers.append("severe_historical_drawdown")
    if tail_backtest_passed is False:
        blockers.append("tail_risk_backtest_failed")
    if model_confidence_status not in {"MODEL_CONFIDENCE_ACCEPTED", "MODEL_CONFIDENCE_WEAK"}:
        blockers.append("model_confidence_not_accepted")
    if data_warnings or not risk_free_proxy_available or tail_backtest_passed is False:
        signal = "BLOCKED_FOR_BROKER_PROMOTION"
    elif severe_drawdown:
        signal = "CAUTION_RISK_REVIEW"
    elif predictive_edge_validated and model_confidence_status == "MODEL_CONFIDENCE_ACCEPTED":
        signal = "FAVORABLE_FOR_FURTHER_RESEARCH"
    elif predictive_edge_validated:
        signal = "NEUTRAL_MONITOR"
    else:
        signal = "UNFAVORABLE_UNDER_CURRENT_EVIDENCE"
    return {
        "research_only_signal": signal,
        "live_order_allowed": False,
        "paper_trading_allowed": False,
        "broker_promotion_allowed": False,
        "blocking_conditions": "; ".join(blockers),
        "disclaimer": RESEARCH_ONLY_DISCLAIMER,
    }
