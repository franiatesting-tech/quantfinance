"""Research-only quantitative decision signal engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ALLOWED_SIGNALS = {
    "FAVORABLE",
    "NEUTRAL",
    "CAUTION",
    "UNFAVORABLE",
    "INSUFFICIENT_DATA",
}

DISCLAIMER = (
    "This is not investment advice. It is a quantitative research signal based on "
    "historical data, assumptions, and model limitations."
)


@dataclass(frozen=True)
class DecisionThresholds:
    """Configurable thresholds for the research-only signal."""

    min_observations: int = 756
    favorable_min_sharpe: float = 1.0
    neutral_min_sharpe: float = 0.5
    max_acceptable_drawdown: float = -0.35
    max_var_95_daily_loss: float = 0.05
    max_es_95_daily_loss: float = 0.08
    min_ml_directional_accuracy_edge: float = 0.02
    min_positive_mc_median_return: float = 0.0


def build_research_decision_signal(
    *,
    metrics: dict[str, Any],
    var_results: dict[str, Any] | None = None,
    monte_carlo: dict[str, Any] | None = None,
    ml_forecasting: dict[str, Any] | None = None,
    backtesting: dict[str, Any] | None = None,
    data_quality_warnings: list[str] | tuple[str, ...] | None = None,
    thresholds: DecisionThresholds | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic non-advisory signal from historical research metrics."""

    limits: list[str] = []
    drivers_positive: list[str] = []
    drivers_negative: list[str] = []
    warnings = [str(item) for item in data_quality_warnings or []]
    clean_thresholds = _thresholds(thresholds)

    observations = _float(metrics.get("observations"))
    if observations is None or observations < clean_thresholds.min_observations:
        return _payload(
            signal="INSUFFICIENT_DATA",
            confidence="LOW",
            score=0.0,
            drivers_positive=drivers_positive,
            drivers_negative=[
                f"Observations below minimum threshold ({observations or 0:.0f} < "
                f"{clean_thresholds.min_observations})."
            ],
            limits_triggered=["MIN_OBSERVATIONS_NOT_MET"],
            model_quality={"observations": observations or 0.0, "data_warnings": warnings},
            what_would_change_the_signal=[
                "Use a longer clean history or reduce the minimum observation threshold."
            ],
        )

    score = 0.0
    sharpe = _float(metrics.get("sharpe_ratio"))
    if sharpe is not None:
        if sharpe >= clean_thresholds.favorable_min_sharpe:
            score += 2.0
            drivers_positive.append(
                f"Sharpe is strong under historical assumptions ({sharpe:.2f})."
            )
        elif sharpe >= clean_thresholds.neutral_min_sharpe:
            score += 1.0
            drivers_positive.append(f"Sharpe is positive but moderate ({sharpe:.2f}).")
        else:
            score -= 1.0
            drivers_negative.append(f"Sharpe is below neutral threshold ({sharpe:.2f}).")

    sortino = _float(metrics.get("sortino_ratio"))
    if sortino is not None:
        if sortino >= 1.0:
            score += 1.0
            drivers_positive.append(
                f"Sortino indicates positive downside compensation ({sortino:.2f})."
            )
        elif sortino <= 0.0:
            score -= 1.0
            drivers_negative.append(f"Sortino is non-positive ({sortino:.2f}).")

    drawdown = _float(metrics.get("max_drawdown"))
    if drawdown is not None:
        if drawdown >= clean_thresholds.max_acceptable_drawdown:
            score += 1.0
            drivers_positive.append(f"Max drawdown is inside configured limit ({drawdown:.2%}).")
        else:
            score -= 2.0
            drivers_negative.append(f"Max drawdown breaches configured limit ({drawdown:.2%}).")
            limits.append("MAX_DRAWDOWN_LIMIT_BREACHED")

    cagr = _float(metrics.get("cagr", metrics.get("annualized_return")))
    if cagr is not None:
        if cagr > 0:
            score += 0.5
            drivers_positive.append(f"Historical CAGR/annualized return is positive ({cagr:.2%}).")
        else:
            score -= 0.5
            drivers_negative.append(
                f"Historical CAGR/annualized return is not positive ({cagr:.2%})."
            )

    volatility = _float(metrics.get("annualized_volatility"))
    if volatility is not None and volatility > 0.40:
        score -= 0.5
        drivers_negative.append(f"Annualized volatility is elevated ({volatility:.2%}).")

    beta = _float(metrics.get("beta_to_benchmark"))
    if beta is not None and beta > 1.5:
        score -= 0.5
        drivers_negative.append(f"Benchmark beta is high ({beta:.2f}).")

    alpha = _float(metrics.get("jensen_alpha"))
    if alpha is not None:
        if alpha > 0:
            score += 0.5
            drivers_positive.append(f"Jensen alpha is positive ({alpha:.2%}).")
        elif alpha < -0.05:
            score -= 0.5
            drivers_negative.append(f"Jensen alpha is materially negative ({alpha:.2%}).")

    treynor = _float(metrics.get("treynor_ratio"))
    if treynor is not None and treynor < 0:
        score -= 0.5
        drivers_negative.append(f"Treynor ratio is negative ({treynor:.4f}).")

    _score_var_es(
        var_results or {}, clean_thresholds, drivers_positive, drivers_negative, limits
    )
    score += _tail_score_adjustment(limits, drivers_positive, drivers_negative)

    mc_quality = _monte_carlo_quality(monte_carlo or {}, clean_thresholds)
    score += float(mc_quality["score_adjustment"])
    drivers_positive.extend(mc_quality["drivers_positive"])
    drivers_negative.extend(mc_quality["drivers_negative"])

    ml_quality = _ml_quality(ml_forecasting or {}, clean_thresholds)
    score += float(ml_quality["score_adjustment"])
    drivers_positive.extend(ml_quality["drivers_positive"])
    drivers_negative.extend(ml_quality["drivers_negative"])

    backtest_quality = _backtest_quality(backtesting or {})
    score += float(backtest_quality["score_adjustment"])
    drivers_positive.extend(backtest_quality["drivers_positive"])
    drivers_negative.extend(backtest_quality["drivers_negative"])

    if warnings:
        score -= 0.5
        limits.append("DATA_QUALITY_WARNINGS_PRESENT")
        drivers_negative.append("Data quality warnings are present and reduce confidence.")

    signal = _signal_from_score(score, limits)
    confidence = _confidence(observations, warnings, ml_quality)
    action = _suggested_action(signal)
    return _payload(
        signal=signal,
        confidence=confidence,
        score=round(float(score), 4),
        drivers_positive=drivers_positive,
        drivers_negative=drivers_negative,
        limits_triggered=sorted(set(limits)),
        model_quality={
            "observations": observations,
            "ml": ml_quality["model_quality"],
            "monte_carlo": mc_quality["model_quality"],
            "backtest": backtest_quality["model_quality"],
            "data_warnings": warnings,
        },
        what_would_change_the_signal=_change_conditions(signal),
        suggested_research_action=action,
    )


def _score_var_es(
    var_results: dict[str, Any],
    thresholds: DecisionThresholds,
    drivers_positive: list[str],
    drivers_negative: list[str],
    limits: list[str],
) -> None:
    historical = _mapping(var_results.get("historical"))
    var_95 = _float(historical.get("var", historical.get("var_95")))
    es_95 = _float(historical.get("expected_shortfall", historical.get("es_95")))
    if var_95 is not None:
        if var_95 <= thresholds.max_var_95_daily_loss:
            drivers_positive.append(f"Historical VaR 95 daily loss is inside limit ({var_95:.2%}).")
        else:
            drivers_negative.append(f"Historical VaR 95 daily loss breaches limit ({var_95:.2%}).")
            limits.append("VAR_95_LIMIT_BREACHED")
    if es_95 is not None:
        if es_95 <= thresholds.max_es_95_daily_loss:
            drivers_positive.append(f"Historical ES 95 daily loss is inside limit ({es_95:.2%}).")
        else:
            drivers_negative.append(f"Historical ES 95 daily loss breaches limit ({es_95:.2%}).")
            limits.append("ES_95_LIMIT_BREACHED")


def _tail_score_adjustment(
    limits: list[str], drivers_positive: list[str], drivers_negative: list[str]
) -> float:
    tail_limits = {"VAR_95_LIMIT_BREACHED", "ES_95_LIMIT_BREACHED"}
    if tail_limits.intersection(limits):
        return -1.0
    if any("VaR" in item or "ES" in item for item in drivers_positive) and not any(
        item in limits for item in tail_limits
    ):
        return 1.0
    if drivers_negative:
        return 0.0
    return 0.0


def _monte_carlo_quality(
    monte_carlo: dict[str, Any], thresholds: DecisionThresholds
) -> dict[str, Any]:
    model = _mapping(monte_carlo.get("parametric_normal", monte_carlo))
    median = _float(model.get("terminal_median"))
    prob_loss = _float(model.get("probability_of_loss"))
    positive: list[str] = []
    negative: list[str] = []
    score = 0.0
    if median is not None:
        if median > thresholds.min_positive_mc_median_return:
            score += 0.5
            positive.append(f"Monte Carlo median terminal return is positive ({median:.2%}).")
        else:
            score -= 0.5
            negative.append(f"Monte Carlo median terminal return is not positive ({median:.2%}).")
    if prob_loss is not None and prob_loss > 0.50:
        score -= 0.5
        negative.append(f"Monte Carlo probability of loss exceeds 50% ({prob_loss:.2%}).")
    return {
        "score_adjustment": score,
        "drivers_positive": positive,
        "drivers_negative": negative,
        "model_quality": {"terminal_median": median, "probability_of_loss": prob_loss},
    }


def _ml_quality(ml_forecasting: dict[str, Any], thresholds: DecisionThresholds) -> dict[str, Any]:
    status = str(ml_forecasting.get("status", ml_forecasting.get("model_status", "NOT_RUN")))
    directional_accuracy = _float(ml_forecasting.get("directional_accuracy"))
    baseline_accuracy = _float(ml_forecasting.get("baseline_directional_accuracy", 0.5))
    positive: list[str] = []
    negative: list[str] = []
    score = 0.0
    if directional_accuracy is not None:
        edge = directional_accuracy - (baseline_accuracy or 0.5)
        if edge >= thresholds.min_ml_directional_accuracy_edge:
            score += 0.5
            positive.append(f"ML directional accuracy has out-of-sample edge ({edge:.2%}).")
        else:
            negative.append("ML model is not better than the naive directional baseline.")
    elif status not in {"", "NOT_RUN"}:
        negative.append("ML model quality is unavailable or inconclusive.")
    return {
        "score_adjustment": score,
        "drivers_positive": positive,
        "drivers_negative": negative,
        "model_quality": {
            "status": status,
            "directional_accuracy": directional_accuracy,
            "baseline_directional_accuracy": baseline_accuracy,
        },
    }


def _backtest_quality(backtesting: dict[str, Any]) -> dict[str, Any]:
    buy_hold = _mapping(backtesting.get("buy_and_hold", backtesting))
    metrics = _mapping(buy_hold.get("metrics", buy_hold))
    sharpe = _float(metrics.get("sharpe_ratio"))
    final_equity = _float(metrics.get("final_equity"))
    positive: list[str] = []
    negative: list[str] = []
    score = 0.0
    if sharpe is not None and sharpe > 0.5:
        score += 0.5
        positive.append(f"Backtest Sharpe is positive under historical assumptions ({sharpe:.2f}).")
    if final_equity is not None and final_equity <= 0:
        score -= 1.0
        negative.append("Backtest equity path ended at or below zero.")
    return {
        "score_adjustment": score,
        "drivers_positive": positive,
        "drivers_negative": negative,
        "model_quality": {"sharpe_ratio": sharpe, "final_equity": final_equity},
    }


def _signal_from_score(score: float, limits: list[str]) -> str:
    if "MAX_DRAWDOWN_LIMIT_BREACHED" in limits and score < 1.0:
        return "UNFAVORABLE"
    if {"VAR_95_LIMIT_BREACHED", "ES_95_LIMIT_BREACHED"}.intersection(limits):
        return "CAUTION" if score >= 0 else "UNFAVORABLE"
    if score >= 4.0:
        return "FAVORABLE"
    if score >= 1.0:
        return "NEUTRAL"
    if score >= -1.5:
        return "CAUTION"
    return "UNFAVORABLE"


def _confidence(
    observations: float | None, warnings: list[str], ml_quality: dict[str, Any]
) -> str:
    if warnings:
        return "LOW"
    if observations is None or observations < 1_260:
        return "LOW"
    ml_status = str(ml_quality["model_quality"].get("status", "NOT_RUN"))
    if ml_status == "MODEL_OUTPERFORMS_NAIVE_UNDER_TEST_ASSUMPTIONS":
        return "HIGH"
    return "MEDIUM"


def _suggested_action(signal: str) -> str:
    return {
        "FAVORABLE": "Risk-adjusted profile appears favorable under historical assumptions",
        "NEUTRAL": "Continue monitoring",
        "CAUTION": "Avoid model-driven decision",
        "UNFAVORABLE": "Risk-adjusted profile appears weak under historical assumptions",
        "INSUFFICIENT_DATA": "Avoid model-driven decision",
    }[signal]


def _change_conditions(signal: str) -> list[str]:
    if signal == "FAVORABLE":
        return [
            "Signal would weaken if drawdown, VaR, ES, or out-of-sample model quality deteriorate."
        ]
    if signal == "INSUFFICIENT_DATA":
        return ["A longer, cleaner historical sample is required before scoring the asset."]
    return [
        "Improved risk-adjusted returns, lower drawdown/tail risk, and stronger out-of-sample "
        "model diagnostics could improve the signal."
    ]


def _payload(
    *,
    signal: str,
    confidence: str,
    score: float,
    drivers_positive: list[str],
    drivers_negative: list[str],
    limits_triggered: list[str],
    model_quality: dict[str, Any],
    what_would_change_the_signal: list[str],
    suggested_research_action: str | None = None,
) -> dict[str, Any]:
    if signal not in ALLOWED_SIGNALS:
        raise ValueError(f"Unsupported research signal: {signal}")
    return {
        "signal": signal,
        "confidence": confidence,
        "score": float(score),
        "drivers_positive": drivers_positive,
        "drivers_negative": drivers_negative,
        "limits_triggered": limits_triggered,
        "model_quality": model_quality,
        "what_would_change_the_signal": what_would_change_the_signal,
        "suggested_research_action": suggested_research_action or _suggested_action(signal),
        "not_investment_advice": True,
        "disclaimer": DISCLAIMER,
    }


def _thresholds(value: DecisionThresholds | dict[str, Any] | None) -> DecisionThresholds:
    if value is None:
        return DecisionThresholds()
    if isinstance(value, DecisionThresholds):
        return value
    fields = DecisionThresholds.__dataclass_fields__
    return DecisionThresholds(**{key: value[key] for key in fields if key in value})


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    try:
        clean = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return clean if np.isfinite(clean) else None
