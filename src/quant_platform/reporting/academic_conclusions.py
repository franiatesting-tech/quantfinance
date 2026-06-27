"""Deterministic interpretations for academic stock reports."""
# ruff: noqa: E501

from __future__ import annotations

from typing import Any

SAFE_PREFIX = "En el periodo analizado"


def interpret_performance(metrics: dict[str, Any]) -> str:
    """Interpret return and performance metrics without recommendations."""

    annual_return = _float(metrics.get("annualized_return"))
    sharpe = _float(metrics.get("sharpe_ratio"))
    final_return = _float(metrics.get("final_cumulative_return", metrics.get("total_return")))
    direction = "positivo" if final_return is not None and final_return >= 0 else "negativo"
    sharpe_note = _ratio_quality(sharpe, "Sharpe")
    return (
        f"{SAFE_PREFIX}, el rendimiento acumulado fue {direction}. "
        f"La rentabilidad anualizada estimada fue {_pct(annual_return)}. {sharpe_note} "
        "Esto describe comportamiento historico bajo la muestra disponible y no implica prediccion."
    )


def interpret_risk(metrics: dict[str, Any]) -> str:
    """Interpret volatility, drawdown and distribution shape."""

    volatility = _float(metrics.get("annualized_volatility"))
    drawdown = _float(metrics.get("max_drawdown"))
    skewness = _float(metrics.get("skewness"))
    kurtosis = _float(metrics.get("kurtosis"))
    return (
        f"Historicamente, la volatilidad anualizada fue {_pct(volatility)} y el maximo drawdown "
        f"fue {_pct(drawdown)}. La asimetria fue {_num(skewness)} y la curtosis fue "
        f"{_num(kurtosis)}, por lo que las colas deben revisarse con cautela. "
        "La muestra historica puede omitir eventos extremos futuros."
    )


def interpret_capm(metrics: dict[str, Any]) -> str:
    """Interpret CAPM-derived metrics."""

    beta = _float(metrics.get("beta_to_benchmark"))
    alpha = _float(metrics.get("jensen_alpha"))
    treynor = _float(metrics.get("treynor_ratio"))
    beta_note = (
        "similar al benchmark"
        if beta is not None and 0.8 <= beta <= 1.2
        else "distinta al benchmark"
    )
    return (
        f"Bajo el benchmark usado, beta fue {_num(beta)}, lo que sugiere sensibilidad {beta_note}. "
        f"Jensen alpha fue {_pct(alpha)} y Treynor fue {_num(treynor)}. "
        "CAPM usa un benchmark imperfecto del mercado y sus resultados dependen de la ventana temporal."
    )


def interpret_var(var_results: dict[str, Any]) -> str:
    """Interpret VaR and ES results with positive-loss convention."""

    historical = _mapping(var_results.get("historical"))
    parametric = _mapping(var_results.get("parametric_normal"))
    monte_carlo = _mapping(var_results.get("monte_carlo"))
    return (
        "Usando perdidas positivas, el VaR historico fue "
        f"{_pct(historical.get('var'))} y el ES historico fue "
        f"{_pct(historical.get('expected_shortfall'))}. El VaR parametrico fue "
        f"{_pct(parametric.get('var'))}; el VaR Monte Carlo fue {_pct(monte_carlo.get('var'))}. "
        "VaR es un umbral; ES resume la cola mas severa y suele ser mas informativo para extremos."
    )


def interpret_monte_carlo(mc_results: dict[str, Any]) -> str:
    """Interpret Monte Carlo paths without implying prediction."""

    normal = _mapping(mc_results.get("parametric_normal"))
    p05 = _float(normal.get("terminal_p05"))
    median = _float(normal.get("terminal_median"))
    p95 = _float(normal.get("terminal_p95"))
    loss_probability = _float(normal.get("probability_of_loss"))
    return (
        "Bajo los supuestos del modelo parametrico normal, la distribucion terminal tuvo "
        f"p5={_pct(p05)}, mediana={_pct(median)} y p95={_pct(p95)}. "
        f"La probabilidad simulada de perdida fue {_pct(loss_probability)}. "
        "La simulacion sugiere escenarios posibles bajo supuestos, no trayectorias esperadas."
    )


def interpret_backtests(backtest_results: dict[str, Any]) -> str:
    """Interpret backtest outputs safely."""

    buy_hold = _mapping(backtest_results.get("buy_and_hold"))
    metrics = _mapping(buy_hold.get("metrics"))
    final_equity = _float(metrics.get("final_equity"))
    drawdown = _float(metrics.get("max_drawdown"))
    return (
        f"La simulacion buy-and-hold termino con capital ficticio {_num(final_equity)} y "
        f"maximo drawdown {_pct(drawdown)}. Las senales y la contabilidad son historicas; "
        "un backtest no representa ejecucion real ni permite inferir resultados fuera de muestra."
    )


def interpret_options(options_results: dict[str, Any]) -> str:
    """Interpret theoretical option analytics."""

    call = _float(options_results.get("black_scholes_call"))
    put = _float(options_results.get("black_scholes_put"))
    volatility = _float(options_results.get("volatility"))
    return (
        "Bajo Black-Scholes-Merton parametrico, el call teorico fue "
        f"{_num(call)} y el put teorico fue {_num(put)}, usando volatilidad historica "
        f"{_pct(volatility)}. No se usa option chain real; el bloque es "
        "PARAMETRIC_EDUCATIONAL_MODEL."
    )


def build_stock_conclusions(stock_report: dict[str, Any]) -> dict[str, str]:
    """Build a complete deterministic conclusion set for one stock."""

    metrics = _mapping(stock_report.get("metrics"))
    conclusions = {
        "Historical performance": interpret_performance(metrics),
        "Risk profile": interpret_risk(metrics),
        "CAPM interpretation": interpret_capm(metrics),
        "Tail-risk interpretation": interpret_var(_mapping(stock_report.get("var"))),
        "Monte Carlo interpretation": interpret_monte_carlo(
            _mapping(stock_report.get("monte_carlo"))
        ),
        "Backtesting interpretation": interpret_backtests(
            _mapping(stock_report.get("backtesting_results"))
        ),
        "Options interpretation": interpret_options(
            _mapping(stock_report.get("options_theoretical_analytics"))
        ),
        "Overall research conclusion": (
            "En conjunto, el informe describe evidencia historica, sensibilidad al benchmark, "
            "riesgo de cola y escenarios simulados bajo supuestos transparentes. No es una "
            "instruccion operativa ni asesoramiento financiero."
        ),
        "What should not be concluded": (
            "No debe inferirse una decision de entrada o salida, una rentabilidad futura, una "
            "proteccion asegurada frente a perdidas ni una valoracion profesional de derivados."
        ),
    }
    return conclusions


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float | None:
    try:
        clean = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if clean != clean:
        return None
    return clean


def _pct(value: object) -> str:
    clean = _float(value)
    return "N/A" if clean is None else f"{clean:.2%}"


def _num(value: object) -> str:
    clean = _float(value)
    return "N/A" if clean is None else f"{clean:,.4f}"


def _ratio_quality(value: float | None, name: str) -> str:
    if value is None:
        return f"{name} no es interpretable con la muestra disponible."
    if value > 1.0:
        return f"{name} fue superior a 1, indicando retorno historico elevado por unidad de riesgo."
    if value > 0.0:
        return f"{name} fue positivo, aunque debe leerse con incertidumbre estadistica."
    return f"{name} fue no positivo, senalando compensacion historica debil por volatilidad."
