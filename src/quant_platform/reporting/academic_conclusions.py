"""Deterministic interpretations for academic stock reports."""
# ruff: noqa: E501

from __future__ import annotations

from typing import Any

from quant_platform.research.decision_engine import (
    DISCLAIMER,
    build_research_decision_signal,
)

SAFE_PREFIX = "En el periodo analizado"


def interpret_performance(metrics: dict[str, Any]) -> str:
    """Interpret return and performance metrics with economic context."""

    annual_return = _float(metrics.get("annualized_return"))
    cagr = _float(metrics.get("cagr"))
    sharpe = _float(metrics.get("sharpe_ratio"))
    sortino = _float(metrics.get("sortino_ratio"))
    ann_vol = _float(metrics.get("annualized_volatility"))
    hit_rate = _float(metrics.get("hit_rate"))
    calmar = _float(metrics.get("calmar_ratio"))
    final_return = _float(metrics.get("final_cumulative_return", metrics.get("total_return")))
    direction = "positivo" if final_return is not None and final_return >= 0 else "negativo"

    vol_note = ""
    if ann_vol is not None:
        if ann_vol > 0.30:
            vol_note = "volatilidad alta (>30% anual), lo que implica riesgo elevado"
        elif ann_vol > 0.20:
            vol_note = "volatilidad moderada-alta (20-30% anual)"
        elif ann_vol > 0.12:
            vol_note = "volatilidad moderada (12-20% anual), dentro de lo esperable para renta variable"
        else:
            vol_note = "volatilidad baja (<12% anual), tipica de activos defensivos"

    sharpe_note = _ratio_quality(sharpe, "Sharpe")
    sortino_note = ""
    if sortino is not None:
        if sortino > 2.0:
            sortino_note = "El Sortino > 2 senala que la rentabilidad historica compensa ampliamente el riesgo de caida."
        elif sortino > 1.0:
            sortino_note = "El Sortino > 1 indica buena compensacion del riesgo downside."
        elif sortino > 0:
            sortino_note = "El Sortino positivo sugiere que el retorno supera la volatilidad negativa."
        else:
            sortino_note = "El Sortino no positivo advierte que las caidas no fueron compensadas."

    return (
        f"{SAFE_PREFIX}, el rendimiento acumulado fue {direction}. "
        f"La rentabilidad anualizada fue {_pct(annual_return)} (CAGR: {_pct(cagr)}) "
        f"con {vol_note}. "
        f"Sharpe: {_num(sharpe)}; Sortino: {_num(sortino)}; "
        f"Calmar: {_num(calmar)}; Hit rate: {_pct(hit_rate)}. "
        f"{sharpe_note} {sortino_note} "
        "En terminos economicos, la muestra historica muestra compensacion "
        "por el riesgo asumido. Este resultado historico no implica resultados futuros."
    )


def interpret_risk(metrics: dict[str, Any]) -> str:
    """Interpret volatility, drawdown and distribution shape with economic significance."""

    volatility = _float(metrics.get("annualized_volatility"))
    drawdown = _float(metrics.get("max_drawdown"))
    skewness = _float(metrics.get("skewness"))
    kurtosis = _float(metrics.get("kurtosis"))

    dd_note = ""
    if drawdown is not None:
        if abs(drawdown) > 0.40:
            dd_note = "Este nivel de drawdown (>40%) implica que el activo sufrio caidas muy severas, " \
                      "lo que habria requerido una alta tolerancia al riesgo psicologico y financiero."
        elif abs(drawdown) > 0.25:
            dd_note = "Con un drawdown maximo entre 25-40%, el activo presento caidas importantes " \
                      "que son tipicas en mercados bajistas severos."
        elif abs(drawdown) > 0.15:
            dd_note = "El drawdown entre 15-25% es consistente con correcciones normales de mercado " \
                      "y refleja un riesgo asumible para perfiles moderados."
        else:
            dd_note = "El drawdown <15% sugiere un perfil defensivo con riesgo de caida limitado."

    skew_note = ""
    if skewness is not None:
        if skewness < -0.5:
            skew_note = "La asimetria negativa indica cola izquierda mas pesada, " \
                        "es decir, mayor probabilidad de retornos extremadamente negativos que positivos."
        elif skewness > 0.5:
            skew_note = "La asimetria positiva es favorable: cola derecha mas pesada, " \
                        "los retornos positivos extremos son mas probables que los negativos."
        else:
            skew_note = "La asimetria cercana a cero sugiere una distribucion aproximadamente simetrica."

    kurt_note = ""
    if kurtosis is not None:
        if kurtosis > 4:
            kurt_note = "La curtosis elevada (>4) alerta de colas gruesas: " \
                        "mayor frecuencia de eventos extremos de lo que predecia el modelo normal."
        elif kurtosis > 3:
            kurt_note = "Curtosis moderadamente superior a la normal (3): " \
                        "ligero exceso de eventos extremos."
        else:
            kurt_note = "Curtosis cercana a 3 (distribucion normal): " \
                        "los eventos extremos son consistentes con lo esperable."

    return (
        f"Historicamente, la volatilidad anualizada fue {_pct(volatility)}. "
        f"El maximo drawdown fue {_pct(drawdown)}. {dd_note} "
        f"Asimetria (skewness): {_num(skewness)}. {skew_note} "
        f"Curtosis (kurtosis): {_num(kurtosis)}. {kurt_note} "
        "IMPORTANTE: La muestra historica (datos sinteticos de demostracion) "
        "puede no reflejar eventos extremos futuros ni riesgos de cola reales."
    )


def interpret_capm(metrics: dict[str, Any]) -> str:
    """Interpret CAPM-derived metrics with economic meaning."""

    beta = _float(metrics.get("beta_to_benchmark"))
    alpha = _float(metrics.get("jensen_alpha"))
    treynor = _float(metrics.get("treynor_ratio"))
    if beta is None:
        return "Beta no disponible para este activo."
    if beta < 0.5:
        beta_note = f"Beta de {_num(beta)}: el activo es defensivo (baja correlacion con el mercado). " \
                    "En teoria, deberia caer menos en mercados bajistas, pero tambien subir menos en alcistas."
    elif beta < 0.8:
        beta_note = f"Beta de {_num(beta)}: el activo tiene sensibilidad inferior al mercado. " \
                    "Riesgo sistematico reducido, comportamiento algo defensivo."
    elif beta <= 1.2:
        beta_note = f"Beta de {_num(beta)} cercano a 1: el activo se mueve en linea con el mercado. " \
                    "Su riesgo sistematico es similar al del benchmark."
    elif beta <= 1.5:
        beta_note = f"Beta de {_num(beta)} superior a 1: el activo es agresivo. " \
                    "Amplifica los movimientos del mercado, tanto al alza como a la baja."
    else:
        beta_note = f"Beta de {_num(beta)} muy elevado: alta volatilidad relativa al mercado. " \
                    "Implica riesgo sistematico significativamente superior al benchmark."

    alpha_note = ""
    if alpha is not None:
        if alpha > 0.05:
            alpha_note = f"El alpha de Jensen de {_pct(alpha)} es positivo y relevante: " \
                         "el activo ha generado retorno extraordinario por encima del esperado por su beta, " \
                         "lo que sugiere que la gestion o las caracteristicas del activo anadieron valor."
        elif alpha > 0:
            alpha_note = f"Alpha positivo ({_pct(alpha)}) pero modesto: " \
                         "el activo supero ligeramente al mercado ajustado por riesgo."
        elif alpha > -0.05:
            alpha_note = f"Alpha ligeramente negativo ({_pct(alpha)}): " \
                         "el activo rindio por debajo del esperado por su riesgo sistematico."
        else:
            alpha_note = f"Alpha negativo significativo ({_pct(alpha)}): " \
                         "el activo destruyo valor respecto al mercado ajustado por riesgo."

    return (
        f"Bajo el benchmark usado, {beta_note} "
        f"{alpha_note} "
        f"Treynor ratio: {_num(treynor)}. "
        "CAPM usa un benchmark imperfecto como proxy del mercado. "
        "Beta y alpha pueden variar significativamente segun la ventana temporal y el benchmark elegido."
    )


def interpret_var(var_results: dict[str, Any]) -> str:
    """Interpret VaR and ES results with economic significance."""

    historical = _mapping(var_results.get("historical"))
    parametric = _mapping(var_results.get("parametric_normal"))
    monte_carlo = _mapping(var_results.get("monte_carlo"))
    hist_var = _float(historical.get("var"))
    hist_es = _float(historical.get("expected_shortfall"))
    par_var = _float(parametric.get("var"))
    mc_var = _float(monte_carlo.get("var"))

    var_note = ""
    if hist_var is not None:
        if hist_var > 0.05:
            var_note = "El VaR >5% indica un riesgo diario significativo: " \
                       "podria perder mas del 5% en el peor 5% de los dias."
        elif hist_var > 0.02:
            var_note = "VaR entre 2-5%: riesgo diario moderado, " \
                       "tipico de activos de renta variable con volatilidad media."
        else:
            var_note = "VaR <2%: riesgo diario acotado, " \
                       "propio de activos de baja volatilidad o carteras diversificadas."

    es_note = ""
    if hist_es is not None and hist_var is not None and hist_es > hist_var:
        ratio = hist_es / hist_var if hist_var != 0 else 0
        es_note = f" ES > VaR ({ratio:.1f}x) confirma cola gruesa: " \
                   "la perdida media en los peores dias supera ampliamente el umbral VaR."
    elif hist_es is not None:
        es_note = f" ES={_pct(hist_es)}."

    return (
        f"Usando perdidas positivas (alpha=95%): "
        f"VaR historico={_pct(hist_var)}, ES historico={_pct(hist_es)}. {var_note} {es_note} "
        f"VaR parametrico normal={_pct(par_var)}, VaR Monte Carlo={_pct(mc_var)}. "
        "IMPORTANTE: ES (Expected Shortfall) es mas informativo que VaR porque cuantifica "
        "la perdida media en el peor escenario, no solo el umbral. "
        "Discrepancias entre modelos sugieren que la distribucion real se desvia de la normal."
    )


def interpret_monte_carlo(mc_results: dict[str, Any]) -> str:
    """Interpret Monte Carlo paths with economic context."""

    normal = _mapping(mc_results.get("parametric_normal"))
    p05 = _float(normal.get("terminal_p05"))
    median = _float(normal.get("terminal_median"))
    p95 = _float(normal.get("terminal_p95"))
    loss_probability = _float(normal.get("probability_of_loss"))

    if median is not None:
        if median > 0:
            mc_note = f"La mediana positiva ({_pct(median)}) sugiere que el escenario central es favorable."
        else:
            mc_note = f"La mediana negativa ({_pct(median)}) advierte que incluso el escenario central es pesimista."
    else:
        mc_note = ""

    loss_note = ""
    if loss_probability is not None:
        if loss_probability > 0.40:
            loss_note = f"Probabilidad de perdida elevada ({_pct(loss_probability)}): " \
                        "mas del 40% de las simulaciones terminan en negativo."
        elif loss_probability > 0.25:
            loss_note = f"Probabilidad de perdida moderada ({_pct(loss_probability)}): " \
                        "aproximadamente 1 de cada 4 simulaciones es negativa."
        else:
            loss_note = f"Probabilidad de perdida baja ({_pct(loss_probability)}): " \
                        "la mayoria de escenarios proyectan resultados positivos."

    return (
        "Modelo parametrico normal (GBM, 1000 simulaciones, horizonte 1 ano): "
        f"P5={_pct(p05)}, Mediana={_pct(median)}, P95={_pct(p95)}. "
        f"El intervalo P5-P95 contiene el 90% de los resultados simulados. "
        f"{mc_note} {loss_note} "
        "ADVERTENCIA: Monte Carlo explora escenarios bajo supuestos matematicos explicitos "
        "(normalidad, GBM). No predice el futuro ni captura riesgos de cola extrema, "
        "cambios de regimen o eventos cisne negro."
    )


def interpret_backtests(backtest_results: dict[str, Any]) -> str:
    """Interpret backtest outputs with economic conclusions."""

    buy_hold = _mapping(backtest_results.get("buy_and_hold"))
    metrics = _mapping(buy_hold.get("metrics"))
    final_equity = _float(metrics.get("final_equity"))
    drawdown = _float(metrics.get("max_drawdown"))

    perf_note = ""
    if final_equity is not None:
        gain = final_equity - 10000.0
        gain_pct = gain / 10000.0 if gain != 0 else 0
        if gain > 0:
            perf_note = f"La estrategia buy-and-hold genero una ganancia de {_num(gain)} EUR " \
                        f"({_pct(gain_pct)} sobre 10.000 EUR de capital inicial)."
        else:
            perf_note = f"La estrategia buy-and-hold habria perdido {_num(abs(gain))} EUR " \
                        f"({_pct(abs(gain_pct))} de perdida sobre 10.000 EUR iniciales)."

    return (
        f"Capital inicial ficticio: 10.000 EUR. "
        f"Resultado buy-and-hold: capital final={_num(final_equity)} EUR, "
        f"maximo drawdown={_pct(drawdown)}. "
        f"{perf_note} "
        "IMPORTANTE: El backtest es una simulacion historica sin costes de transaccion, "
        "sin impacto de mercado, sin fiscalidad y sin ejecucion real. "
        "Los resultados historicos no implican rentabilidades futuras."
    )


def interpret_options(options_results: dict[str, Any]) -> str:
    """Interpret theoretical option analytics with context."""

    call = _float(options_results.get("black_scholes_call"))
    put = _float(options_results.get("black_scholes_put"))
    volatility = _float(options_results.get("volatility"))
    return (
        "Bajo Black-Scholes-Merton parametrico (opcion ATM, madurez configurada, "
        f"volatilidad historica {_pct(volatility)}): "
        f"precio teorico call={_num(call)} EUR, put={_num(put)} EUR. "
        "La paridad put-call se evalua con dividend yield continuo si esta configurado. "
        "ADVERTENCIA: Valoracion puramente academica. Sin option chain real, "
        "sin smile de volatilidad ni microestructura. PARAMETRIC_EDUCATIONAL_MODEL."
    )


def interpret_ml(ml_results: dict[str, Any]) -> str:
    """Interpret walk-forward ML diagnostics without predictive overclaiming."""

    status = str(ml_results.get("status", ml_results.get("model_status", "MODEL_NOT_RUN")))
    rmse = _float(ml_results.get("rmse"))
    mae = _float(ml_results.get("mae"))
    directional_accuracy = _float(ml_results.get("directional_accuracy"))
    baseline = _float(ml_results.get("baseline_directional_accuracy"))
    return (
        "Bajo los supuestos de validacion walk-forward, el bloque ML debe leerse como "
        "diagnostico out-of-sample, no como capacidad predictiva asegurada. "
        f"Estado del modelo: {status}. RMSE={_num(rmse)}, MAE={_num(mae)}, "
        f"directional accuracy={_pct(directional_accuracy)}, baseline={_pct(baseline)}. "
        "Si el modelo no supera al baseline naive, la salida correcta es mantener cautela "
        "sobre decisiones impulsadas por el modelo."
    )


def build_stock_conclusions(stock_report: dict[str, Any], asset_id: str = "") -> dict[str, str]:
    """Build a complete deterministic conclusion set for one stock."""

    metrics = _mapping(stock_report.get("metrics"))
    decision_signal = build_research_decision_signal(
        metrics=metrics,
        var_results=_mapping(stock_report.get("var")),
        monte_carlo=_mapping(stock_report.get("monte_carlo")),
        ml_forecasting=_mapping(stock_report.get("ml_forecasting")),
        backtesting=_mapping(stock_report.get("backtesting_results")),
        data_quality_warnings=list(stock_report.get("warnings", [])),
    )
    conclusions = {
        "Historical performance": interpret_performance(metrics),
        "Risk profile": interpret_risk(metrics),
        "CAPM interpretation": interpret_capm(metrics),
        "Tail-risk interpretation": interpret_var(_mapping(stock_report.get("var"))),
        "Monte Carlo interpretation": interpret_monte_carlo(
            _mapping(stock_report.get("monte_carlo"))
        ),
        "ML forecasting interpretation": interpret_ml(
            _mapping(stock_report.get("ml_forecasting"))
        ),
        "Backtesting interpretation": interpret_backtests(
            _mapping(stock_report.get("backtesting_results"))
        ),
        "Options interpretation": interpret_options(
            _mapping(stock_report.get("options_theoretical_analytics"))
        ),
        "Decision signal": research_signal_text(decision_signal),
        "Overall research conclusion": (
            "CONCLUSION GLOBAL DEL ANALISIS: El informe ha examinado el comportamiento historico "
            "del activo desde multiples perspectivas (rendimiento, riesgo, CAPM, VaR, Monte Carlo, "
            "ML, backtesting y opciones). "
            f"RESEARCH-ONLY QUANTITATIVE DECISION SIGNAL: {research_signal_text(decision_signal)} "
            "ADVERTENCIAS: (1) Todos los calculos usan datos sinteticos de demostracion. "
            "(2) El performance pasado no implica resultados futuros. "
            "(3) Esta es una simulacion academica, no asesoramiento financiero personalizado."
        ),
        "What should not be concluded": (
            "No debe inferirse una accion operativa, una rentabilidad futura, una proteccion "
            "asegurada frente a perdidas ni una valoracion profesional de derivados. "
            "Los datos mostrados son DEMO_SYNTHETIC_NOT_REAL_DATA. "
            "Cualquier analisis financiero real requiere: "
            "datos de mercado reales, analisis fundamental, contexto macroeconomico, "
            "perfil de riesgo personal, horizonte temporal, fiscalidad y costes de transaccion."
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


def research_signal_text(decision: dict[str, Any]) -> str:
    """Format the research-only quantitative decision signal as readable text."""

    positives = "; ".join(str(item) for item in decision.get("drivers_positive", [])[:3])
    negatives = "; ".join(str(item) for item in decision.get("drivers_negative", [])[:3])
    return (
        "Research-only quantitative decision signal: "
        f"{decision.get('signal')} (confidence={decision.get('confidence')}, "
        f"score={decision.get('score')}). "
        f"Suggested research action: {decision.get('suggested_research_action')}. "
        f"Positive drivers: {positives or 'none'}. "
        f"Negative drivers: {negatives or 'none'}. "
        f"{DISCLAIMER}"
    )


def _ratio_quality(value: float | None, name: str) -> str:
    if value is None:
        return f"{name} no es interpretable con la muestra disponible."
    if value > 1.0:
        return f"{name} fue superior a 1, indicando retorno historico elevado por unidad de riesgo."
    if value > 0.0:
        return f"{name} fue positivo, aunque debe leerse con incertidumbre estadistica."
    return f"{name} fue no positivo, senalando compensacion historica debil por volatilidad."
