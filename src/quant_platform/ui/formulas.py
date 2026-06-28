"""Plain-language formula cards for the read-only research UI."""

from __future__ import annotations

FormulaCard = dict[str, object]


def returns_formula() -> FormulaCard:
    return _formula(
        "Retorno simple",
        r"R_t = \frac{P_t}{P_{t-1}} - 1",
        "Mide el cambio porcentual de un precio entre dos fechas.",
        "Si un precio pasa de 100 a 102, el retorno simple es 2%.",
        ["Precio actual P_t", "Precio anterior P_{t-1}"],
        "Retorno del periodo en porcentaje decimal.",
        "Returns, backtests, metricas de performance.",
        "Un retorno positivo historico no significa que el siguiente periodo sera positivo.",
        "100 -> 102 produce 0.02.",
    )


def log_returns_formula() -> FormulaCard:
    return _formula(
        "Retorno logaritmico",
        r"r_t = \log(P_t) - \log(P_{t-1})",
        "Mide el cambio de precio usando logaritmos para facilitar sumas temporales.",
        "Es util cuando se acumulan muchos periodos pequenos.",
        ["Precio actual P_t", "Precio anterior P_{t-1}"],
        "Retorno logaritmico del periodo.",
        "Analisis de retornos, volatilidad y modelos futuros.",
        "No es exactamente igual al retorno simple cuando los cambios son grandes.",
        "100 -> 102 produce aproximadamente 0.0198.",
    )


def volatility_formula() -> FormulaCard:
    return _formula(
        "Volatilidad anualizada",
        r"\sigma_{ann} = \operatorname{std}(R_t)\sqrt{A}",
        "Resume cuanta variabilidad historica tuvieron los retornos.",
        "Mas volatilidad significa mas movimiento, no necesariamente mas perdida.",
        ["Serie de retornos", "Periodos por ano A"],
        "Volatilidad anualizada.",
        "Risk profiles, Sharpe, backtest report.",
        "Baja volatilidad historica no elimina eventos extremos futuros.",
        "Con datos diarios de equity se suele usar A=252.",
    )


def sharpe_formula() -> FormulaCard:
    return _formula(
        "Sharpe Ratio",
        r"Sharpe = \frac{\mathbb{E}[R_t - R_{f,t}]}{\sigma(R_t - R_{f,t})}\sqrt{A}",
        "Mide retorno extra por unidad de volatilidad historica.",
        "Pregunta cuanto retorno se obtuvo por soportar variabilidad.",
        ["Retornos", "Retorno libre de riesgo", "Periodos por ano"],
        "Ratio sin unidades.",
        "Backtest metrics y comparacion de perfiles.",
        "No implica beneficios y puede enganar con colas gruesas o pocos datos.",
        "Sharpe 1.0 suele leerse como una unidad de retorno por unidad de volatilidad.",
    )


def sortino_formula() -> FormulaCard:
    return _formula(
        "Sortino Ratio",
        r"Sortino = \frac{\mathbb{E}[R_t - T]}{\sigma(\min(R_t - T, 0))}\sqrt{A}",
        "Parecido a Sharpe, pero penaliza principalmente la variabilidad negativa.",
        "Diferencia movimientos buenos de movimientos por debajo del umbral.",
        ["Retornos", "Umbral T", "Periodos por ano"],
        "Ratio sin unidades.",
        "Backtest metrics.",
        "Puede verse alto si hay pocos periodos negativos en una muestra pequena.",
        "Con T=0, mira downside deviation de retornos negativos.",
    )


def max_drawdown_formula() -> FormulaCard:
    return _formula(
        "Max drawdown",
        r"MDD = \min_t \left(\frac{V_t}{\max_{s \le t} V_s} - 1\right)",
        "Mide la peor caida desde un maximo historico de la curva de capital.",
        "Responde cuanto se llego a caer antes de recuperarse o terminar.",
        ["Curva de capital V_t"],
        "Caida maxima como numero negativo o magnitud de perdida.",
        "Risk report, profile comparison, backtest summary.",
        "Un drawdown pasado no limita el drawdown futuro.",
        "De 120 a 90 implica -25% desde ese maximo.",
    )


def var_formula() -> FormulaCard:
    return _formula(
        "Value at Risk historico",
        r"VaR_{\alpha}(L) = \inf\{l: P(L \le l) \ge \alpha\}",
        "Estima un umbral de perdida para un nivel de confianza.",
        "Dice: con estos datos, la perdida rara vez supero este umbral.",
        ["Perdidas positivas L", "Nivel de confianza alpha"],
        "Perdida umbral.",
        "Backtest metrics y risk report.",
        "VaR no dice cuanto se pierde si se cruza el umbral.",
        "VaR 95% mira un percentil alto de perdidas historicas.",
    )


def expected_shortfall_formula() -> FormulaCard:
    return _formula(
        "Expected Shortfall historico",
        r"ES_{\alpha}(L) = \mathbb{E}[L \mid L \ge VaR_{\alpha}(L)]",
        "Mide la perdida media dentro de la cola mala mas extrema.",
        "Complementa VaR mirando que pasa despues de cruzar el umbral.",
        ["Perdidas positivas L", "VaR al nivel alpha"],
        "Perdida media de cola.",
        "Backtest metrics y riesgo de eventos extremos.",
        "Depende mucho de la muestra historica o del modelo usado.",
        "Si las peores perdidas son 4%, 5% y 7%, ES promedia esa cola.",
    )


def turnover_formula() -> FormulaCard:
    return _formula(
        "Turnover",
        r"Turnover_t = \sum_i |w_{i,t} - w_{i,t-1}|",
        "Mide cuanto cambia la cartera entre dos rebalanceos.",
        "Mas turnover suele implicar mas costes y mas friccion operativa.",
        ["Pesos actuales", "Pesos anteriores"],
        "Cambio total de pesos.",
        "Cost model, backtest metrics.",
        "Turnover bajo no implica bajo riesgo de mercado.",
        "Cambiar de 50/50 a 60/40 produce turnover 0.20.",
    )


def transaction_cost_formula() -> FormulaCard:
    return _formula(
        "Coste de transaccion",
        r"Cost_t = Turnover_t \times (commission + spread/2 + slippage)",
        "Aproxima el coste de rebalancear por comisiones, spread y slippage.",
        "El rendimiento bruto no basta; las fricciones reducen el resultado neto.",
        ["Turnover", "Comision", "Spread", "Slippage"],
        "Coste restado al retorno o capital.",
        "Backtester vectorizado y reportes netos.",
        "Un coste promedio puede subestimar mercados iliquidos o crisis.",
        "Turnover 0.2 y coste 10 bps produce 2 bps de coste.",
    )


def portfolio_return_formula() -> FormulaCard:
    return _formula(
        "Retorno de cartera",
        r"R_{p,t} = \sum_i w_{i,t-1} R_{i,t}",
        "Combina retornos de activos usando pesos decididos antes del periodo.",
        "Evita mirar el futuro: pesos de ayer explican retorno de hoy.",
        ["Pesos por activo", "Retornos por activo"],
        "Retorno de cartera.",
        "Backtesting y curva de capital.",
        "Usar pesos calculados con informacion futura introduce look-ahead bias.",
        "60% SPY y 40% BTC combina ambos retornos con esos pesos.",
    )


def equity_curve_formula() -> FormulaCard:
    return _formula(
        "Curva de capital",
        r"V_t = V_{t-1}(1 + R_{p,t} - Cost_t)",
        "Actualiza el capital ficticio con retorno neto de costes.",
        "Es el recorrido historico simulado de la estrategia.",
        ["Capital anterior", "Retorno de cartera", "Costes"],
        "Capital simulado en cada fecha.",
        "Backtest report, drawdown, metricas finales.",
        "Una curva historica ascendente no es una prediccion asegurada.",
        "10000 con retorno neto 1% pasa a 10100.",
    )


def all_formulas() -> list[FormulaCard]:
    """Return all formula cards in UI display order."""

    return [
        returns_formula(),
        log_returns_formula(),
        portfolio_return_formula(),
        equity_curve_formula(),
        volatility_formula(),
        sharpe_formula(),
        sortino_formula(),
        max_drawdown_formula(),
        var_formula(),
        expected_shortfall_formula(),
        turnover_formula(),
        transaction_cost_formula(),
    ]


def _formula(
    name: str,
    latex: str,
    plain_explanation: str,
    intuition: str,
    inputs: list[str],
    outputs: str,
    used_in: str,
    common_misinterpretation: str,
    example_short: str,
) -> FormulaCard:
    return {
        "name": name,
        "latex": latex,
        "plain_explanation": plain_explanation,
        "intuition": intuition,
        "inputs": inputs,
        "outputs": outputs,
        "used_in": used_in,
        "common_misinterpretation": common_misinterpretation,
        "example_short": example_short,
    }
