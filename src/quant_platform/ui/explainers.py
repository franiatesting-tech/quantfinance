"""Non-technical concept explainers for the local read-only UI."""

from __future__ import annotations

Explainer = dict[str, object]


def platform_explainer() -> Explainer:
    return _explainer(
        "Que es la plataforma",
        "Una herramienta local para investigar estrategias con datos historicos y dinero ficticio.",
        "Centraliza datos, calidad, backtests y reportes auditables.",
        "Como un laboratorio: observa y mide, pero no opera en mercado.",
        "Modo research-only; no existe ruta de ordenes ni broker.",
        "No es asesoramiento financiero ni predice resultados futuros.",
    )


def provider_explainer() -> Explainer:
    return _explainer(
        "Que es un provider",
        "Una fuente de datos, por ejemplo yfinance, Binance public o Polygon.",
        "Sin datos confiables no hay backtest confiable.",
        "Como una biblioteca de precios: algunas son publicas y otras requieren credencial.",
        "La UI muestra configured/enabled/status, nunca valores de claves.",
        "Cada proveedor tiene limites, licencias y posibles huecos de cobertura.",
    )


def ohlcv_explainer() -> Explainer:
    return _explainer(
        "Que es OHLCV",
        "Open, High, Low, Close y Volume para cada activo y fecha.",
        "Es la materia prima de retornos, calidad y backtests.",
        "Como una ficha diaria con apertura, maximo, minimo, cierre y volumen.",
        "El schema tambien registra timestamp, available_at, source y venue.",
        "OHLCV historico puede contener errores, ajustes o gaps.",
    )


def dataset_explainer() -> Explainer:
    return _explainer(
        "Que es un dataset",
        "Una version local de datos OHLCV registrada con manifest y metadata.",
        "Permite repetir un analisis sabiendo que datos se usaron.",
        "Como congelar una foto de datos para auditarla despues.",
        "Se guarda bajo data/registry y esta ignorado por Git.",
        "Un dataset puede ser incompleto o sesgado aunque este bien registrado.",
    )


def quality_report_explainer() -> Explainer:
    return _explainer(
        "Que es un quality report",
        "Un resumen de cobertura, gaps, warnings y checks fallidos.",
        "Evita usar datos con problemas invisibles.",
        "Como una inspeccion tecnica antes de conducir un coche.",
        "Se genera junto al manifest local del dataset.",
        "Pasar checks no convierte datos gratuitos en datos profesionales.",
    )


def backtest_explainer() -> Explainer:
    return _explainer(
        "Que es un backtest",
        "Una simulacion historica con reglas fijas y capital ficticio.",
        "Sirve para medir comportamiento pasado neto de costes basicos.",
        "Como ensayar una estrategia con una grabacion del pasado.",
        "El motor actual es vectorizado offline con execution_lag >= 1.",
        "Un backtest no es una prediccion asegurada ni una recomendacion.",
    )


def equity_curve_explainer() -> Explainer:
    return _explainer(
        "Que es una equity curve",
        "La evolucion del capital ficticio durante un backtest.",
        "Permite ver crecimiento, caidas y recuperaciones.",
        "Como el marcador historico de una simulacion.",
        "Se calcula acumulando retornos netos de costes.",
        "Una curva suave puede ocultar riesgos de cola o sesgo de datos.",
    )


def drawdown_explainer() -> Explainer:
    return _explainer(
        "Que es drawdown",
        "La caida desde un maximo previo de la curva de capital.",
        "Muestra cuanto dolor historico tuvo una estrategia antes de recuperarse.",
        "Como medir cuanto bajo el nivel del agua desde su marca mas alta.",
        "Max drawdown toma la peor caida de toda la muestra.",
        "El drawdown futuro puede ser peor que el historico.",
    )


def risk_profile_explainer() -> Explainer:
    return _explainer(
        "Conservative vs aggressive",
        "Son perfiles de evaluacion con distintos umbrales y costes.",
        "Ayudan a comparar comportamiento bajo reglas de riesgo distintas.",
        "Como probar un coche en modo ahorro y modo deportivo.",
        "15% y 30% son targets de drawdown, no restricciones aseguradas.",
        "Mas agresivo no significa mayor retorno ni menor riesgo.",
    )


def configured_enabled_explainer() -> Explainer:
    return _explainer(
        "Configured vs enabled",
        "Configured significa que hay credencial local; enabled significa que se permite usar.",
        "Separa seguridad de disponibilidad: tener clave no activa automaticamente el proveedor.",
        "Como tener una llave guardada pero la puerta aun cerrada por politica.",
        "configured_but_disabled es valido y seguro.",
        "La UI nunca muestra la credencial ni sus ultimos caracteres.",
    )


def read_only_explainer() -> Explainer:
    return _explainer(
        "Que significa read-only",
        "La plataforma puede leer datos, pero no operar ni enviar instrucciones financieras.",
        "Reduce superficie de riesgo mientras se audita la investigacion.",
        "Como mirar una pantalla informativa sin botones de compra o venta.",
        "Providers usan endpoints publicos/read-only o keyed read-only.",
        "Read-only no elimina riesgos de calidad, licencia o interpretacion.",
    )


def prohibited_actions_explainer() -> Explainer:
    return _explainer(
        "Acciones prohibidas",
        "Trading, paper trading, ordenes, brokers, endpoints privados y mostrar claves.",
        "Evita perdidas reales, fugas de secretos y scope creep.",
        "Como quitar el motor de ejecucion de un simulador de investigacion.",
        "La UI solo muestra comandos y artefactos; no ejecuta acciones peligrosas.",
        "Si aparece una clave en diff o logs, se debe parar y rotar.",
    )


def backtest_not_prediction_explainer() -> Explainer:
    return _explainer(
        "Backtest no es prediccion",
        "Un resultado historico no implica resultados futuros.",
        "Mercados cambian, datos pueden tener sesgos y costes reales pueden variar.",
        "Como aprobar un examen antiguo: ayuda, pero no asegura aprobar el proximo.",
        "Se requieren splits temporales, benchmarks, trials y stress testing para mejorar rigor.",
        "No usar metricas historicas como recomendacion financiera.",
    )


def overfitting_trials_explainer() -> Explainer:
    return _explainer(
        "Por que registrar trials",
        "Cada prueba de estrategia debe quedar registrada con hipotesis y parametros.",
        "Reduce la tentacion de elegir solo la mejor curva historica.",
        "Como anotar todos los experimentos de laboratorio, no solo los exitosos.",
        "El registry JSONL es append-only para auditoria local.",
        "Registrar trials ayuda, pero no elimina por completo el riesgo de overfitting.",
    )


def all_explainers() -> list[Explainer]:
    """Return all concept explainers in UI display order."""

    return [
        platform_explainer(),
        provider_explainer(),
        ohlcv_explainer(),
        dataset_explainer(),
        quality_report_explainer(),
        backtest_explainer(),
        equity_curve_explainer(),
        drawdown_explainer(),
        risk_profile_explainer(),
        configured_enabled_explainer(),
        read_only_explainer(),
        prohibited_actions_explainer(),
        backtest_not_prediction_explainer(),
        overfitting_trials_explainer(),
    ]


def _explainer(
    title: str,
    summary: str,
    why_it_matters: str,
    analogy: str,
    technical_note: str,
    risk_warning: str,
) -> Explainer:
    return {
        "title": title,
        "summary": summary,
        "why_it_matters": why_it_matters,
        "analogy": analogy,
        "technical_note": technical_note,
        "risk_warning": risk_warning,
    }
