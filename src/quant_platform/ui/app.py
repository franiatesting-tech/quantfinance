"""Streamlit application for local read-only research inspection."""

from __future__ import annotations

import json
from typing import Any

from quant_platform.config.settings import load_settings_from_env
from quant_platform.ui.actions import action_catalog, build_ui_status, safe_command_catalog
from quant_platform.ui.charts import (
    backtest_metric_figure,
    data_quality_warnings_figure,
    drawdown_curve_figure,
    equity_curve_figure,
    profile_comparison_figure,
    provider_status_figure,
    quality_coverage_figure,
    returns_histogram_figure,
    transaction_cost_breakdown_figure,
    universe_mix_figure,
    var_es_conceptual_figure,
)
from quant_platform.ui.explainers import all_explainers
from quant_platform.ui.formulas import all_formulas
from quant_platform.ui.report_loader import load_dataset_quality_report, read_json_report
from quant_platform.ui.view_models import (
    backtest_metric_rows,
    build_platform_snapshot,
    dataset_quality_rows,
    report_series,
    transaction_cost_components,
)

SECTIONS = (
    "Inicio",
    "Flujo conceptual",
    "Estado del sistema",
    "Providers",
    "Universo",
    "Datasets",
    "Calidad de datos",
    "Backtests",
    "Comparacion de perfiles",
    "Formulas",
    "Acciones permitidas",
    "Riesgos y limites",
)


def main() -> None:
    """Run the Streamlit UI without starting data downloads, backtests, or trading."""

    import pandas as pd
    import streamlit as st

    st.set_page_config(
        page_title="Quant Platform Research Console",
        page_icon="QP",
        layout="wide",
    )
    _inject_style(st)

    st.sidebar.title("Quant Research Console")
    st.sidebar.markdown("**Research-only / No trading**")
    section = st.sidebar.radio("Navegacion", SECTIONS, label_visibility="collapsed")
    st.sidebar.divider()
    registry_dir = st.sidebar.text_input("Registry dir", value="data/registry")
    report_dir = st.sidebar.text_input("Report dir", value="reports/generated")
    universe_config = st.sidebar.text_input(
        "Universe config",
        value="configs/universe_etfs_crypto_daily.yaml",
    )
    risk_profiles = st.sidebar.text_input("Risk profiles", value="configs/risk_profiles.yaml")
    st.sidebar.caption("No automatic network calls. No orders. No secrets displayed.")

    settings = load_settings_from_env()
    snapshot = build_platform_snapshot(
        settings=settings,
        registry_dir=registry_dir,
        report_dir=report_dir,
        universe_config_path=universe_config,
        risk_profiles_path=risk_profiles,
    )
    status = build_ui_status(
        settings=settings,
        registry_dir=registry_dir,
        report_dir=report_dir,
        universe_config_path=universe_config,
        risk_profiles_path=risk_profiles,
    )

    _render_header(st)
    _render_safety_strip(st)

    routes = {
        "Inicio": _render_home,
        "Flujo conceptual": _render_conceptual_flow,
        "Estado del sistema": _render_system_status,
        "Providers": _render_providers,
        "Universo": _render_universe,
        "Datasets": _render_datasets,
        "Calidad de datos": _render_quality,
        "Backtests": _render_backtests,
        "Comparacion de perfiles": _render_profile_comparison,
        "Formulas": _render_formulas,
        "Acciones permitidas": _render_actions,
        "Riesgos y limites": _render_risks,
    }
    routes[section](st, pd, snapshot, status)


def _render_header(st) -> None:  # noqa: ANN001
    st.markdown(
        """
        <div class="hero">
          <div>
            <p class="eyebrow">Local Quant Finance Research</p>
            <h1>Quant Platform Research Console</h1>
            <p class="hero-copy">
              Esta aplicacion sirve para investigar estrategias con datos historicos y dinero
              ficticio. No compra ni vende nada.
            </p>
          </div>
          <div class="hero-badge">Research-only<br/>No trading</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_safety_strip(st) -> None:  # noqa: ANN001
    cols = st.columns(3)
    cols[0].success("Research-only / No trading")
    cols[1].info("No se muestran claves")
    cols[2].warning("Backtest no es prediccion garantizada")


def _render_home(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ANN001
    settings = snapshot["settings"]
    provider_summary = snapshot["provider_summary"]
    universe = snapshot["universe"]
    capital = settings["safety"]["initial_capital"]
    frequency = settings["research"]["default_frequency"]

    st.subheader("Inicio")
    _kpi_row(
        st,
        [
            ("Modo", "Research-only"),
            ("Capital simulado", _money(capital, settings["safety"]["base_currency"])),
            ("Frecuencia", str(frequency)),
            ("Providers listos", str(provider_summary["available"])),
            ("Datasets locales", str(len(snapshot["datasets"]))),
            ("Reports locales", str(len(snapshot["reports"]))),
        ],
    )
    st.markdown(
        """
        Esta consola muestra el estado de la plataforma, datos locales, calidad, backtests y
        formulas. Es una interfaz de inspeccion: no lanza ordenes, no hace trading y no ejecuta
        descargas de red automaticamente.
        """
    )
    if any(_is_demo_artifact(item) for item in [*snapshot["datasets"], *snapshot["reports"]]):
        st.warning("Se detectaron artefactos demo marcados como DEMO_SYNTHETIC_NOT_REAL_DATA.")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(provider_status_figure(snapshot["providers"]), use_container_width=True)
    with right:
        st.plotly_chart(universe_mix_figure(universe), use_container_width=True)

    st.markdown("### Comandos seguros desde terminal")
    st.caption("La UI no ejecuta estos comandos por ti; se muestran para ejecucion explicita.")
    st.dataframe(_safe_df(pd, safe_command_catalog()), use_container_width=True, hide_index=True)
    with st.expander("Estado UI JSON publico"):
        st.json(status)


def _render_conceptual_flow(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Flujo conceptual")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", color="#d6b35a"];
          Providers -> OHLCV -> "Quality checks" -> Registry -> Returns;
          Returns -> Weights -> Backtest -> "Risk metrics" -> Reports -> "Visual UI";
        }
        """
    )
    st.dataframe(
        _safe_df(pd, snapshot["conceptual_flow"]),
        use_container_width=True,
        hide_index=True,
    )
    _concept_cards(st)


def _render_system_status(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ANN001
    st.subheader("Estado del sistema")
    _kpi_row(
        st,
        [
            ("UI available", str(status["ui_available"])),
            ("Streamlit", str(status["streamlit_installed"])),
            ("Plotly", str(status["plotly_installed"])),
            ("Network auto-run", str(status["network_auto_run"])),
            ("No secrets", str(status["no_secrets_exposed"])),
        ],
    )
    st.dataframe(_safe_df(pd, [status]), use_container_width=True, hide_index=True)
    with st.expander("Public settings snapshot"):
        st.json(snapshot["settings"])


def _render_providers(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Providers")
    st.write(
        "Configured significa que existe credencial local. Enabled significa que la politica "
        "local permite usar el provider. configured_but_disabled es seguro: la clave existe, "
        "pero el provider esta apagado."
    )
    st.plotly_chart(provider_status_figure(snapshot["providers"]), use_container_width=True)
    st.dataframe(_safe_df(pd, snapshot["providers"]), use_container_width=True, hide_index=True)
    with st.expander("Como leer los estados"):
        st.markdown(
            """
            - `available`: listo para uso read-only.
            - `configured_but_disabled`: hay credencial local, pero el flag esta apagado.
            - `missing_api_key`: se quiere usar, pero falta credencial.
            - `disabled`: apagado por configuracion.

            La UI nunca muestra valores de claves ni ultimos caracteres.
            """
        )


def _render_universe(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Universo")
    universe = snapshot["universe"]
    st.write(
        "El universo de investigacion combina ETFs liquidos, watchlists de equity e indices con "
        "crypto majors spot. No es una recomendacion de inversion."
    )
    st.plotly_chart(universe_mix_figure(universe), use_container_width=True)
    st.dataframe(_safe_df(pd, universe["groups"]), use_container_width=True, hide_index=True)
    st.info(
        "ETFs liquidos ayudan a empezar con datos mas estables. Crypto spot es mas volatil y "
        "opera 24/7, por eso sus gaps se interpretan distinto."
    )


def _render_datasets(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Datasets")
    datasets = snapshot["datasets"]
    if not datasets:
        _empty_state(st, "No hay manifests locales.", _dataset_commands())
        return
    st.dataframe(
        _safe_df(pd, dataset_quality_rows(datasets)),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(_safe_df(pd, datasets), use_container_width=True, hide_index=True)
    with st.expander("Rutas locales"):
        st.write("Los datasets viven bajo `data/registry/` y estan ignorados por Git.")


def _render_quality(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Calidad de datos")
    datasets = snapshot["datasets"]
    if not datasets:
        _empty_state(st, "No hay datasets para revisar calidad.", _dataset_commands())
        return
    labels = [f"{row['dataset_id']} / {row['version']}" for row in datasets]
    selected = st.selectbox("Dataset", labels)
    dataset = datasets[labels.index(selected)]
    report = load_dataset_quality_report(dataset)
    if report is None:
        st.warning("Este dataset no tiene data_quality_report.json local.")
        return
    cols = st.columns(4)
    cols[0].metric("Symbols total", report.get("symbols_total", 0))
    cols[1].metric("Successful", len(report.get("symbols_successful", [])))
    cols[2].metric("Failed", len(report.get("symbols_failed", [])))
    cols[3].metric("Suitable demo", str(report.get("suitable_for_backtest_demo")))
    st.plotly_chart(quality_coverage_figure(report), use_container_width=True)
    st.plotly_chart(data_quality_warnings_figure(report), use_container_width=True)
    st.dataframe(
        _safe_df(pd, report.get("coverage_by_asset", [])),
        use_container_width=True,
        hide_index=True,
    )
    if report.get("date_gaps"):
        st.markdown("### Gaps")
        st.dataframe(_safe_df(pd, report["date_gaps"]), use_container_width=True, hide_index=True)
    with st.expander("Quality report JSON"):
        st.json(report)


def _render_backtests(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Backtests")
    reports = [row for row in snapshot["reports"] if row.get("report_type") == "backtest"]
    if not reports:
        _empty_state(st, "No hay backtest reports locales.", _backtest_commands())
        return
    report_row = _select_report(st, reports, "Backtest report")
    report = read_json_report(report_row["path"])
    metric_rows = backtest_metric_rows(report)
    cols = st.columns(5)
    for col, key in zip(
        cols,
        (
            "final_equity",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ),
        strict=False,
    ):
        row = next(item for item in metric_rows if item["metric"] == key)
        col.metric(row["label"], _format_metric(row["value"]))
    st.plotly_chart(backtest_metric_figure(report), use_container_width=True)
    st.dataframe(_safe_df(pd, metric_rows), use_container_width=True, hide_index=True)

    equity_series = report_series(report, "equity_curve")
    drawdown_series = report_series(report, "drawdown_curve")
    returns = report_series(report, "net_returns", "returns")
    if equity_series or drawdown_series or returns:
        st.plotly_chart(equity_curve_figure(equity_series), use_container_width=True)
        st.plotly_chart(drawdown_curve_figure(drawdown_series), use_container_width=True)
        st.plotly_chart(returns_histogram_figure(returns), use_container_width=True)
    else:
        st.info(
            "Este report contiene metricas agregadas, pero no series temporales. Las curvas "
            "requieren guardar equity_curve/net_returns en un pipeline posterior."
        )
    st.plotly_chart(
        transaction_cost_breakdown_figure(transaction_cost_components(report)),
        use_container_width=True,
    )
    st.plotly_chart(
        var_es_conceptual_figure(
            report.get("historical_var_95"),
            report.get("historical_expected_shortfall_95"),
        ),
        use_container_width=True,
    )
    with st.expander("Backtest report JSON"):
        st.json(report)


def _render_profile_comparison(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Comparacion de perfiles")
    reports = [row for row in snapshot["reports"] if row.get("report_type") == "profile_comparison"]
    if not reports:
        _empty_state(st, "No hay profile comparison report local.", _comparison_commands())
        return
    report_row = _select_report(st, reports, "Profile comparison")
    report = read_json_report(report_row["path"])
    st.write(
        "Conservative y aggressive son perfiles de evaluacion. Sus targets de 15% y 30% "
        "son umbrales, no garantias. Una estrategia agresiva puede no ganar mas."
    )
    st.plotly_chart(profile_comparison_figure(report), use_container_width=True)
    st.dataframe(
        _safe_df(pd, report.get("comparison_table", [])),
        use_container_width=True,
        hide_index=True,
    )
    if report.get("warnings"):
        st.warning(" | ".join(str(item) for item in report["warnings"]))
    with st.expander("Profile comparison JSON"):
        st.json(report)


def _render_formulas(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Formulas")
    st.write("Formulas usadas o preparadas para entender retornos, riesgo y backtests.")
    for formula in all_formulas():
        with st.expander(formula["name"], expanded=False):
            st.latex(formula["latex"])
            st.write(formula["plain_explanation"])
            st.info(formula["intuition"])
            st.markdown(f"**Inputs:** {', '.join(formula['inputs'])}")
            st.markdown(f"**Output:** {formula['outputs']}")
            st.markdown(f"**Se usa en:** {formula['used_in']}")
            st.warning(f"Mala interpretacion comun: {formula['common_misinterpretation']}")
            st.caption(formula["example_short"])


def _render_actions(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Acciones permitidas")
    catalog = action_catalog()
    st.markdown("### Permitidas")
    st.dataframe(
        _safe_df(pd, catalog["allowed_actions"]),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### Prohibidas")
    st.dataframe(
        _safe_df(pd, catalog["prohibited_actions"]),
        use_container_width=True,
        hide_index=True,
    )
    st.error("La UI no ejecuta trading, paper trading, ordenes, brokers ni endpoints privados.")


def _render_risks(st, pd, snapshot: dict[str, Any], status: dict[str, Any]) -> None:  # noqa: ARG001, ANN001
    st.subheader("Riesgos y limites")
    st.warning("No es asesoramiento financiero. Backtest no garantiza resultados futuros.")
    st.info("Datos demo, si existen, se marcan como DEMO_SYNTHETIC_NOT_REAL_DATA.")
    st.dataframe(_safe_df(pd, all_explainers()), use_container_width=True, hide_index=True)
    with st.expander("Mensajes clave"):
        st.markdown(
            """
            - Graficos sin contexto pueden inducir conclusiones incorrectas.
            - Datos reales o generados viven bajo rutas ignoradas y no deben commitearse.
            - Providers gratuitos pueden tener limites, licencias, gaps y revisiones.
            - Trials se registran para reducir overfitting, no para eliminarlo por completo.
            """
        )


def _concept_cards(st) -> None:  # noqa: ANN001
    explainers = all_explainers()[:6]
    cols = st.columns(3)
    for index, explainer in enumerate(explainers):
        with cols[index % 3]:
            st.markdown(f"### {explainer['title']}")
            st.write(explainer["summary"])
            st.caption(explainer["why_it_matters"])


def _kpi_row(st, items: list[tuple[str, str]]) -> None:  # noqa: ANN001
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items, strict=True):
        col.metric(label, value)


def _select_report(st, reports: list[dict[str, Any]], label: str) -> dict[str, Any]:  # noqa: ANN001
    labels = [f"{row['report_type']} | {row['name']}" for row in reports]
    selected = st.selectbox(label, labels)
    return reports[labels.index(selected)]


def _empty_state(st, message: str, commands: list[str]) -> None:  # noqa: ANN001
    st.warning(message)
    st.write("Puedes generar artefactos locales con comandos explicitos desde terminal:")
    for command in commands:
        st.code(command, language="powershell")


def _dataset_commands() -> list[str]:
    return [
        (
            "py -3 -m quant_platform.cli download-real-data --config "
            "configs/universe_etfs_crypto_daily.yaml --start 2024-01-01 --end 2024-03-31 "
            "--limit-equity 3 --limit-crypto 2 --dataset-id real_daily_demo --version v1"
        )
    ]


def _backtest_commands() -> list[str]:
    return [
        "py -3 -m quant_platform.cli run-backtest-demo --dataset-id real_daily_demo --version v1",
    ]


def _comparison_commands() -> list[str]:
    return [
        (
            "py -3 -m quant_platform.cli compare-profiles --dataset-id real_daily_demo "
            "--version v1 --config configs/universe_etfs_crypto_daily.yaml"
        )
    ]


def _safe_df(pd, rows: object):  # noqa: ANN001, ANN202
    text = json.loads(json.dumps(rows, default=str))
    if isinstance(text, dict):
        text = [text]
    normalized = []
    for row in text if isinstance(text, list) else []:
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                key: ", ".join(str(item) for item in value)
                if isinstance(value, list)
                else json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in row.items()
            }
        )
    return pd.DataFrame(normalized)


def _format_metric(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        clean = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(clean) < 1:
        return f"{clean:.2%}"
    return f"{clean:,.2f}"


def _money(value: object, currency: str) -> str:
    try:
        return f"{float(value):,.0f} {currency}"
    except (TypeError, ValueError):
        return f"{value} {currency}"


def _is_demo_artifact(value: object) -> bool:
    return "DEMO_SYNTHETIC_NOT_REAL_DATA" in json.dumps(value, default=str)


def _inject_style(st) -> None:  # noqa: ANN001
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(214, 179, 90, 0.18), transparent 28rem),
                radial-gradient(circle at 85% 15%, rgba(61, 214, 198, 0.14), transparent 30rem),
                linear-gradient(135deg, #091014 0%, #111820 48%, #15110b 100%);
            color: #e8ecef;
        }
        .hero {
            display: flex;
            justify-content: space-between;
            gap: 2rem;
            padding: 2rem;
            border: 1px solid rgba(214, 179, 90, 0.28);
            border-radius: 28px;
            background: rgba(8, 13, 18, 0.78);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #d6b35a;
            font-size: 0.78rem;
            margin-bottom: 0.25rem;
        }
        .hero h1 { margin: 0; font-size: 2.4rem; }
        .hero-copy { max-width: 52rem; color: #c8d0d8; font-size: 1.05rem; }
        .hero-badge {
            min-width: 170px;
            align-self: center;
            padding: 1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #d6b35a, #3dd6c6);
            color: #071014;
            font-weight: 800;
            text-align: center;
        }
        [data-testid="stMetric"] {
            background: rgba(12, 18, 24, 0.78);
            border: 1px solid rgba(214, 179, 90, 0.28);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1117 0%, #111820 100%);
        }
        .stDataFrame { border-radius: 18px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
