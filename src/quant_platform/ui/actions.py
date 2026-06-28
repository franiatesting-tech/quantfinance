"""CLI and UI action helpers for the local read-only interface."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from quant_platform.config.settings import PlatformSettings
from quant_platform.ui.safety import assert_read_only_action
from quant_platform.ui.view_models import build_platform_snapshot


def build_ui_status(
    settings: PlatformSettings,
    registry_dir: str | Path = "data/registry",
    report_dir: str | Path = "reports/generated",
    universe_config_path: str | Path = "configs/universe_etfs_crypto_daily.yaml",
    risk_profiles_path: str | Path = "configs/risk_profiles.yaml",
) -> dict[str, Any]:
    """Return a read-only UI readiness summary with no secret values."""

    assert_read_only_action("inspect_local_ui_status")
    snapshot = build_platform_snapshot(
        settings=settings,
        registry_dir=registry_dir,
        report_dir=report_dir,
        universe_config_path=universe_config_path,
        risk_profiles_path=risk_profiles_path,
    )
    app_path = Path(__file__).with_name("app.py")
    streamlit_installed = importlib.util.find_spec("streamlit") is not None
    plotly_installed = importlib.util.find_spec("plotly") is not None
    launch_command = ["py", "-3", "-m", "quant_platform.cli", "launch-ui"]
    return {
        "ui_available": app_path.exists() and streamlit_installed and plotly_installed,
        "ui_ready": app_path.exists() and streamlit_installed and plotly_installed,
        "streamlit_installed": streamlit_installed,
        "plotly_installed": plotly_installed,
        "app_path": str(app_path),
        "read_only": True,
        "network_auto_run": False,
        "provider_summary": snapshot["provider_summary"],
        "dataset_count": len(snapshot["datasets"]),
        "report_count": len(snapshot["reports"]),
        "trial_count": snapshot["trial_count"],
        "universe_total_symbols": snapshot["universe"]["total_symbols"],
        "risk_profile_count": len(snapshot["risk_profiles"]["profiles"]),
        "no_secrets_exposed": True,
        "launch_command": launch_command,
    }


def launch_ui_command(
    host: str = "localhost",
    port: int = 8501,
    show_browser: bool = True,
) -> list[str]:
    """Build the Streamlit launch command without importing Streamlit."""

    if port <= 0:
        raise ValueError("port must be positive")
    app_path = Path(__file__).with_name("app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    if not show_browser:
        command.extend(["--server.headless", "true"])
    return command


def action_catalog() -> dict[str, list[dict[str, str]]]:
    """Return allowed and prohibited actions for display in the UI."""

    allowed = [
        _allowed(
            "Ver estado",
            "Inspeccionar si la UI, providers, datasets y reports locales estan disponibles.",
            "Es lectura local de metadata publica.",
            "Puede mostrar que faltan datos, pero no ejecuta nada.",
            "py -3 -m quant_platform.cli ui-status",
            "Salida JSON en terminal.",
        ),
        _allowed(
            "Ver providers",
            "Listar providers, flags enabled/configured y estado read-only.",
            "Permite entender que fuente de datos esta lista sin exponer claves.",
            "Un provider configured_but_disabled no se usa hasta cambiar flags locales.",
            "py -3 -m quant_platform.cli list-providers",
            "Salida JSON en terminal.",
        ),
        _allowed(
            "Validar providers sin red",
            "Revisar configuracion offline de providers.",
            "No llama APIs externas y no usa datos privados.",
            "No confirma cuotas ni disponibilidad real del proveedor.",
            "py -3 -m quant_platform.cli validate-providers",
            "Salida JSON en terminal.",
        ),
        _allowed(
            "Network smoke explicito",
            "Probar providers keyed solo cuando el usuario lo ejecuta en terminal.",
            "Sirve para diagnosticar conectividad y permisos read-only.",
            "Puede consumir cuota o fallar por limites del proveedor.",
            "py -3 -m quant_platform.cli validate-providers --network-smoke",
            "Salida JSON sanitizada en terminal.",
        ),
        _allowed(
            "Descargar datos read-only",
            "Crear datasets locales desde providers publicos o keyed read-only.",
            "Los datos quedan versionados en registry local ignorado por Git.",
            "No implica que los datos sean completos, licenciados para produccion o sin sesgo.",
            (
                "py -3 -m quant_platform.cli download-real-data --config "
                "configs/universe_etfs_crypto_daily.yaml --start 2024-01-01 --end 2024-03-31 "
                "--limit-equity 3 --limit-crypto 2"
            ),
            "data/registry/<dataset>/<version>/",
        ),
        _allowed(
            "Ver datasets",
            "Inspeccionar manifests y cobertura de datasets locales.",
            "Ayuda a saber que simbolos y rangos temporales existen.",
            "No valida terminos de uso ni survivorship bias por si solo.",
            "py -3 -m quant_platform.cli ui-status",
            "data/registry/",
        ),
        _allowed(
            "Ver quality reports",
            "Leer reportes locales de calidad de datos.",
            "Muestra gaps, fallos, warnings y aptitud para demos.",
            "Un reporte sin fallos no implica datos profesionales.",
            "py -3 -m quant_platform.cli ui-status",
            "data/registry/<dataset>/<version>/data_quality_report.json",
        ),
        _allowed(
            "Ejecutar backtest demo",
            "Correr un backtest offline con capital ficticio y datos locales.",
            "Sirve para auditar metricas y costes antes de modelos complejos.",
            "Un backtest no predice resultados futuros.",
            (
                "py -3 -m quant_platform.cli run-backtest-demo --dataset-id real_daily_demo "
                "--version v1 --profile conservative"
            ),
            "reports/generated/",
        ),
        _allowed(
            "Comparar perfiles",
            "Comparar conservative vs aggressive sobre el mismo dataset.",
            "Ayuda a separar retorno historico, volatilidad, drawdown y costes.",
            "Los objetivos de drawdown son umbrales, no garantias.",
            (
                "py -3 -m quant_platform.cli compare-profiles --dataset-id real_daily_demo "
                "--version v1 --config configs/universe_etfs_crypto_daily.yaml"
            ),
            "reports/generated/*_profile_comparison.json",
        ),
        _allowed(
            "Ver graficos",
            "Abrir la UI local para explorar charts y explicaciones.",
            "Es una capa visual read-only sobre artefactos locales.",
            "Graficos sin contexto pueden inducir conclusiones incorrectas.",
            "py -3 -m quant_platform.cli launch-ui --host localhost --port 8501",
            "http://localhost:8501",
        ),
        _allowed(
            "Exportar reports locales",
            "Usar los JSON generados localmente para auditoria fuera de Git.",
            "Mantiene trazabilidad de resultados y supuestos.",
            "No subir reports reales sin revisar licencias y privacidad.",
            "No automatic command; copy only approved ignored artifacts if needed.",
            "reports/generated/",
        ),
    ]
    prohibited = [
        _prohibited("Trading real", "Ejecutar operaciones con dinero real.", "Fuera de alcance."),
        _prohibited(
            "Paper trading",
            "Simular ordenes conectadas a broker/exchange.",
            "No implementado.",
        ),
        _prohibited("Enviar ordenes", "Crear, modificar o cancelar ordenes.", "Riesgo operativo."),
        _prohibited("Conectar broker", "Usar credenciales o APIs de broker.", "Riesgo financiero."),
        _prohibited(
            "Endpoints privados",
            "Usar account/order/private endpoints.",
            "Riesgo de secrets.",
        ),
        _prohibited("Mostrar claves", "Imprimir o revelar valores de API keys.", "Riesgo critico."),
        _prohibited("Subir .env", "Commit o upload de `.env`/`.env.*`.", "Riesgo critico."),
        _prohibited("Leverage real", "Usar apalancamiento real.", "Fuera de alcance."),
        _prohibited("Margin", "Operar con margen.", "Fuera de alcance."),
        _prohibited("Futures", "Operar futuros.", "Fuera de alcance."),
        _prohibited("Perpetuals", "Operar perpetuos.", "Fuera de alcance."),
        _prohibited("Funding real", "Usar pagos reales de funding.", "Fuera de alcance."),
    ]
    return {"allowed_actions": allowed, "prohibited_actions": prohibited}


def safe_command_catalog() -> list[dict[str, str]]:
    """Return safe commands displayed by the UI; execution stays explicit."""

    return [row for row in action_catalog()["allowed_actions"] if row["command"]]


def _allowed(
    label: str,
    description: str,
    reason: str,
    risk: str,
    command: str,
    output_location: str,
) -> dict[str, str]:
    return {
        "label": label,
        "description": description,
        "reason": reason,
        "risk": risk,
        "command": command,
        "output_location": output_location,
    }


def _prohibited(label: str, description: str, risk: str) -> dict[str, str]:
    return {
        "label": label,
        "description": description,
        "reason": "Prohibido por el alcance research-only de la plataforma.",
        "risk": risk,
        "command": "",
        "output_location": "N/A",
    }
