"""Pure view-model builders for the local read-only UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_platform.config.settings import PlatformSettings, public_settings_dict
from quant_platform.config.simple_yaml import load_simple_yaml
from quant_platform.data.providers.factory import list_available_providers
from quant_platform.risk.profiles import load_risk_profiles
from quant_platform.ui.report_loader import (
    discover_datasets,
    discover_reports,
    load_trial_registry,
)
from quant_platform.ui.safety import safety_manifest


def build_platform_snapshot(
    settings: PlatformSettings,
    registry_dir: str | Path = "data/registry",
    report_dir: str | Path = "reports/generated",
    universe_config_path: str | Path = "configs/universe_etfs_crypto_daily.yaml",
    risk_profiles_path: str | Path = "configs/risk_profiles.yaml",
) -> dict[str, Any]:
    """Build a JSON-safe UI snapshot with no secret values and no network calls."""

    provider_rows = build_provider_rows(settings)
    datasets = discover_datasets(registry_dir)
    reports = discover_reports(report_dir)
    trials = load_trial_registry(Path(report_dir) / "strategy_trials.jsonl")
    snapshot = {
        "settings": public_settings_dict(settings),
        "providers": provider_rows,
        "provider_summary": summarize_providers(provider_rows),
        "universe": load_universe_summary(universe_config_path),
        "risk_profiles": load_risk_profile_summary(risk_profiles_path),
        "datasets": datasets,
        "reports": reports,
        "trial_count": len(trials),
        "safety": safety_manifest(),
        "conceptual_flow": conceptual_flow_steps(),
    }
    return snapshot


def build_provider_rows(settings: PlatformSettings) -> list[dict[str, Any]]:
    """Return UI provider rows with credential values reduced to labels."""

    rows = []
    for provider in list_available_providers(settings):
        configured = bool(provider["configured"])
        requires_key = bool(provider["requires_api_key"])
        if not requires_key:
            credential_state = "not_required"
        elif configured:
            credential_state = "configured"
        else:
            credential_state = "missing"
        rows.append(
            {
                **provider,
                "credential_state": credential_state,
                "read_only": True,
            }
        )
    return rows


def summarize_providers(provider_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize provider status counts for UI cards."""

    statuses: dict[str, int] = {}
    for provider in provider_rows:
        status = str(provider.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "total": len(provider_rows),
        "available": statuses.get("available", 0),
        "configured_but_disabled": statuses.get("configured_but_disabled", 0),
        "missing_api_key": statuses.get("missing_api_key", 0),
        "disabled": statuses.get("disabled", 0),
    }


def load_universe_summary(path: str | Path) -> dict[str, Any]:
    """Load a local universe config and summarize symbol groups."""

    config_path = Path(path)
    if not config_path.exists():
        return {"path": str(config_path), "exists": False, "groups": [], "total_symbols": 0}
    config = load_simple_yaml(config_path)
    groups = []
    for section_name in ("equity_universe", "crypto_universe"):
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for group_name, symbols in section.items():
            symbol_list = [str(symbol) for symbol in symbols] if isinstance(symbols, list) else []
            groups.append(
                {
                    "asset_class": section_name.replace("_universe", ""),
                    "group": str(group_name),
                    "symbol_count": len(symbol_list),
                    "symbols": symbol_list,
                }
            )
    return {
        "path": str(config_path),
        "exists": True,
        "base_currency": config.get("base_currency"),
        "frequency": config.get("frequency"),
        "groups": groups,
        "total_symbols": sum(int(group["symbol_count"]) for group in groups),
    }


def load_risk_profile_summary(path: str | Path) -> dict[str, Any]:
    """Load local risk profiles and return display-safe rows."""

    profile_path = Path(path)
    if not profile_path.exists():
        return {"path": str(profile_path), "exists": False, "profiles": []}
    profiles = load_risk_profiles(profile_path)
    rows = []
    for name, profile in profiles.items():
        rows.append(
            {
                "profile": name,
                "target_max_drawdown": float(profile["target_max_drawdown"]),
                "max_single_asset_weight": float(profile["max_single_asset_weight"]),
                "max_gross_leverage": float(profile["max_gross_leverage"]),
                "max_turnover": float(profile["max_turnover"]),
                "equity_total_bps": _total_bps(profile, "equity"),
                "crypto_total_bps": _total_bps(profile, "crypto"),
            }
        )
    return {"path": str(profile_path), "exists": True, "profiles": rows}


def dataset_quality_rows(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten dataset coverage metadata into a UI table."""

    rows = []
    for dataset in datasets:
        coverage = dataset.get("coverage_metadata", {})
        if not isinstance(coverage, dict):
            coverage = {}
        rows.append(
            {
                "dataset_id": dataset.get("dataset_id"),
                "version": dataset.get("version"),
                "row_count": dataset.get("row_count"),
                "symbols_total": len(coverage.get("symbols_successful", []))
                + len(coverage.get("symbols_failed", [])),
                "failed_checks": len(coverage.get("failed_checks", [])),
                "warnings": len(coverage.get("warnings", [])),
                "suitable_for_backtest_demo": coverage.get("suitable_for_backtest_demo"),
                "quality_report_exists": dataset.get("quality_report_exists"),
            }
        )
    return rows


def backtest_metric_rows(report: dict[str, Any]) -> list[dict[str, float | str | None]]:
    """Extract display metrics from one backtest report."""

    return [
        {
            "metric": metric,
            "label": metadata["label"],
            "value": _clean_float(report.get(metric)),
            "plain_explanation": metadata["plain_explanation"],
        }
        for metric, metadata in metric_explanations().items()
    ]


def metric_explanations() -> dict[str, dict[str, str]]:
    """Return concise explanations for backtest metrics."""

    return {
        "final_equity": {
            "label": "Final equity",
            "plain_explanation": "Capital ficticio al final de la simulacion.",
        },
        "annualized_return": {
            "label": "Annualized return",
            "plain_explanation": "Retorno historico escalado a un ano comparable.",
        },
        "annualized_volatility": {
            "label": "Annualized volatility",
            "plain_explanation": "Variabilidad historica anualizada de retornos.",
        },
        "sharpe_ratio": {
            "label": "Sharpe",
            "plain_explanation": "Retorno por unidad de volatilidad; no garantiza beneficios.",
        },
        "sortino_ratio": {
            "label": "Sortino",
            "plain_explanation": "Retorno ajustado por variabilidad negativa.",
        },
        "max_drawdown": {
            "label": "Max drawdown",
            "plain_explanation": "Peor caida historica desde un maximo previo.",
        },
        "historical_var_95": {
            "label": "VaR 95",
            "plain_explanation": "Umbral historico de perdida al 95%; no mide la cola mas alla.",
        },
        "historical_expected_shortfall_95": {
            "label": "Expected Shortfall 95",
            "plain_explanation": "Perdida media historica en la cola mala mas extrema.",
        },
        "total_turnover": {
            "label": "Turnover total",
            "plain_explanation": "Cuanto cambio la cartera durante la simulacion.",
        },
        "total_transaction_cost": {
            "label": "Transaction costs",
            "plain_explanation": "Costes historicos simulados por comision, spread y slippage.",
        },
    }


def conceptual_flow_steps() -> list[dict[str, str]]:
    """Return the conceptual research flow displayed in the UI."""

    return [
        _flow(
            "Providers",
            "Fuentes read-only de datos.",
            "OHLCV bruto por simbolo.",
            "Configured/enabled/status sin claves.",
            "Rate limits, licencias o gaps.",
            "Tabla y grafico de provider status.",
        ),
        _flow(
            "OHLCV",
            "Open, high, low, close y volumen.",
            "Datos normalizados por activo/fecha.",
            "Schema, timestamp y available_at.",
            "Precios invalidos o timestamps inconsistentes.",
            "Dataset manifest y quality report.",
        ),
        _flow(
            "Quality checks",
            "Datos OHLCV normalizados.",
            "Warnings, failed checks, cobertura.",
            "Duplicados, OHLC, gaps, missing values.",
            "Datos incompletos pueden sesgar el backtest.",
            "Seccion Calidad de datos.",
        ),
        _flow(
            "Registry",
            "Dataset validado y metadata.",
            "Version local reproducible.",
            "Manifest JSON y rutas locales.",
            "Commitear data real por accidente.",
            "Seccion Datasets.",
        ),
        _flow(
            "Returns",
            "Precios de cierre.",
            "Retornos por activo y fecha.",
            "Precios positivos, alineacion temporal.",
            "Look-ahead bias si se usa informacion futura.",
            "Seccion Formulas y Backtests.",
        ),
        _flow(
            "Weights",
            "Retornos y reglas de cartera.",
            "Pesos objetivo long-only.",
            "Suma de pesos, leverage <= 1, execution lag.",
            "Pesos extremos o no ejecutables.",
            "Backtest demo.",
        ),
        _flow(
            "Backtest",
            "Retornos, pesos y costes.",
            "Equity curve y metricas netas.",
            "Execution lag >= 1 y costes.",
            "Backtest overfitting o costes subestimados.",
            "Seccion Backtests.",
        ),
        _flow(
            "Risk metrics",
            "Equity curve y retornos netos.",
            "Volatilidad, drawdown, VaR, ES.",
            "Convenciones de perdidas positivas.",
            "Interpretar metricas como garantias.",
            "Cards y graficos de metricas.",
        ),
        _flow(
            "Reports",
            "Metricas, supuestos y trial metadata.",
            "JSON auditable local.",
            "Registro de hipotesis y parametros.",
            "Seleccionar solo la mejor curva.",
            "Reports y Comparacion.",
        ),
        _flow(
            "Visual UI",
            "Artefactos locales ignorados por Git.",
            "Vista didactica y read-only.",
            "No red automatica, no secrets, no trading.",
            "Graficos sin contexto pueden confundir.",
            "Esta app local.",
        ),
    ]


def report_series(report: dict[str, Any], *keys: str) -> object | None:
    """Return the first available time-series-like value from a report."""

    for key in keys:
        value = report.get(key)
        if value:
            return value
    metadata = report.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value:
                return value
    return None


def transaction_cost_components(report: dict[str, Any]) -> dict[str, float]:
    """Extract simple transaction cost components when a report has enough data."""

    total = _clean_float(report.get("total_transaction_cost"))
    if total is None:
        return {}
    return {"total_transaction_cost": total}


def _flow(
    block: str,
    inputs: str,
    outputs: str,
    validation: str,
    failure_mode: str,
    ui_location: str,
) -> dict[str, str]:
    return {
        "block": block,
        "inputs": inputs,
        "outputs": outputs,
        "validation": validation,
        "failure_mode": failure_mode,
        "ui_location": ui_location,
    }


def _total_bps(profile: dict[str, Any], suffix: str) -> float:
    return (
        float(profile[f"commission_bps_{suffix}"])
        + float(profile[f"spread_bps_{suffix}"])
        + float(profile[f"slippage_bps_{suffix}"])
    )


def _clean_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
