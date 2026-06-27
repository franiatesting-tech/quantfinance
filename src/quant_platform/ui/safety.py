"""Read-only UI safety policy with no trading or secret exposure."""

from __future__ import annotations

from dataclasses import dataclass


class UISafetyError(ValueError):
    """Raised when a UI action is outside the approved read-only scope."""


@dataclass(frozen=True)
class SafetyAction:
    """Human-readable UI action policy row."""

    name: str
    status: str
    reason: str


ALLOWED_ACTIONS: tuple[SafetyAction, ...] = (
    SafetyAction(
        "inspect_settings",
        "allowed",
        "Show public settings only; API key values are never exposed.",
    ),
    SafetyAction(
        "inspect_providers",
        "allowed",
        "Show provider enabled/configured/status metadata without secrets.",
    ),
    SafetyAction(
        "inspect_local_registry",
        "allowed",
        "Read local dataset manifests and data quality JSON reports.",
    ),
    SafetyAction(
        "inspect_generated_reports",
        "allowed",
        "Read local generated backtest and comparison JSON reports.",
    ),
    SafetyAction(
        "show_safe_commands",
        "allowed",
        "Display CLI commands for explicit user execution.",
    ),
)

PROHIBITED_ACTIONS: tuple[SafetyAction, ...] = (
    SafetyAction("live_trading", "prohibited", "No live trading path is in scope."),
    SafetyAction("paper_trading", "prohibited", "Paper trading is intentionally absent."),
    SafetyAction("broker_orders", "prohibited", "No broker, order, or account endpoint use."),
    SafetyAction(
        "private_exchange_endpoints",
        "prohibited",
        "No private exchange keys or account calls.",
    ),
    SafetyAction(
        "margin_futures_perpetuals",
        "prohibited",
        "No margin, futures, perpetuals, or leverage.",
    ),
    SafetyAction(
        "network_auto_run",
        "prohibited",
        "The UI must not start network data downloads automatically.",
    ),
    SafetyAction("secret_display", "prohibited", "API key values must never be printed or cached."),
)

SECRET_MARKERS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "private_key",
)

TRADING_MARKERS = (
    "order",
    "broker",
    "account",
    "live_trading",
    "paper_trading",
    "margin",
    "future",
    "perpetual",
    "leverage",
)


def safety_manifest() -> dict[str, list[dict[str, str]]]:
    """Return the public UI safety policy."""

    return {
        "allowed_actions": [action.__dict__ for action in ALLOWED_ACTIONS],
        "prohibited_actions": [action.__dict__ for action in PROHIBITED_ACTIONS],
    }


def assert_read_only_action(action_name: str) -> None:
    """Reject action names that imply trading, private endpoints, or secret handling."""

    clean_name = action_name.strip().lower()
    if any(marker in clean_name for marker in TRADING_MARKERS):
        raise UISafetyError(f"UI action is prohibited: {action_name}")


def redact_sensitive_text(value: object) -> str:
    """Return a conservative text representation with credential markers redacted."""

    text = str(value)
    lower = text.lower()
    if any(marker in lower for marker in SECRET_MARKERS):
        return "[redacted_sensitive_text]"
    return text


def assert_no_sensitive_keys(payload: object) -> None:
    """Fail if a nested payload contains key names that commonly hold secrets."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            if any(marker in str(key).lower() for marker in SECRET_MARKERS):
                raise UISafetyError(f"Sensitive key is not allowed in UI payload: {key}")
            assert_no_sensitive_keys(value)
    elif isinstance(payload, list | tuple):
        for item in payload:
            assert_no_sensitive_keys(item)
