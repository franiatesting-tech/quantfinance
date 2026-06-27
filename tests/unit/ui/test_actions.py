from __future__ import annotations

import json

from quant_platform.config.settings import load_settings_from_env
from quant_platform.ui.actions import (
    action_catalog,
    build_ui_status,
    launch_ui_command,
    safe_command_catalog,
)


def test_build_ui_status_contains_safe_counts(tmp_path) -> None:  # noqa: ANN001
    status = build_ui_status(
        load_settings_from_env({}),
        registry_dir=tmp_path / "registry",
        report_dir=tmp_path / "reports",
    )

    assert "ui_available" in status
    assert status["app_path"].endswith("app.py")
    assert status["network_auto_run"] is False
    assert status["no_secrets_exposed"] is True
    assert status["dataset_count"] == 0
    assert "alpha_vantage_api_key" not in json.dumps(status).lower()


def test_launch_ui_command_is_streamlit_command_without_importing_streamlit() -> None:
    command = launch_ui_command(host="127.0.0.1", port=8510, show_browser=False)

    assert command[1:4] == ["-m", "streamlit", "run"]
    assert "127.0.0.1" in command
    assert "8510" in command
    assert "--server.headless" in command


def test_safe_command_catalog_is_read_only() -> None:
    commands = safe_command_catalog()
    joined = " ".join(command["command"] for command in commands)

    assert commands
    assert "ui-status" in joined
    assert "launch-ui" in joined
    assert "broker" not in joined.lower()
    assert "order" not in joined.lower()


def test_action_catalog_has_required_fields() -> None:
    catalog = action_catalog()
    required = {"label", "description", "reason", "risk", "command", "output_location"}

    assert catalog["allowed_actions"]
    assert catalog["prohibited_actions"]
    assert required <= set(catalog["allowed_actions"][0])
    assert required <= set(catalog["prohibited_actions"][0])
    assert any(row["label"] == "Trading real" for row in catalog["prohibited_actions"])
