from __future__ import annotations

import pytest

from quant_platform.ui.safety import (
    UISafetyError,
    assert_no_sensitive_keys,
    assert_read_only_action,
    redact_sensitive_text,
    safety_manifest,
)


def test_safety_manifest_lists_allowed_and_prohibited_actions() -> None:
    manifest = safety_manifest()

    assert manifest["allowed_actions"]
    assert manifest["prohibited_actions"]
    assert any(action["name"] == "secret_display" for action in manifest["prohibited_actions"])


def test_assert_read_only_action_rejects_trading_language() -> None:
    with pytest.raises(UISafetyError):
        assert_read_only_action("place_broker_order")


def test_redact_sensitive_text_blocks_secret_markers() -> None:
    assert redact_sensitive_text("ALPHA_VANTAGE_API_KEY placeholder") == "[redacted_sensitive_text]"
    assert redact_sensitive_text("configured=true") == "configured=true"


def test_assert_no_sensitive_keys_rejects_nested_secret_fields() -> None:
    with pytest.raises(UISafetyError):
        assert_no_sensitive_keys({"outer": {"token": "abc"}})
