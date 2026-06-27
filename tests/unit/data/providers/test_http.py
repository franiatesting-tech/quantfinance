from __future__ import annotations

import pytest

from quant_platform.data.providers import http
from quant_platform.data.providers.http import (
    HttpClientConfig,
    ProviderHttpError,
    ReadOnlyHttpClient,
)


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object | None = None,
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.json_error = json_error

    def json(self) -> object:
        if self.json_error:
            raise ValueError("bad json")
        return self.payload


def test_http_client_returns_200_json(monkeypatch) -> None:  # noqa: ANN001
    def fake_get(**kwargs):  # noqa: ANN001
        return FakeResponse(200, {"ok": True})

    monkeypatch.setattr(http.requests, "get", lambda *args, **kwargs: fake_get(**kwargs))

    payload = ReadOnlyHttpClient(HttpClientConfig(max_retries=0)).get_json(
        "https://example.test/path",
        params={"symbol": "SPY"},
    )

    assert payload == {"ok": True}


def test_http_error_sanitizes_api_key(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(http.requests, "get", lambda *args, **kwargs: FakeResponse(403, {}))

    with pytest.raises(ProviderHttpError) as excinfo:
        ReadOnlyHttpClient(HttpClientConfig(max_retries=0)).get_json(
            "https://example.test/path",
            params={"apikey": "secret", "symbol": "SPY"},
        )

    message = str(excinfo.value)
    assert "secret" not in message
    assert "***" in message
    assert "?apikey" not in message


def test_invalid_json_raises_sanitized_error(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        http.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(200, json_error=True),
    )

    with pytest.raises(ProviderHttpError, match="Invalid JSON"):
        ReadOnlyHttpClient(HttpClientConfig(max_retries=0)).get_json("https://example.test/path")


def test_retry_on_temporary_server_error(monkeypatch) -> None:  # noqa: ANN001
    calls = {"count": 0}

    def fake_get(*args, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeResponse(500, {})
        return FakeResponse(200, {"ok": True})

    monkeypatch.setattr(http.requests, "get", fake_get)
    monkeypatch.setattr(http.time, "sleep", lambda seconds: None)

    payload = ReadOnlyHttpClient(
        HttpClientConfig(max_retries=1, retry_backoff_seconds=0),
    ).get_json("https://example.test/path")

    assert payload == {"ok": True}
    assert calls["count"] == 2


def test_cache_uses_sanitized_fingerprint(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    calls = {"count": 0}

    def fake_get(*args, **kwargs):  # noqa: ANN001
        calls["count"] += 1
        return FakeResponse(200, {"ok": True})

    monkeypatch.setattr(http.requests, "get", fake_get)
    client = ReadOnlyHttpClient(
        HttpClientConfig(max_retries=0, cache_enabled=True, cache_dir=tmp_path),
    )

    assert client.get_json(
        "https://example.test/path", params={"api_key": "secret"}
    ) == {"ok": True}
    assert client.get_json(
        "https://example.test/path", params={"api_key": "secret"}
    ) == {"ok": True}
    assert calls["count"] == 1
    assert "secret" not in "\n".join(path.name for path in tmp_path.iterdir())
