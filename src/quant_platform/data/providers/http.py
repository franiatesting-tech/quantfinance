"""Small read-only HTTP client with sanitized errors and retry backoff."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


class ProviderHttpError(ValueError):
    """Raised when a read-only provider HTTP request fails safely."""


SENSITIVE_PARAM_NAMES = {"apikey", "api_key", "token", "key"}


@dataclass(frozen=True)
class HttpClientConfig:
    """HTTP runtime settings for public read-only providers."""

    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    cache_enabled: bool = False
    cache_dir: str | Path = ".cache/providers"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ProviderHttpError("timeout_seconds must be > 0.")
        if self.max_retries < 0:
            raise ProviderHttpError("max_retries must be >= 0.")
        if self.retry_backoff_seconds < 0:
            raise ProviderHttpError("retry_backoff_seconds must be >= 0.")


class ReadOnlyHttpClient:
    """Wrapper around `requests.get` that never exposes sensitive params."""

    def __init__(self, config: HttpClientConfig | None = None) -> None:
        self.config = config or HttpClientConfig()

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET a public JSON endpoint with retry/backoff and sanitized errors."""

        cache_path = self._cache_path(url, params)
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code >= 500 and attempt < self.config.max_retries:
                    last_error = ProviderHttpError(
                        f"HTTP {response.status_code} from {_clean_url(url)}"
                    )
                    self._sleep_before_retry(attempt)
                    continue
                if response.status_code >= 400:
                    raise ProviderHttpError(
                        "HTTP request failed: "
                        f"status={response.status_code} url={_clean_url(url)} "
                        f"params={_sanitize_params(params)}"
                    )
                try:
                    payload = response.json()
                    if cache_path is not None:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(
                            json.dumps(payload, sort_keys=True),
                            encoding="utf-8",
                        )
                    return payload
                except ValueError as exc:
                    raise ProviderHttpError(f"Invalid JSON from {_clean_url(url)}") from exc
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                self._sleep_before_retry(attempt)
        raise ProviderHttpError(
            f"HTTP request failed after retries: url={_clean_url(url)} "
            f"params={_sanitize_params(params)} error="
            f"{last_error.__class__.__name__ if last_error else 'unknown'}"
        )

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.config.retry_backoff_seconds > 0:
            time.sleep(self.config.retry_backoff_seconds * (attempt + 1))

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path | None:
        if not self.config.cache_enabled:
            return None
        fingerprint = hashlib.sha256(
            json.dumps(
                {"url": _clean_url(url), "params": _sanitize_params(params)},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return Path(self.config.cache_dir) / f"{fingerprint}.json"


def _clean_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _sanitize_params(params: dict[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    clean: dict[str, Any] = {}
    for key, value in params.items():
        clean[key] = "***" if key.lower() in SENSITIVE_PARAM_NAMES else value
    return clean
