"""Read-only Polygon daily aggregate provider."""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse
from quant_platform.data.providers.http import ReadOnlyHttpClient
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


class PolygonDailyProvider:
    """Daily equity aggregate provider using Polygon read-only aggregates."""

    source = "polygon"
    venue = "POLYGON"
    base_url = "https://api.polygon.io"

    def __init__(self, api_key: str, http_client: ReadOnlyHttpClient | None = None) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise DataProviderError("Polygon API key is required.")
        self.http_client = http_client or ReadOnlyHttpClient()

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download daily adjusted equity aggregates without trading endpoints."""

        if MarketType(request.market_type) != MarketType.EQUITY:
            raise DataProviderError("PolygonDailyProvider only supports equity market_type.")
        frames = []
        failed: dict[str, str] = {}
        for symbol in request.symbols:
            try:
                payload = self.http_client.get_json(
                    f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{request.start}/{request.end}",
                    params={
                        "adjusted": "true",
                        "sort": "asc",
                        "limit": 50000,
                        "apiKey": self.api_key,
                    },
                )
                frame = self._normalize_payload(payload, symbol, request.currency)
                frames.append(frame)
            except Exception as exc:  # noqa: BLE001 - provider boundary records per-symbol failures.
                failed[symbol] = str(exc)
        return _response_from_frames(frames, failed, self.source, self.venue, request.frequency)

    def _normalize_payload(self, payload: object, symbol: str, currency: str) -> pd.DataFrame:
        if not isinstance(payload, dict):
            raise DataProviderError("Polygon response must be a JSON object.")
        if str(payload.get("status", "")).upper() not in {"OK", "DELAYED"}:
            raise DataProviderError(f"Polygon provider status: {payload.get('status', 'missing')}")
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise DataProviderError("Polygon response has no aggregate results.")
        rows = []
        for item in results:
            if not isinstance(item, dict):
                continue
            timestamp = pd.to_datetime(int(item["t"]), unit="ms", utc=True).normalize()
            rows.append(
                {
                    "asset_id": symbol,
                    "timestamp": timestamp,
                    "available_at": timestamp + pd.Timedelta(days=1),
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item["v"]),
                    "market_type": MarketType.EQUITY.value,
                    "currency": currency,
                    "source": self.source,
                    "venue": self.venue,
                }
            )
        return run_market_data_quality_checks(pd.DataFrame(rows))


def _response_from_frames(
    frames: list[pd.DataFrame],
    failed: dict[str, str],
    source: str,
    venue: str,
    frequency: str,
) -> OHLCVResponse:
    if frames:
        data = pd.concat(frames, ignore_index=True)
        run_market_data_quality_checks(data)
        successful = tuple(sorted(set(data["asset_id"].astype(str))))
    else:
        data = pd.DataFrame()
        successful = ()
    return OHLCVResponse(
        data=data,
        successful_symbols=successful,
        failed_symbols=failed,
        metadata={"provider": source, "venue": venue, "frequency": frequency},
    )
