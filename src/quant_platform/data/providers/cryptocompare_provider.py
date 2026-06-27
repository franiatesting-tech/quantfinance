"""Read-only CryptoCompare daily spot OHLCV provider."""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse
from quant_platform.data.providers.http import ReadOnlyHttpClient
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


class CryptoCompareDailyProvider:
    """Daily crypto spot provider using CryptoCompare public histoday data."""

    source = "cryptocompare"
    venue = "CRYPTOCOMPARE"
    url = "https://min-api.cryptocompare.com/data/v2/histoday"

    def __init__(self, api_key: str, http_client: ReadOnlyHttpClient | None = None) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise DataProviderError("CryptoCompare API key is required.")
        self.http_client = http_client or ReadOnlyHttpClient()

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download daily crypto spot OHLCV without account or derivatives endpoints."""

        if MarketType(request.market_type) != MarketType.CRYPTO:
            raise DataProviderError("CryptoCompareDailyProvider only supports crypto market_type.")
        frames = []
        failed: dict[str, str] = {}
        end_ts = int(pd.Timestamp(request.end, tz="UTC").timestamp())
        for symbol in request.symbols:
            try:
                base, quote = _parse_pair(symbol, request.currency)
                payload = self.http_client.get_json(
                    self.url,
                    params={
                        "fsym": base,
                        "tsym": quote,
                        "toTs": end_ts,
                        "limit": 2000,
                        "api_key": self.api_key,
                    },
                )
                frame = self._normalize_payload(payload, symbol, quote)
                frames.append(_filter_dates(frame, request.start, request.end))
            except Exception as exc:  # noqa: BLE001 - provider boundary records per-symbol failures.
                failed[symbol] = str(exc)
        return _response_from_frames(frames, failed, self.source, self.venue, request.frequency)

    def _normalize_payload(self, payload: object, symbol: str, quote: str) -> pd.DataFrame:
        if not isinstance(payload, dict):
            raise DataProviderError("CryptoCompare response must be a JSON object.")
        if str(payload.get("Response", "Success")) != "Success":
            raise DataProviderError("CryptoCompare API error.")
        data_section = payload.get("Data")
        rows_payload = data_section.get("Data") if isinstance(data_section, dict) else None
        if not isinstance(rows_payload, list) or not rows_payload:
            raise DataProviderError("CryptoCompare response has no daily rows.")
        rows = []
        for item in rows_payload:
            if not isinstance(item, dict):
                continue
            timestamp = pd.to_datetime(int(item["time"]), unit="s", utc=True).normalize()
            rows.append(
                {
                    "asset_id": symbol,
                    "timestamp": timestamp,
                    "available_at": timestamp + pd.Timedelta(days=1),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item.get("volumefrom", item.get("volumeto", 0.0))),
                    "market_type": MarketType.CRYPTO.value,
                    "currency": quote,
                    "source": self.source,
                    "venue": self.venue,
                }
            )
        return run_market_data_quality_checks(pd.DataFrame(rows))


def _parse_pair(symbol: str, default_quote: str) -> tuple[str, str]:
    clean = symbol.strip().upper()
    if "/" in clean:
        base, quote = clean.split("/", maxsplit=1)
    elif clean.endswith(default_quote.upper()):
        quote = default_quote.upper()
        base = clean[: -len(quote)]
    elif clean.endswith("USDT"):
        base, quote = clean[:-4], "USDT"
    elif clean.endswith("USD"):
        base, quote = clean[:-3], "USD"
    else:
        raise DataProviderError(f"Invalid CryptoCompare pair format: {symbol}")
    if not base or not quote:
        raise DataProviderError(f"Invalid CryptoCompare pair format: {symbol}")
    return base, quote


def _filter_dates(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    filtered = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)]
    if filtered.empty:
        raise DataProviderError("No rows in requested date range.")
    return filtered


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
