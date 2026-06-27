"""Read-only Alpha Vantage daily adjusted OHLCV provider."""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse
from quant_platform.data.providers.http import ReadOnlyHttpClient
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


class AlphaVantageDailyProvider:
    """Daily adjusted equity/ETF provider using Alpha Vantage public data API."""

    source = "alpha_vantage"
    venue = "ALPHA_VANTAGE"
    url = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, http_client: ReadOnlyHttpClient | None = None) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise DataProviderError("Alpha Vantage API key is required.")
        self.http_client = http_client or ReadOnlyHttpClient()

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download daily adjusted equity bars without exposing the API key."""

        if MarketType(request.market_type) != MarketType.EQUITY:
            raise DataProviderError("AlphaVantageDailyProvider only supports equity market_type.")
        frames = []
        failed: dict[str, str] = {}
        for symbol in request.symbols:
            try:
                payload = self.http_client.get_json(
                    self.url,
                    params={
                        "function": "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol": symbol,
                        "outputsize": "full",
                        "apikey": self.api_key,
                    },
                )
                _raise_for_alpha_vantage_message(payload)
                frame = self._normalize_payload(payload, symbol, request.currency)
                frames.append(_filter_dates(frame, request.start, request.end))
            except Exception as exc:  # noqa: BLE001 - provider boundary records per-symbol failures.
                failed[symbol] = _safe_message(exc)
        return _response_from_frames(frames, failed, self.source, self.venue, request.frequency)

    def _normalize_payload(
        self,
        payload: dict[str, object],
        symbol: str,
        currency: str,
    ) -> pd.DataFrame:
        series = payload.get("Time Series (Daily)")
        if not isinstance(series, dict) or not series:
            raise DataProviderError("Alpha Vantage response missing Time Series (Daily).")
        rows = []
        for date_text, values in series.items():
            if not isinstance(values, dict):
                continue
            raw_close = float(values["4. close"])
            adjusted_close = float(values.get("5. adjusted close", raw_close))
            adjustment = adjusted_close / raw_close if raw_close else 1.0
            timestamp = pd.Timestamp(str(date_text), tz="UTC")
            rows.append(
                {
                    "asset_id": symbol,
                    "timestamp": timestamp,
                    "available_at": timestamp + pd.Timedelta(days=1),
                    "open": float(values["1. open"]) * adjustment,
                    "high": float(values["2. high"]) * adjustment,
                    "low": float(values["3. low"]) * adjustment,
                    "close": adjusted_close,
                    "volume": float(values["6. volume"]),
                    "market_type": MarketType.EQUITY.value,
                    "currency": currency,
                    "source": self.source,
                    "venue": self.venue,
                }
            )
        return run_market_data_quality_checks(pd.DataFrame(rows))


def _raise_for_alpha_vantage_message(payload: object) -> None:
    if not isinstance(payload, dict):
        raise DataProviderError("Alpha Vantage response must be a JSON object.")
    for key in ("Note", "Information", "Error Message"):
        if key in payload:
            raise DataProviderError(f"Alpha Vantage provider message: {key}")


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


def _safe_message(exc: Exception) -> str:
    return str(exc).replace("apikey", "api_key_param")
