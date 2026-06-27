"""Minimal read-only Nasdaq Data Link provider."""

from __future__ import annotations

import pandas as pd

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse
from quant_platform.data.providers.http import ReadOnlyHttpClient
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


class NasdaqDataLinkProvider:
    """Configurable Nasdaq Data Link dataset provider for read-only research."""

    source = "nasdaq_data_link"
    venue = "NASDAQ_DATA_LINK"
    base_url = "https://data.nasdaq.com/api/v3/datasets"

    def __init__(
        self,
        api_key: str,
        dataset_code: str | None = None,
        market_type: MarketType = MarketType.EQUITY,
        http_client: ReadOnlyHttpClient | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise DataProviderError("Nasdaq Data Link API key is required.")
        self.dataset_code = dataset_code
        self.market_type = MarketType(market_type)
        self.http_client = http_client or ReadOnlyHttpClient()

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download one configured dataset per symbol or templated dataset code."""

        if self.dataset_code is None:
            raise DataProviderError("Nasdaq Data Link dataset_code is required.")
        frames = []
        failed: dict[str, str] = {}
        for symbol in request.symbols:
            try:
                dataset_code = self.dataset_code.format(symbol=symbol)
                payload = self.http_client.get_json(
                    f"{self.base_url}/{dataset_code}.json",
                    params={
                        "api_key": self.api_key,
                        "start_date": request.start,
                        "end_date": request.end,
                    },
                )
                frames.append(self._normalize_payload(payload, symbol, request.currency))
            except Exception as exc:  # noqa: BLE001 - provider boundary records per-symbol failures.
                failed[symbol] = str(exc)
        return _response_from_frames(frames, failed, self.source, self.venue, request.frequency)

    def _normalize_payload(self, payload: object, symbol: str, currency: str) -> pd.DataFrame:
        if not isinstance(payload, dict):
            raise DataProviderError("Nasdaq Data Link response must be a JSON object.")
        if "quandl_error" in payload:
            raise DataProviderError("Nasdaq Data Link API error.")
        dataset = payload.get("dataset")
        if not isinstance(dataset, dict):
            raise DataProviderError("Nasdaq Data Link response missing dataset.")
        columns = [str(column) for column in dataset.get("column_names", [])]
        data = dataset.get("data", [])
        if not columns or not isinstance(data, list) or not data:
            raise DataProviderError("Nasdaq Data Link dataset has no rows.")
        raw = pd.DataFrame(data, columns=columns)
        column_map = {column.lower().replace(" ", "_"): column for column in raw.columns}
        date_col = column_map.get("date")
        open_col = column_map.get("open")
        high_col = column_map.get("high")
        low_col = column_map.get("low")
        close_col = column_map.get("adj_close") or column_map.get("close")
        volume_col = column_map.get("volume")
        required = [date_col, open_col, high_col, low_col, close_col, volume_col]
        if any(column is None for column in required):
            raise DataProviderError("Nasdaq Data Link dataset lacks OHLCV columns.")
        timestamp = pd.to_datetime(raw[date_col], utc=True, format="mixed")
        frame = pd.DataFrame(
            {
                "asset_id": symbol,
                "timestamp": timestamp,
                "available_at": timestamp + pd.Timedelta(days=1),
                "open": raw[open_col].astype(float),
                "high": raw[high_col].astype(float),
                "low": raw[low_col].astype(float),
                "close": raw[close_col].astype(float),
                "volume": raw[volume_col].astype(float),
                "market_type": self.market_type.value,
                "currency": currency,
                "source": self.source,
                "venue": self.venue,
            }
        )
        return run_market_data_quality_checks(frame)


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
