"""Read-only Binance public spot OHLCV provider."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


class BinancePublicSpotProvider:
    """Daily spot OHLCV provider using Binance public `/api/v3/klines` only."""

    source = "binance_public"
    venue = "BINANCE_SPOT"
    base_url = "https://api.binance.com"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download public spot klines without API keys or private endpoints."""

        if MarketType(request.market_type) != MarketType.CRYPTO:
            raise DataProviderError("BinancePublicSpotProvider only supports crypto market_type.")
        frames = []
        failed: dict[str, str] = {}
        for symbol in request.symbols:
            try:
                response = requests.get(
                    f"{self.base_url}/api/v3/klines",
                    params=self._params(symbol, request.start, request.end),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not payload:
                    failed[symbol] = "No klines returned."
                    continue
                frames.append(self._normalize_klines(payload, symbol, request.currency))
            except Exception as exc:  # noqa: BLE001 - provider boundary records failures per symbol.
                failed[symbol] = str(exc)
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
            metadata={"provider": self.source, "venue": self.venue, "frequency": request.frequency},
        )

    @staticmethod
    def _params(symbol: str, start: str, end: str) -> dict[str, Any]:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        return {"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms}

    def _normalize_klines(
        self,
        payload: list[list[Any]],
        symbol: str,
        currency: str,
    ) -> pd.DataFrame:
        rows = []
        for item in payload:
            if len(item) < 7:
                raise DataProviderError(f"Malformed kline for {symbol}.")
            rows.append(
                {
                    "asset_id": symbol,
                    "timestamp": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "available_at": pd.to_datetime(int(item[6]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "market_type": MarketType.CRYPTO.value,
                    "currency": currency,
                    "source": self.source,
                    "venue": self.venue,
                }
            )
        return run_market_data_quality_checks(pd.DataFrame(rows))


def normalize_binance_klines(
    payload: list[list[Any]],
    symbol: str,
    currency: str = "USDT",
) -> pd.DataFrame:
    """Testable helper for Binance kline normalization."""

    return BinancePublicSpotProvider()._normalize_klines(payload, symbol, currency)
