"""Read-only Yahoo Finance provider via yfinance."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.data.schemas import MarketType


class YFinanceDailyProvider:
    """Daily read-only equity/ETF provider backed by `yfinance.download`."""

    source = "yfinance"
    venue = "YAHOO_PUBLIC"

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download and normalize daily equity/ETF bars without API keys."""

        if MarketType(request.market_type) != MarketType.EQUITY:
            raise DataProviderError("YFinanceDailyProvider only supports equity market_type.")
        frames = []
        failed: dict[str, str] = {}
        for symbol in request.symbols:
            try:
                raw = yf.download(
                    symbol,
                    start=request.start,
                    end=request.end,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                )
                if raw is None or raw.empty:
                    failed[symbol] = "No rows returned."
                    continue
                frames.append(self._normalize_symbol(raw, symbol, request.currency))
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
            metadata={
                "provider": self.source,
                "venue": self.venue,
                "frequency": request.frequency,
                "adjusted_prices": True,
            },
        )

    def _normalize_symbol(self, raw: pd.DataFrame, symbol: str, currency: str) -> pd.DataFrame:
        frame = self._flatten_columns(raw, symbol).copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise DataProviderError(f"Missing yfinance columns for {symbol}: {sorted(missing)}")

        open_prices = frame["Open"].astype(float)
        high = frame["High"].astype(float)
        low = frame["Low"].astype(float)
        close = frame["Close"].astype(float)
        if "Adj Close" in frame.columns:
            adjusted_close = frame["Adj Close"].astype(float)
            adjustment = adjusted_close / close
            open_prices = open_prices * adjustment
            high = high * adjustment
            low = low * adjustment
            close = adjusted_close

        timestamp = pd.to_datetime(frame.index, utc=True)
        bars = pd.DataFrame(
            {
                "asset_id": symbol,
                "timestamp": timestamp,
                "available_at": timestamp + pd.Timedelta(days=1),
                "open": open_prices.to_numpy(dtype=float),
                "high": high.to_numpy(dtype=float),
                "low": low.to_numpy(dtype=float),
                "close": close.to_numpy(dtype=float),
                "volume": frame["Volume"].astype(float).to_numpy(dtype=float),
                "market_type": MarketType.EQUITY.value,
                "currency": currency,
                "source": self.source,
                "venue": self.venue,
            }
        )
        return run_market_data_quality_checks(bars)

    @staticmethod
    def _flatten_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if not isinstance(raw.columns, pd.MultiIndex):
            return raw
        for level in range(raw.columns.nlevels):
            if symbol in raw.columns.get_level_values(level):
                return raw.xs(symbol, axis=1, level=level)
        for level in range(raw.columns.nlevels):
            values = raw.columns.get_level_values(level).unique()
            if len(values) == 1:
                return raw.droplevel(level, axis=1)
        raise DataProviderError("Unsupported yfinance MultiIndex columns.")


def normalize_yfinance_frame(raw: pd.DataFrame, symbol: str, currency: str = "USD") -> pd.DataFrame:
    """Testable helper for yfinance frame normalization."""

    return YFinanceDailyProvider()._normalize_symbol(raw, symbol, currency)
