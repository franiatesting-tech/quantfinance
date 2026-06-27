"""Provider interfaces for read-only OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from quant_platform.data.schemas import MarketType


class DataProviderError(ValueError):
    """Raised when a read-only data provider fails safely."""


@dataclass(frozen=True)
class OHLCVRequest:
    """Read-only OHLCV request.

    This request never carries API secrets and never targets trading, account, or
    private endpoints.
    """

    symbols: tuple[str, ...]
    start: str
    end: str
    frequency: str
    market_type: MarketType
    source: str
    currency: str

    def __post_init__(self) -> None:
        if not self.symbols:
            raise DataProviderError("symbols must not be empty.")
        if not self.start or not self.end:
            raise DataProviderError("start and end must be provided.")
        if self.frequency != "1d":
            raise DataProviderError("Only daily frequency '1d' is supported in Iteration 005.")
        if not self.currency:
            raise DataProviderError("currency must not be empty.")


@dataclass(frozen=True)
class OHLCVResponse:
    """Normalized OHLCV response with coverage metadata."""

    data: pd.DataFrame
    successful_symbols: tuple[str, ...]
    failed_symbols: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Protocol for read-only market data providers."""

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Download and normalize read-only OHLCV bars."""
