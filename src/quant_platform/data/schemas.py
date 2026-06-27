"""Canonical market data schema definitions.

The schemas encode the minimum fields required by the roadmap before any feature,
risk, or backtest calculation can be considered auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar


class MarketType(StrEnum):
    """Supported market calendars and annualization domains."""

    EQUITY = "equity"
    CRYPTO = "crypto"


class Frequency(StrEnum):
    """Canonical data frequencies used by configs and metadata."""

    DAILY = "1d"
    HOURLY = "1h"
    MINUTE = "1m"


@dataclass(frozen=True)
class AssetMetadata:
    """Static asset metadata required to interpret a market data series."""

    asset_id: str
    market_type: MarketType
    currency: str
    source: str
    venue: str | None = None
    name: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset-level audit metadata for versioned research inputs."""

    dataset_id: str
    version: str
    source: str
    market_type: MarketType
    frequency: Frequency
    created_at: datetime
    schema_version: str = "market_bars_v1"
    description: str | None = None


@dataclass(frozen=True)
class MarketBarSchema:
    """Column contract for OHLCV bars.

    `venue` is optional because single-source equity datasets may not include it.
    If present, it is included in duplicate detection.
    """

    required_columns: ClassVar[tuple[str, ...]] = (
        "asset_id",
        "timestamp",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "market_type",
        "currency",
        "source",
    )
    optional_columns: ClassVar[tuple[str, ...]] = ("venue",)
    price_columns: ClassVar[tuple[str, ...]] = ("open", "high", "low", "close")
    duplicate_key_base_columns: ClassVar[tuple[str, ...]] = (
        "asset_id",
        "timestamp",
        "source",
    )

    @classmethod
    def duplicate_key_columns(cls, columns: list[str] | tuple[str, ...]) -> list[str]:
        """Return duplicate key columns, including `venue` when available."""

        key_columns = list(cls.duplicate_key_base_columns)
        if "venue" in columns:
            key_columns.append("venue")
        return key_columns
