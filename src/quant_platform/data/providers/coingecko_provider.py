"""CoinGecko metadata provider placeholder.

CoinGecko is intentionally not used as the primary OHLCV source in Iteration 005.
It is reserved for future read-only metadata or universe enrichment.
"""

from __future__ import annotations

from quant_platform.data.providers.base import DataProviderError, OHLCVRequest, OHLCVResponse


class CoinGeckoMetadataProvider:
    """Placeholder for future CoinGecko metadata enrichment."""

    source = "coingecko"
    status = "PENDIENTE_IMPLEMENTACION"

    def download_ohlcv(self, request: OHLCVRequest) -> OHLCVResponse:
        """Reject OHLCV requests because CoinGecko is metadata-only for now."""

        raise DataProviderError("CoinGecko OHLCV is not implemented; use metadata only in future.")

    def metadata_status(self) -> dict[str, str]:
        """Return explicit placeholder status for documentation and CLI display."""

        return {"source": self.source, "status": self.status}
