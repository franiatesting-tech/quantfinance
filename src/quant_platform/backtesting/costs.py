"""Basic transaction cost utilities for vectorized backtests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series


class TransactionCostError(ValueError):
    """Raised when transaction cost inputs are invalid."""


def bps_to_rate(bps: float) -> float:
    """Convert basis points to a decimal rate.

    Convention: `1 bps = 0.0001`, so `10 bps = 0.001`.
    """

    clean_bps = float(bps)
    if not np.isfinite(clean_bps) or clean_bps < 0:
        raise TransactionCostError("bps must be finite and >= 0.")
    return clean_bps * 0.0001


def rate_to_bps(rate: float) -> float:
    """Convert a non-negative decimal rate to basis points."""

    clean_rate = _non_negative_rate(rate, "rate")
    return clean_rate / 0.0001


def _non_negative_rate(value: float, name: str) -> float:
    clean_value = float(value)
    if not np.isfinite(clean_value) or clean_value < 0:
        raise TransactionCostError(f"{name} must be finite and >= 0.")
    return clean_value


@dataclass(frozen=True)
class TransactionCostConfig:
    """Per-period cost rates expressed as decimal fractions of notional.

    `spread_rate` represents the full bid-ask spread. The implemented cost uses
    half-spread per unit of turnover, plus commission and slippage.
    """

    commission_rate: float = 0.0
    spread_rate: float = 0.0
    slippage_rate: float = 0.0
    funding_rate: float = 0.0

    @classmethod
    def from_bps(
        cls,
        commission_bps: float = 0.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        funding_bps: float = 0.0,
    ) -> TransactionCostConfig:
        """Build a transaction cost config from human-readable bps inputs."""

        return cls(
            commission_rate=bps_to_rate(commission_bps),
            spread_rate=bps_to_rate(spread_bps),
            slippage_rate=bps_to_rate(slippage_bps),
            funding_rate=bps_to_rate(funding_bps),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "commission_rate",
            _non_negative_rate(self.commission_rate, "commission_rate"),
        )
        object.__setattr__(self, "spread_rate", _non_negative_rate(self.spread_rate, "spread_rate"))
        object.__setattr__(
            self,
            "slippage_rate",
            _non_negative_rate(self.slippage_rate, "slippage_rate"),
        )
        object.__setattr__(
            self,
            "funding_rate",
            _non_negative_rate(self.funding_rate, "funding_rate"),
        )

    @property
    def per_turnover_rate(self) -> float:
        """Return cost rate per unit of portfolio turnover."""

        return self.commission_rate + 0.5 * self.spread_rate + self.slippage_rate


def _as_float_series(values: ArrayLike, name: str) -> pd.Series:
    try:
        series = (
            values.astype(float)
            if isinstance(values, pd.Series)
            else pd.Series(values, dtype=float)
        )
    except (TypeError, ValueError) as exc:
        raise TransactionCostError(f"{name} must be numeric.") from exc
    if series.empty:
        raise TransactionCostError(f"{name} must not be empty.")
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise TransactionCostError(f"{name} must not contain NaN or infinite values.")
    return series


def _as_weight_frame(weights: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame.")
    if weights.empty or weights.shape[1] == 0:
        raise TransactionCostError("weights must not be empty.")
    if weights.columns.has_duplicates:
        raise TransactionCostError("weight columns must be unique asset ids.")
    try:
        clean_weights = weights.astype(float)
    except (TypeError, ValueError) as exc:
        raise TransactionCostError("weights must be numeric.") from exc
    if not np.isfinite(clean_weights.to_numpy(dtype=float)).all():
        raise TransactionCostError("weights must not contain NaN or infinite values.")
    return clean_weights


def portfolio_turnover(
    weights: pd.DataFrame,
    initial_weights: pd.Series | None = None,
) -> pd.Series:
    """Compute turnover `sum_i |w_{i,t} - w_{i,t-1}|` for each rebalance row."""

    clean_weights = _as_weight_frame(weights)
    if initial_weights is None:
        initial = pd.Series(0.0, index=clean_weights.columns)
    else:
        initial = initial_weights.astype(float).reindex(clean_weights.columns)
        if initial.isna().any() or not np.isfinite(initial.to_numpy(dtype=float)).all():
            raise TransactionCostError(
                "initial_weights must align to weight columns and be finite."
            )

    previous = clean_weights.shift(1)
    previous.loc[clean_weights.index[0], :] = initial.to_numpy(dtype=float)
    return (clean_weights - previous).abs().sum(axis=1)


def transaction_costs_from_turnover(
    turnover: ArrayLike,
    config: TransactionCostConfig,
) -> pd.Series:
    """Convert turnover into per-period return drag from transaction costs."""

    clean_turnover = _as_float_series(turnover, "turnover")
    if (clean_turnover < 0).any():
        raise TransactionCostError("turnover must be >= 0.")
    return clean_turnover * config.per_turnover_rate


def apply_transaction_costs(
    gross_returns: ArrayLike,
    turnover: ArrayLike,
    config: TransactionCostConfig,
) -> pd.Series:
    """Subtract transaction costs and funding drag from gross simple returns."""

    clean_returns = _as_float_series(gross_returns, "gross_returns")
    if isinstance(turnover, pd.Series):
        aligned_turnover = turnover.astype(float).reindex(clean_returns.index)
    else:
        aligned_turnover = pd.Series(turnover, index=clean_returns.index, dtype=float)
    if aligned_turnover.isna().any():
        raise TransactionCostError("turnover is missing observations after alignment.")
    costs = transaction_costs_from_turnover(aligned_turnover, config)
    return clean_returns - costs - config.funding_rate
