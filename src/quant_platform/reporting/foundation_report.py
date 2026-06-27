"""Foundation pipeline report for synthetic bars and core metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quant_platform.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
)
from quant_platform.data.quality import run_market_data_quality_checks
from quant_platform.features.returns import cumulative_returns, simple_returns
from quant_platform.portfolio.baselines import equal_weight_weights
from quant_platform.risk.drawdown import max_drawdown
from quant_platform.risk.expected_shortfall import historical_expected_shortfall
from quant_platform.risk.var import historical_var


class FoundationReportError(ValueError):
    """Raised when a foundation report cannot be built from supplied bars."""


@dataclass(frozen=True)
class FoundationReport:
    """Auditable summary of the minimum data -> returns -> risk pipeline."""

    dataset_id: str
    dataset_version: str
    row_count: int
    asset_count: int
    observation_count: int
    periods_per_year: float
    alpha: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    historical_var: float
    historical_expected_shortfall: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly representation of the report."""

        return asdict(self)


def _close_price_matrix(market_bars: pd.DataFrame) -> pd.DataFrame:
    bars = market_bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    close = bars.pivot(index="timestamp", columns="asset_id", values="close").sort_index()
    close = close.astype(float).dropna(axis=0, how="any")
    if close.shape[0] < 2:
        raise FoundationReportError("At least two complete close-price rows are required.")
    return close


def build_foundation_report(
    market_bars: pd.DataFrame,
    periods_per_year: int | float,
    alpha: float = 0.95,
    dataset_id: str = "in_memory",
    dataset_version: str = "unversioned",
) -> FoundationReport:
    """Run the foundation pipeline and return a compact metric report.

    The report uses an equal-weight portfolio over complete close-price rows. Risk
    metrics convert portfolio returns to non-negative realized losses with
    `max(-R_t, 0)` to satisfy the platform's positive-loss convention.
    """

    run_market_data_quality_checks(market_bars)
    close = _close_price_matrix(market_bars)
    returns = simple_returns(close).dropna(axis=0, how="any")
    if returns.empty:
        raise FoundationReportError("No complete return rows are available after alignment.")

    weights = equal_weight_weights(tuple(str(column) for column in returns.columns))
    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    losses = (-portfolio_returns).clip(lower=0.0)
    equity_curve = cumulative_returns(portfolio_returns, initial_value=1.0)

    return FoundationReport(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        row_count=int(len(market_bars)),
        asset_count=int(close.shape[1]),
        observation_count=int(len(portfolio_returns)),
        periods_per_year=float(periods_per_year),
        alpha=float(alpha),
        annualized_return=annualized_return(portfolio_returns, periods_per_year),
        annualized_volatility=annualized_volatility(portfolio_returns, periods_per_year),
        sharpe_ratio=sharpe_ratio(portfolio_returns, periods_per_year=periods_per_year),
        maximum_drawdown=float(max_drawdown(equity_curve)),
        historical_var=historical_var(losses, alpha=alpha),
        historical_expected_shortfall=historical_expected_shortfall(losses, alpha=alpha),
    )


def report_has_finite_core_metrics(report: FoundationReport) -> bool:
    """Return whether all always-defined scalar metrics are finite."""

    values = [
        report.annualized_return,
        report.annualized_volatility,
        report.maximum_drawdown,
        report.historical_var,
        report.historical_expected_shortfall,
    ]
    return bool(np.isfinite(values).all())
