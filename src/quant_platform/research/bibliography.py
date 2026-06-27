"""Local bibliography metadata for professional terminal methods."""

from __future__ import annotations

from typing import Any


def method_catalog() -> list[dict[str, Any]]:
    """Return method-source rows used by the terminal report and UI."""

    return [
        _row("Returns", "Simple/log returns", "math_conventions.md", "TEST_OK"),
        _row("Performance", "Sharpe/Sortino/drawdown", "Benhamou; roadmap", "PARTIAL"),
        _row("CAPM", "Beta, Treynor, Jensen alpha", "Sharpe local references", "PENDING"),
        _row("Portfolio", "Mean-variance analytics", "Markowitz local references", "PENDING"),
        _row("VaR", "Historical VaR/ES", "BIS 1996; Acerbi-Tasche", "PARTIAL"),
        _row("Monte Carlo", "Bootstrap, normal, GBM", "Bormetti; BSM refs", "PENDING"),
        _row(
            "Backtesting", "Execution lag and transaction costs", "backtesting_costs.md", "TEST_OK"
        ),
        _row("Options", "Black-Scholes, CRR, Greeks", "Black-Scholes/Merton", "PENDING"),
        _row("Fixed Income", "Bond PV/duration/convexity", "roadmap refs", "PENDING"),
        _row("Rates", "Swap NPV, SOFR implied rate", "roadmap refs", "PENDING"),
        _row("Hedging", "Minimum-variance hedge ratio", "Grasselli-Hurd/roadmap", "PENDING"),
        _row("Exposure", "EE/EPE/ENE/PFE", "risk roadmap", "PENDING"),
    ]


def _row(block: str, method: str, source: str, validation: str) -> dict[str, Any]:
    return {
        "block": block,
        "method": method,
        "local_source": source,
        "validation_status": validation,
        "page_level_validation": "PENDING_PAGE_LEVEL_VALIDATION"
        if validation == "PENDING"
        else validation,
    }
