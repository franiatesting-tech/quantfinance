"""VaR exception backtesting: Kupiec and Christoffersen diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def backtest_var(
    returns: pd.Series,
    var_loss: float | pd.Series,
    *,
    alpha: float = 0.95,
    label: str = "historical_var",
) -> dict[str, Any]:
    """Backtest a VaR loss threshold against realized returns."""

    exceptions = var_exceptions(returns, var_loss)
    kupiec = kupiec_unconditional_coverage(exceptions, alpha=alpha)
    christoffersen = christoffersen_independence(exceptions)
    return {
        "label": label,
        "alpha": float(alpha),
        "observations": int(len(exceptions)),
        "exceptions": int(exceptions.sum()),
        "exception_rate": float(exceptions.mean()) if len(exceptions) else None,
        "expected_exception_rate": float(1.0 - alpha),
        "kupiec_lr_uc": kupiec["lr_uc"],
        "kupiec_p_value": kupiec["p_value"],
        "kupiec_status": kupiec["status"],
        "christoffersen_lr_ind": christoffersen["lr_ind"],
        "christoffersen_p_value": christoffersen["p_value"],
        "christoffersen_status": christoffersen["status"],
        "traffic_light": traffic_light(exceptions, alpha=alpha),
        "loss_convention": "L_t = -R_t; exception when L_t > VaR_alpha",
        "literature": "Kupiec1995; Christoffersen1998",
    }


def var_exceptions(returns: pd.Series, var_loss: float | pd.Series) -> pd.Series:
    """Return boolean VaR exceptions under positive loss convention `L_t=-R_t`."""

    clean_returns = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    losses = -clean_returns
    if isinstance(var_loss, pd.Series):
        aligned = var_loss.astype(float).reindex(losses.index).dropna()
        losses = losses.reindex(aligned.index)
        threshold = aligned
    else:
        threshold = float(var_loss)
    exceptions = losses > threshold
    return exceptions.astype(bool)


def kupiec_unconditional_coverage(exceptions: pd.Series, *, alpha: float) -> dict[str, Any]:
    """Kupiec likelihood-ratio test for unconditional VaR coverage."""

    observed = exceptions.astype(bool).dropna()
    n = int(len(observed))
    x = int(observed.sum())
    p = 1.0 - alpha
    if n == 0 or not 0.0 < p < 1.0:
        return {"lr_uc": None, "p_value": None, "status": "INSUFFICIENT_DATA"}
    phat = x / n
    log_null = _binomial_log_likelihood(x, n, p)
    log_alt = _binomial_log_likelihood(x, n, phat)
    lr_uc = max(float(-2.0 * (log_null - log_alt)), 0.0)
    p_value = _chi_square_1_sf(lr_uc)
    return {
        "lr_uc": lr_uc,
        "p_value": p_value,
        "status": "PASS" if p_value is not None and p_value >= 0.05 else "REJECT",
    }


def christoffersen_independence(exceptions: pd.Series) -> dict[str, Any]:
    """Christoffersen independence LR test for clustered VaR exceptions."""

    values = exceptions.astype(bool).dropna().astype(int).to_numpy(dtype=int)
    if len(values) < 3:
        return {"lr_ind": None, "p_value": None, "status": "INSUFFICIENT_DATA"}
    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(values[:-1], values[1:]):
        if previous == 0 and current == 0:
            n00 += 1
        elif previous == 0 and current == 1:
            n01 += 1
        elif previous == 1 and current == 0:
            n10 += 1
        else:
            n11 += 1
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    log_restricted = _transition_log_likelihood(n00, n01, n10, n11, pi, pi)
    log_unrestricted = _transition_log_likelihood(n00, n01, n10, n11, pi0, pi1)
    lr_ind = max(float(-2.0 * (log_restricted - log_unrestricted)), 0.0)
    p_value = _chi_square_1_sf(lr_ind)
    return {
        "lr_ind": lr_ind,
        "p_value": p_value,
        "status": "PASS" if p_value is not None and p_value >= 0.05 else "REJECT",
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def traffic_light(exceptions: pd.Series, *, alpha: float) -> str:
    """Simple traffic-light interpretation for exception frequency."""

    observed = exceptions.astype(bool).dropna()
    if observed.empty:
        return "INSUFFICIENT_DATA"
    rate = float(observed.mean())
    expected = 1.0 - alpha
    if rate <= expected * 1.5:
        return "GREEN_EXCEPTION_RATE_WITHIN_TOLERANCE"
    if rate <= expected * 2.5:
        return "AMBER_EXCEPTION_RATE_REVIEW"
    return "RED_EXCEPTION_RATE_REJECT"


def _binomial_log_likelihood(x: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if x == 0 else -float("inf")
    if p >= 1.0:
        return 0.0 if x == n else -float("inf")
    return x * math.log(p) + (n - x) * math.log(1.0 - p)


def _transition_log_likelihood(
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    pi0: float,
    pi1: float,
) -> float:
    return (
        _binomial_log_likelihood(n01, n00 + n01, pi0)
        + _binomial_log_likelihood(n11, n10 + n11, pi1)
    )


def _chi_square_1_sf(value: float) -> float:
    return float(math.erfc(math.sqrt(max(value, 0.0) / 2.0)))
