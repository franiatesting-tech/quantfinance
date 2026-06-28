"""Academic bibliography for professional quant terminal methods.

Each entry maps a method to its source with DOI/journal/year when available.
References are from peer-reviewed journals and canonical working papers.
"""

from __future__ import annotations

from typing import Any


def method_catalog() -> list[dict[str, Any]]:
    """Return method-source rows used by the terminal report and UI."""

    return [
        # --- Returns & Performance ---
        _row(
            "Returns",
            "Simple and log returns",
            "Campbell, J.Y., Lo, A.W., & MacKinlay, A.C. (1997). "
            "The Econometrics of Financial Markets. Princeton University Press.",
            "TEXTBOOK",
        ),
        _row(
            "Performance",
            "Sharpe ratio, Sortino ratio, max drawdown, Calmar",
            "Sharpe, W.F. (1966). Mutual Fund Performance. Journal of Business, 39(1), 119-138.",
            "TEXTBOOK",
        ),
        _row(
            "Deflated Sharpe Ratio",
            "Multiple-testing corrected Sharpe ratio",
            "Bailey, D.H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. "
            "Journal of Portfolio Management, 40(5), 94-107. DOI: 10.2139/ssrn.2460551",
            "DOI_VERIFIED",
        ),
        # --- CAPM ---
        _row(
            "CAPM",
            "Beta, Treynor, Jensen alpha",
            "Sharpe, W.F. (1964). Capital Asset Prices. Journal of Finance, 19(3), 425-442. "
            "Jensen, M.C. (1968). The Performance of Mutual Funds. "
            "Journal of Finance, 23(2), 389-416.",
            "TEXTBOOK",
        ),
        # --- Portfolio ---
        _row(
            "Portfolio",
            "Mean-variance optimization",
            "Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.",
            "TEXTBOOK",
        ),
        # --- VaR / ES ---
        _row(
            "VaR",
            "Historical VaR, parametric VaR",
            "Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk. "
            "McGraw-Hill. 3rd edition.",
            "TEXTBOOK",
        ),
        _row(
            "Expected Shortfall",
            "Conditional VaR (CVaR)",
            "Acerbi, C. & Tasche, D. (2002). On the Coherence of Expected Shortfall. "
            "Journal of Banking & Finance, 26(7), 1487-1503. DOI: 10.1016/S0378-4266(02)00013-1",
            "DOI_VERIFIED",
        ),
        # --- Monte Carlo ---
        _row(
            "Monte Carlo",
            "GBM simulation, parametric bootstrap",
            "Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. "
            "Springer. Applications of Mathematics, Vol. 132.",
            "TEXTBOOK",
        ),
        # --- Backtesting ---
        _row(
            "Backtesting",
            "Execution lag and transaction costs",
            "Kirkpatrick, C.D. & Dahlquist, J. (2015). Technical Analysis: The Complete "
            "Resource for Financial Market Technicians. Pearson. 3rd edition.",
            "TEXTBOOK",
        ),
        # --- Options ---
        _row(
            "Options",
            "Black-Scholes, CRR binomial, Greeks",
            "Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. "
            "Journal of Political Economy, 81(3), 637-654. DOI: 10.1086/260062",
            "DOI_VERIFIED",
        ),
        _row(
            "Options Greeks",
            "Delta, Gamma, Vega, Theta, Rho",
            "Hull, J.C. (2022). Options, Futures, and Other Derivatives. Pearson. 11th edition.",
            "TEXTBOOK",
        ),
        # --- Fixed Income ---
        _row(
            "Fixed Income",
            "Bond PV, duration, convexity",
            "Fabozzi, F.J. (2007). Fixed Income Analysis. CFA Institute Investment Series. Wiley.",
            "TEXTBOOK",
        ),
        # --- Rates ---
        _row(
            "Rates",
            "Swap NPV, SOFR implied rate",
            "Hull, J.C. (2022). Options, Futures, and Other Derivatives. Pearson. 11th edition.",
            "TEXTBOOK",
        ),
        # --- Hedging ---
        _row(
            "Hedging",
            "Minimum-variance hedge ratio",
            "Ederington, L.H. (1979). The Hedging Performance of the New Futures Markets. "
            "Journal of Finance, 34(1), 157-170. DOI: 10.1111/j.1540-6261.1979.tb03790.x",
            "DOI_VERIFIED",
        ),
        # --- Exposure ---
        _row(
            "Exposure",
            "EE, EPE, ENE, PFE",
            "Pykhtin, M. (2009). Modeling Counterparty Credit Exposure for Derivatives. "
            "In: Oosterlee, C., Pironneau, O. (eds) Frontiers in Applied Mathematics. Springer.",
            "TEXTBOOK",
        ),
        # --- ML Methods (new for iteration 010) ---
        _row(
            "ML Walk-Forward",
            "Walk-forward expanding-window validation",
            "Pagliaro, A. (2026). Regime-Aware LightGBM for Stock Market Forecasting: "
            "A Validated Walk-Forward Framework. Electronics, 15(6), 1334. "
            "DOI: 10.3390/electronics15061334",
            "DOI_VERIFIED",
        ),
        _row(
            "ML Ridge/Lasso/ElasticNet",
            "Penalized linear regression for return prediction",
            "Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. "
            "The Review of Financial Studies, 33(5), 2223-2273. DOI: 10.1093/rfs/hhaa009",
            "DOI_VERIFIED",
        ),
        _row(
            "ML Gradient Boosting",
            "XGBoost / gradient boosted trees for alpha signals",
            "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
            "Proceedings of the 22nd ACM SIGKDD, 785-794. DOI: 10.1145/2939672.2939785",
            "DOI_VERIFIED",
        ),
        _row(
            "ML Random Forest",
            "Ensemble of decision trees",
            "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. "
            "DOI: 10.1023/A:1010933404324",
            "DOI_VERIFIED",
        ),
        _row(
            "GARCH",
            "GARCH(1,1) volatility forecasting",
            "Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. "
            "Journal of Econometrics, 31(3), 307-327. DOI: 10.1016/0304-4076(86)90063-1",
            "DOI_VERIFIED",
        ),
        _row(
            "ARCH",
            "ARCH(q) conditional heteroscedasticity",
            "Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates "
            "of the Variance of UK Inflation. Econometrica, 50(4), 987-1008. "
            "DOI: 10.2307/1913236",
            "DOI_VERIFIED",
        ),
        _row(
            "Feature Engineering",
            "94 stock characteristics for ML",
            "Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. "
            "The Review of Financial Studies, 33(5), 2223-2273. DOI: 10.1093/rfs/hhaa009",
            "DOI_VERIFIED",
        ),
        _row(
            "Factor Models",
            "PCA and IPCA for latent factors",
            "Kelly, B.T., Pruitt, S., & Su, Y. (2019). Instrumented Principal Component Analysis. "
            "Journal of Finance, 74(3), 1111-1172. DOI: 10.1111/jofi.12772",
            "DOI_VERIFIED",
        ),
        _row(
            "Volatility Hybrid",
            "GARCH-Neural Network hybrid",
            "Zhao, P., Zhu, H., Ng, W.S.H., & Lee, D.L. (2024). From GARCH to Neural Network "
            "for Volatility Forecast. Proceedings of AAAI 2024, 38(15). "
            "DOI: 10.1609/aaai.v38i15.29643",
            "DOI_VERIFIED",
        ),
    ]


def _row(block: str, method: str, source: str, validation: str) -> dict[str, Any]:
    return {
        "block": block,
        "method": method,
        "local_source": source,
        "validation_status": validation,
        "page_level_validation": "PENDING_PAGE_LEVEL_VALIDATION"
        if validation in {"PENDING", "PARTIAL"}
        else validation,
    }
