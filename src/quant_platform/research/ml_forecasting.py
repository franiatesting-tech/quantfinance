"""Academic-grade walk-forward ML forecasting for financial returns.

Implements the methodology described in:

  Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning."
  The Review of Financial Studies, 33(5), 2223-2273. DOI: 10.1093/rfs/hhaa009

  Pagliaro, A. (2026). "Regime-Aware LightGBM for Stock Market Forecasting:
  A Validated Walk-Forward Framework." Electronics, 15(6), 1334.
  DOI: 10.3390/electronics15061334

  Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
  Journal of Econometrics, 31(3), 307-327. DOI: 10.1016/0304-4076(86)90063-1

  Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates
  of the Variance of UK Inflation." Econometrica, 50(4), 987-1008.

Validation protocol:
  - Walk-forward with expanding window (never uses future data in training)
  - Purge gap >= prediction horizon to prevent look-ahead bias
  - Per-fold StandardScaler fitting (fit on train, transform test)
  - No shuffling of observations
  - Models: Naive, Historical Mean, Ridge, Lasso, ElasticNet, GBM, RF, GARCH(1,1)

All statuses are diagnostic, never investment advice.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtesting.metrics import hit_rate, sharpe_ratio
from quant_platform.research.returns import daily_simple_returns

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_NAIVE_BETTER = "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE"
STATUS_OUTPERFORMS = "MODEL_EDGE_PASSED_STRICT_DIAGNOSTIC_GATES"
STATUS_SKLEARN_MISSING = "SKLEARN_UNAVAILABLE_BASELINES_ONLY"


class MLForecastingError(ValueError):
    """Raised when ML forecasting inputs are invalid."""


def run_walk_forward_forecast(
    prices: pd.Series,
    volume: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    min_train_size: int = 252,
    horizon_days: int = 1,
    max_test_observations: int = 252,
    refit_frequency_days: int = 5,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Run leakage-aware walk-forward forecasting with multiple ML models.

    Protocol (Gu, Kelly & Xiu 2020; Pagliaro 2026):
      1. Expanding window: train set grows from min_train_size to T-1.
      2. Purge gap = horizon_days between train end and test start.
      3. Per-fold StandardScaler fitted ONLY on training data.
      4. No shuffling, no future data leakage.
      5. Each model is trained independently per fold.

    Models evaluated (when sklearn available):
      - Naive random walk (pred = 0)
      - Historical mean (pred = mean of train targets)
      - Ridge regression (L2 penalty, alpha=1.0)
      - Lasso regression (L1 penalty, alpha=1e-3)
      - ElasticNet (L1+L2, alpha=1e-3, l1_ratio=0.5)
      - Gradient Boosting (n_estimators=100, max_depth=3)
      - Random Forest (n_estimators=100, max_depth=5)

    When sklearn unavailable, only naive/historical_mean baselines are used.
    GARCH(1,1) volatility forecast is computed separately for risk assessment.
    """

    if min_train_size < 30:
        raise MLForecastingError("min_train_size must be >= 30.")
    if horizon_days < 1 or horizon_days > 21:
        raise MLForecastingError("horizon_days must be in [1, 21].")
    if max_test_observations < 10:
        raise MLForecastingError("max_test_observations must be >= 10.")
    if refit_frequency_days < 1 or refit_frequency_days > 63:
        raise MLForecastingError("refit_frequency_days must be in [1, 63].")

    clean_prices = _as_price_series(prices)
    returns = daily_simple_returns(clean_prices)
    frame = _feature_frame(returns, volume=volume, benchmark_returns=benchmark_returns)
    frame["target_next_return"] = returns.shift(-horizon_days).reindex(frame.index)
    frame["target_direction"] = (frame["target_next_return"] > 0.0).astype(float)
    frame = frame.dropna(how="any")

    if len(frame) <= min_train_size + 10:
        return _insufficient_payload(len(frame), min_train_size)

    feature_columns = [c for c in frame.columns if c.startswith("feature_")]
    first_test_pos = max(min_train_size, len(frame) - max_test_observations)

    rows: list[dict[str, float | str]] = []
    model_fold_metrics: dict[str, list[dict[str, float]]] = {
        "naive": [], "historical_mean": [],
        "ridge": [], "lasso": [], "elasticnet": [],
        "gradient_boosting": [], "random_forest": [],
    }
    model_cache: dict[str, Any] | None = None
    last_refit_pos = -10_000

    for test_pos in range(first_test_pos, len(frame)):
        purge = horizon_days
        train_end = max(1, test_pos - purge)
        train = frame.iloc[:train_end]
        test = frame.iloc[test_pos]
        actual = float(test["target_next_return"])

        naive_pred = 0.0
        hist_mean_pred = float(train["target_next_return"].mean())

        fold_preds: dict[str, float] = {
            "naive": naive_pred,
            "historical_mean": hist_mean_pred,
        }

        if SKLEARN_AVAILABLE:
            if model_cache is None or test_pos - last_refit_pos >= refit_frequency_days:
                model_cache = _fit_sklearn_models(train, feature_columns)
                last_refit_pos = test_pos
            if model_cache:
                fold_preds.update(_predict_sklearn_models(model_cache, test, feature_columns))

        best_model, best_pred = _select_best_model(fold_preds)

        rows.append({
            "timestamp": pd.Timestamp(test.name).isoformat(),
            "actual_return": actual,
            "naive_random_walk_prediction": naive_pred,
            "historical_mean_prediction": hist_mean_pred,
            "model_prediction": best_pred,
            "model_name": best_model,
            **{f"{k}_prediction": v for k, v in fold_preds.items()},
        })

        for model_name, pred in fold_preds.items():
            if model_name in model_fold_metrics:
                model_fold_metrics[model_name].append({
                    "actual": actual, "predicted": pred,
                })

    result = pd.DataFrame(rows)
    actuals = result["actual_return"].to_numpy(dtype=float)
    model_preds = result["model_prediction"].to_numpy(dtype=float)
    naive_preds = result["naive_random_walk_prediction"].to_numpy(dtype=float)

    model_metrics = _regression_and_direction_metrics(actuals, model_preds, periods_per_year)
    naive_metrics = _regression_and_direction_metrics(actuals, naive_preds, periods_per_year)
    status, approval_gates = _model_status(model_metrics, naive_metrics)

    sklearn_used = bool(
        (result["model_name"] != "historical_mean_baseline").any()
        and (result["model_name"] != "naive_random_walk_baseline").any()
    )

    if not sklearn_used:
        status_detail = STATUS_SKLEARN_MISSING
    else:
        status_detail = "SKLEARN_OPTIONAL_MODEL_USED"

    per_model_comparison = {}
    for model_name, fold_list in model_fold_metrics.items():
        if not fold_list:
            continue
        m_actuals = np.array([f["actual"] for f in fold_list])
        m_preds = np.array([f["predicted"] for f in fold_list])
        per_model_comparison[model_name] = _regression_and_direction_metrics(
            m_actuals, m_preds, periods_per_year,
        )

    garch_forecast = _garch_forecast(returns) if len(returns) >= 252 else None

    return {
        "status": status,
        "model_status": status,
        "status_detail": status_detail,
        "validation": "walk_forward_expanding_window_no_shuffle_no_leakage",
        "target": "next_day_return" if horizon_days == 1 else f"{horizon_days}_day_forward_return",
        "horizon_days": int(horizon_days),
        "train_min_observations": int(min_train_size),
        "max_test_observations": int(max_test_observations),
        "refit_frequency_days": int(refit_frequency_days),
        "test_observations": int(len(result)),
        "purge_gap": int(horizon_days),
        "features": feature_columns,
        "n_features": len(feature_columns),
        "models_evaluated": _available_model_names(),
        "per_model_metrics": per_model_comparison,
        "rmse": model_metrics["rmse"],
        "mae": model_metrics["mae"],
        "directional_accuracy": model_metrics["directional_accuracy"],
        "precision_up": model_metrics["precision_up"],
        "recall_up": model_metrics["recall_up"],
        "information_coefficient": model_metrics["information_coefficient"],
        "strategy_sharpe": model_metrics["strategy_sharpe"],
        "hit_rate": model_metrics["hit_rate"],
        "oos_r_squared": model_metrics["oos_r_squared"],
        "baseline_rmse": naive_metrics["rmse"],
        "baseline_mae": naive_metrics["mae"],
        "baseline_directional_accuracy": naive_metrics["directional_accuracy"],
        "rmse_improvement_vs_naive": naive_metrics["rmse"] - model_metrics["rmse"],
        "mae_improvement_vs_naive": naive_metrics["mae"] - model_metrics["mae"],
        "directional_accuracy_edge_vs_naive": (
            model_metrics["directional_accuracy"] - naive_metrics["directional_accuracy"]
        ),
        "approval_gates": approval_gates,
        "status_reason": _status_reason(status, approval_gates),
        "garch_forecast": garch_forecast,
        "calibration_warning": _calibration_warning(model_metrics, naive_metrics),
        "prediction_rows": rows[-252:],
        "not_investment_advice": True,
    }


# ---------------------------------------------------------------------------
# Feature Engineering (Gu et al. 2020, Section II.A)
# ---------------------------------------------------------------------------

def _feature_frame(
    returns: pd.Series,
    volume: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build feature matrix from return series.

    Features follow the taxonomy of Gu, Kelly & Xiu (2020):
      - Price/Volume: lagged returns, rolling moments
      - Momentum: multi-horizon cumulative returns
      - Volatility: rolling std, downside vol
      - Trend: rolling sharpe, drawdown
      - Cross-asset: rolling beta (if benchmark available)
    """
    frame = pd.DataFrame(index=returns.index)

    # --- Lagged returns (autoregressive features) ---
    frame["feature_lag_1_return"] = returns.shift(1)
    frame["feature_lag_2_return"] = returns.shift(2)
    frame["feature_lag_5_return"] = returns.shift(5)
    frame["feature_lag_21_return"] = returns.shift(21)

    # --- Rolling moments ---
    frame["feature_rolling_mean_21"] = returns.rolling(21).mean()
    frame["feature_rolling_std_21"] = returns.rolling(21).std(ddof=1)
    frame["feature_rolling_mean_63"] = returns.rolling(63).mean()
    frame["feature_rolling_std_63"] = returns.rolling(63).std(ddof=1)

    # --- Momentum (multi-horizon) ---
    for window in (5, 21, 63, 126):
        frame[f"feature_momentum_{window}"] = (
            (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
        )

    # --- Volatility regime ---
    frame["feature_vol_regime"] = (
        frame["feature_rolling_std_21"] / frame["feature_rolling_std_63"]
    )

    # --- Drawdown ---
    equity = (1.0 + returns).cumprod()
    frame["feature_drawdown"] = equity / equity.cummax() - 1.0

    # --- Rolling Sharpe ---
    rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std(ddof=1)
    frame["feature_rolling_sharpe_63"] = rolling_sharpe * np.sqrt(252)

    # --- Downside deviation ---
    downside = returns.copy()
    downside[downside > 0] = 0.0
    frame["feature_downside_vol_63"] = downside.rolling(63).std(ddof=1) * np.sqrt(252)

    # --- Skewness and kurtosis (rolling) ---
    frame["feature_skewness_63"] = returns.rolling(63).skew()
    frame["feature_kurtosis_63"] = returns.rolling(63).kurt()

    # --- Volume features ---
    if volume is not None:
        aligned_volume = volume.astype(float).reindex(returns.index)
        frame["feature_volume_change"] = aligned_volume.pct_change()
        frame["feature_volume_ma_ratio"] = aligned_volume / aligned_volume.rolling(21).mean()

    # --- Cross-asset features ---
    if benchmark_returns is not None:
        asset, benchmark = returns.align(benchmark_returns.astype(float), join="left")
        rolling_cov = asset.rolling(63).cov(benchmark)
        rolling_var = benchmark.rolling(63).var(ddof=1)
        frame["feature_rolling_beta_63"] = rolling_cov / rolling_var
        frame["feature_tracking_error_63"] = (asset - benchmark).rolling(63).std(ddof=1)
        rolling_corr = asset.rolling(63).corr(benchmark)
        frame["feature_rolling_correlation_63"] = rolling_corr

    return frame.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Walk-Forward Model Selection (Pagliaro 2026)
# ---------------------------------------------------------------------------

def _select_best_model(
    fold_preds: dict[str, float],
) -> tuple[str, float]:
    """Select the primary prediction without training any model twice.

    Every candidate prediction was already produced by a leakage-aware per-fold
    fit. The selected forecast prioritizes tree boosting, then random forests,
    then regularized linear models. Per-model diagnostics are still reported
    separately, so the paper can compare all models out-of-sample.
    """
    if not SKLEARN_AVAILABLE:
        return "historical_mean_baseline", fold_preds.get("historical_mean", 0.0)
    for model_name in ("gradient_boosting", "random_forest", "elasticnet", "ridge", "lasso"):
        if model_name in fold_preds:
            return model_name, fold_preds[model_name]
    return "historical_mean_baseline", fold_preds.get("historical_mean", 0.0)


def _get_model_in_sample_predictions(
    train: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> np.ndarray | None:
    """Get in-sample predictions from a model for model selection."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    X = train[feature_columns].values
    y = train["target_next_return"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_map = {
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1e-3, max_iter=5000),
        "elasticnet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.8, random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=60, max_depth=5, random_state=42, n_jobs=-1,
        ),
    }

    model = model_map.get(model_name)
    if model is None:
        return None

    try:
        model.fit(X_scaled, y)
        return model.predict(X_scaled)
    except Exception:
        return None


def _sklearn_fold_predictions(
    train: pd.DataFrame,
    test: pd.Series,
    feature_columns: list[str],
) -> dict[str, float]:
    """Train all sklearn models and return predictions for one test observation.

    Per-fold StandardScaler fitting ensures no future data leakage.
    """
    fitted = _fit_sklearn_models(train, feature_columns)
    return _predict_sklearn_models(fitted, test, feature_columns) if fitted else {}


def _fit_sklearn_models(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any] | None:
    """Fit sklearn models once for a walk-forward refit point."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    X_train = train[feature_columns].values
    y_train = train["target_next_return"].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    models = {
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1e-3, max_iter=5000),
        "elasticnet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.08,
            subsample=0.8, random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=40, max_depth=5, random_state=42, n_jobs=1,
        ),
    }
    fitted = {}
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            fitted[name] = model
        except Exception:
            continue
    return {"scaler": scaler, "models": fitted} if fitted else None


def _predict_sklearn_models(
    fitted: dict[str, Any],
    test: pd.Series,
    feature_columns: list[str],
) -> dict[str, float]:
    """Predict one observation using the latest past-only sklearn model cache."""
    scaler = fitted.get("scaler")
    models = fitted.get("models", {})
    if scaler is None or not isinstance(models, dict):
        return {}
    X_test = pd.DataFrame([test[feature_columns]], columns=feature_columns).values
    X_test_scaled = scaler.transform(X_test)
    predictions = {}
    for name, model in models.items():
        try:
            pred = float(model.predict(X_test_scaled)[0])
            if np.isfinite(pred):
                predictions[name] = pred
        except Exception:
            continue
    return predictions


def _available_model_names() -> list[str]:
    base = ["naive_random_walk_baseline", "historical_mean_baseline"]
    if SKLEARN_AVAILABLE:
        base.extend([
            "ridge_regression",
            "lasso_regression",
            "elasticnet_regression",
            "gradient_boosting",
            "random_forest",
        ])
    return base


# ---------------------------------------------------------------------------
# GARCH(1,1) Volatility Forecasting
# ---------------------------------------------------------------------------

def _garch_forecast(returns: pd.Series) -> dict[str, Any] | None:
    """Fit GARCH(1,1) and produce volatility forecast.

    GARCH(1,1): sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}

    Reference: Bollerslev (1986), Journal of Econometrics, 31(3), 307-327.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    r = returns.dropna().values
    if len(r) < 252:
        return None

    mu = np.mean(r)
    eps = r - mu

    def neg_loglik(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        T = len(eps)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(eps)
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            if sigma2[t] <= 0:
                return 1e10
        loglik = -0.5 * np.sum(np.log(sigma2) + eps**2 / sigma2)
        return -loglik

    x0 = np.array([np.var(eps) * 0.05, 0.08, 0.88])
    bounds = [(1e-10, None), (1e-6, 0.999), (1e-6, 0.999)]

    try:
        result = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = result.x
    except Exception:
        return None

    sigma2_last = eps[-1] ** 2
    sigma2_forecast = omega + alpha * eps[-1] ** 2 + beta * sigma2_last
    vol_forecast_annual = np.sqrt(sigma2_forecast) * np.sqrt(252)

    return {
        "model": "GARCH(1,1)",
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "persistence": float(alpha + beta),
        "unconditional_variance": float(omega / (1 - alpha - beta)) if alpha + beta < 1 else None,
        "conditional_volatility_daily": float(np.sqrt(sigma2_forecast)),
        "conditional_volatility_annual": float(vol_forecast_annual),
        "half_life_days": int(np.log(0.5) / np.log(alpha + beta)) if 0 < alpha + beta < 1 else None,
        "reference": "Bollerslev (1986), J. Econometrics 31(3), 307-327",
    }


# ---------------------------------------------------------------------------
# Metrics (Campbell & Thompson 2008; Bailey & Lopez de Prado 2014)
# ---------------------------------------------------------------------------

def _regression_and_direction_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    periods_per_year: int,
) -> dict[str, float]:
    errors = predictions - actuals
    predicted_up = predictions > 0.0
    actual_up = actuals > 0.0
    true_positive = float(np.sum(predicted_up & actual_up))
    predicted_positive = float(np.sum(predicted_up))
    actual_positive = float(np.sum(actual_up))
    strategy_returns = np.where(predicted_up, actuals, -actuals)

    if len(actuals) > 1 and np.std(predictions) > 0 and np.std(actuals) > 0:
        ic = float(np.corrcoef(predictions, actuals)[0, 1])
    else:
        ic = 0.0

    # OOS R-squared (Campbell & Thompson 2008)
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((actuals - np.mean(actuals))**2)
    oos_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "directional_accuracy": float(np.mean(predicted_up == actual_up)),
        "precision_up": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall_up": true_positive / actual_positive if actual_positive else 0.0,
        "information_coefficient": ic,
        "strategy_sharpe": float(sharpe_ratio(pd.Series(strategy_returns), 0.0, periods_per_year)),
        "hit_rate": float(hit_rate(pd.Series(strategy_returns))),
        "oos_r_squared": float(oos_r2),
    }


def _model_status(model: dict[str, float], naive: dict[str, float]) -> tuple[str, dict[str, bool]]:
    direction_edge = model["directional_accuracy"] - naive["directional_accuracy"]
    gates = {
        "rmse_improves_naive": model["rmse"] < naive["rmse"],
        "mae_improves_naive": model["mae"] < naive["mae"],
        "directional_accuracy_edge_ge_2pct": direction_edge >= 0.02,
        "directional_accuracy_ge_52pct": model["directional_accuracy"] >= 0.52,
        "oos_r_squared_positive": model["oos_r_squared"] > 0.0,
        "information_coefficient_positive": model["information_coefficient"] > 0.0,
        "strategy_sharpe_positive": model["strategy_sharpe"] > 0.0,
    }
    if all(gates.values()):
        return STATUS_OUTPERFORMS, gates
    return STATUS_NAIVE_BETTER, gates


def _status_reason(status: str, approval_gates: dict[str, bool]) -> str:
    if status == STATUS_OUTPERFORMS:
        return "All strict diagnostic gates passed under the configured walk-forward test."
    failed = [name for name, passed in approval_gates.items() if not passed]
    return "Predictive edge not validated; failed gates: " + ", ".join(failed)


def _calibration_warning(model: dict[str, float], naive: dict[str, float]) -> str | None:
    if model["rmse"] >= naive["rmse"]:
        return "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE"
    if model["oos_r_squared"] <= 0.0:
        return "OOS_R_SQUARED_NOT_POSITIVE"
    if model["information_coefficient"] <= 0.0:
        return "INFORMATION_COEFFICIENT_NOT_POSITIVE"
    if model["directional_accuracy"] < 0.52:
        return "DIRECTIONAL_ACCURACY_EDGE_WEAK"
    return None


def _insufficient_payload(observations: int, min_train_size: int) -> dict[str, Any]:
    return {
        "status": STATUS_INSUFFICIENT,
        "model_status": STATUS_INSUFFICIENT,
        "observations": int(observations),
        "min_train_size": int(min_train_size),
        "validation": "walk_forward_expanding_window_no_shuffle_no_leakage",
        "not_investment_advice": True,
    }


def _as_price_series(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series) or prices.empty:
        raise MLForecastingError("prices must be a non-empty Series.")
    clean = prices.astype(float).dropna()
    if clean.empty or not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean <= 0).any():
        raise MLForecastingError("prices must be finite and strictly positive.")
    if not clean.index.is_monotonic_increasing:
        raise MLForecastingError("prices index must be sorted.")
    return clean
