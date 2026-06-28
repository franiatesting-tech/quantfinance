"""Responsible walk-forward forecasting diagnostics for financial returns."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtesting.metrics import hit_rate, sharpe_ratio
from quant_platform.research.returns import daily_simple_returns

MODEL_NOT_BETTER = "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE"
MODEL_OUTPERFORMS = "MODEL_OUTPERFORMS_NAIVE_UNDER_TEST_ASSUMPTIONS"
BASELINES_ONLY = "SKLEARN_UNAVAILABLE_BASELINES_ONLY"
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class MLForecastingError(ValueError):
    """Raised when ML forecasting inputs are invalid."""


def run_walk_forward_forecast(
    prices: pd.Series,
    volume: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    min_train_size: int = 252,
    horizon_days: int = 1,
    max_test_observations: int = 252,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """Run leakage-aware walk-forward baselines and optional sklearn models.

    The function never shuffles observations. For each test timestamp, training data
    ends strictly before the tested target observation.
    """

    if min_train_size < 30:
        raise MLForecastingError("min_train_size must be >= 30.")
    if horizon_days < 1 or horizon_days > 21:
        raise MLForecastingError("horizon_days must be in [1, 21].")
    if max_test_observations < 10:
        raise MLForecastingError("max_test_observations must be >= 10.")
    clean_prices = _as_price_series(prices)
    returns = daily_simple_returns(clean_prices)
    frame = _feature_frame(returns, volume=volume, benchmark_returns=benchmark_returns)
    frame["target_next_return"] = returns.shift(-horizon_days).reindex(frame.index)
    frame["target_direction"] = (frame["target_next_return"] > 0.0).astype(float)
    frame = frame.dropna(how="any")
    if len(frame) <= min_train_size + 10:
        return _insufficient_payload(len(frame), min_train_size)

    rows: list[dict[str, float | str]] = []
    feature_columns = [column for column in frame.columns if column.startswith("feature_")]
    first_test_pos = max(min_train_size, len(frame) - max_test_observations)
    for test_pos in range(first_test_pos, len(frame)):
        train = frame.iloc[:test_pos]
        test = frame.iloc[test_pos]
        actual = float(test["target_next_return"])
        naive_pred = 0.0
        hist_mean_pred = float(train["target_next_return"].mean())
        selected_pred = hist_mean_pred
        model_name = "historical_mean_baseline"
        sklearn_pred = (
            _optional_sklearn_prediction(train, test, feature_columns)
            if SKLEARN_AVAILABLE
            else None
        )
        if sklearn_pred is not None:
            selected_pred = sklearn_pred
            model_name = "ridge_regression_optional_sklearn"
        rows.append(
            {
                "timestamp": pd.Timestamp(test.name).isoformat(),
                "actual_return": actual,
                "naive_random_walk_prediction": naive_pred,
                "historical_mean_prediction": hist_mean_pred,
                "model_prediction": selected_pred,
                "model_name": model_name,
            }
        )

    result = pd.DataFrame(rows)
    actuals = result["actual_return"].to_numpy(dtype=float)
    model_preds = result["model_prediction"].to_numpy(dtype=float)
    naive_preds = result["naive_random_walk_prediction"].to_numpy(dtype=float)
    model_metrics = _regression_and_direction_metrics(actuals, model_preds, periods_per_year)
    naive_metrics = _regression_and_direction_metrics(actuals, naive_preds, periods_per_year)
    status = _model_status(model_metrics, naive_metrics)
    sklearn_used = bool((result["model_name"] == "ridge_regression_optional_sklearn").any())
    if not sklearn_used:
        status_detail = BASELINES_ONLY
    else:
        status_detail = "SKLEARN_OPTIONAL_MODEL_USED"
    return {
        "status": status,
        "model_status": status,
        "status_detail": status_detail,
        "validation": "walk_forward_no_shuffle_no_leakage",
        "target": "next_day_return" if horizon_days == 1 else f"{horizon_days}_day_forward_return",
        "horizon_days": int(horizon_days),
        "train_min_observations": int(min_train_size),
        "max_test_observations": int(max_test_observations),
        "test_observations": int(len(result)),
        "features": feature_columns,
        "models_evaluated": [
            "naive_random_walk_baseline",
            "historical_mean_baseline",
            "ridge_regression_optional_sklearn" if sklearn_used else "sklearn_models_not_available",
        ],
        "rmse": model_metrics["rmse"],
        "mae": model_metrics["mae"],
        "directional_accuracy": model_metrics["directional_accuracy"],
        "precision_up": model_metrics["precision_up"],
        "recall_up": model_metrics["recall_up"],
        "information_coefficient": model_metrics["information_coefficient"],
        "strategy_sharpe": model_metrics["strategy_sharpe"],
        "hit_rate": model_metrics["hit_rate"],
        "baseline_rmse": naive_metrics["rmse"],
        "baseline_mae": naive_metrics["mae"],
        "baseline_directional_accuracy": naive_metrics["directional_accuracy"],
        "calibration_warning": _calibration_warning(model_metrics, naive_metrics),
        "prediction_rows": rows[-252:],
        "not_investment_advice": True,
    }


def _feature_frame(
    returns: pd.Series,
    volume: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=returns.index)
    frame["feature_lag_1_return"] = returns.shift(1)
    frame["feature_lag_5_return"] = returns.shift(5)
    frame["feature_rolling_mean_21"] = returns.rolling(21).mean()
    frame["feature_rolling_volatility_21"] = returns.rolling(21).std(ddof=1)
    frame["feature_momentum_63"] = (1.0 + returns).rolling(63).apply(np.prod, raw=True) - 1.0
    equity = (1.0 + returns).cumprod()
    frame["feature_drawdown"] = equity / equity.cummax() - 1.0
    rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std(ddof=1)
    frame["feature_rolling_sharpe_63"] = rolling_sharpe * np.sqrt(252)
    if volume is not None:
        aligned_volume = volume.astype(float).reindex(returns.index)
        frame["feature_volume_change"] = aligned_volume.pct_change()
    if benchmark_returns is not None:
        asset, benchmark = returns.align(benchmark_returns.astype(float), join="left")
        rolling_cov = asset.rolling(63).cov(benchmark)
        rolling_var = benchmark.rolling(63).var(ddof=1)
        frame["feature_rolling_beta_63"] = rolling_cov / rolling_var
    return frame.replace([np.inf, -np.inf], np.nan)


def _optional_sklearn_prediction(
    train: pd.DataFrame,
    test: pd.Series,
    feature_columns: list[str],
) -> float | None:
    try:
        from sklearn.linear_model import Ridge  # type: ignore[import-not-found]
        from sklearn.pipeline import make_pipeline  # type: ignore[import-not-found]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
    except Exception:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[feature_columns], train["target_next_return"])
    return float(model.predict(pd.DataFrame([test[feature_columns]], columns=feature_columns))[0])


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
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "directional_accuracy": float(np.mean(predicted_up == actual_up)),
        "precision_up": true_positive / predicted_positive if predicted_positive else 0.0,
        "recall_up": true_positive / actual_positive if actual_positive else 0.0,
        "information_coefficient": ic,
        "strategy_sharpe": float(sharpe_ratio(pd.Series(strategy_returns), 0.0, periods_per_year)),
        "hit_rate": float(hit_rate(pd.Series(strategy_returns))),
    }


def _model_status(model: dict[str, float], naive: dict[str, float]) -> str:
    rmse_better = model["rmse"] < naive["rmse"]
    direction_edge = model["directional_accuracy"] - naive["directional_accuracy"]
    if rmse_better and direction_edge >= 0.02:
        return MODEL_OUTPERFORMS
    return MODEL_NOT_BETTER


def _calibration_warning(model: dict[str, float], naive: dict[str, float]) -> str | None:
    if model["rmse"] >= naive["rmse"]:
        return "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE"
    if model["directional_accuracy"] < 0.52:
        return "DIRECTIONAL_ACCURACY_EDGE_WEAK"
    return None


def _insufficient_payload(observations: int, min_train_size: int) -> dict[str, Any]:
    return {
        "status": "INSUFFICIENT_DATA",
        "model_status": "INSUFFICIENT_DATA",
        "observations": int(observations),
        "min_train_size": int(min_train_size),
        "validation": "walk_forward_no_shuffle_no_leakage",
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
