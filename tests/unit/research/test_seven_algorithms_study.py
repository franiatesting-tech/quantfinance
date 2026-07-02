from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_platform.research import seven_algorithms_study
from quant_platform.research.seven_algorithms_study import write_seven_algorithm_study


def test_write_seven_algorithm_study_outputs_paper_and_metadata(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        seven_algorithms_study,
        "export_html_report_to_pdf",
        lambda html_path, pdf_path: {"success": False, "renderer": "test"},
    )
    pack = tmp_path / "papers_pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "filename": "01_deep_hedging_buehler_2019.pdf",
                    "title": "Deep Hedging",
                    "authors": "Buehler et al.",
                    "year": "2019",
                    "source": "arXiv",
                    "landing_url": "https://arxiv.org/abs/1802.03042",
                }
            ]
        ),
        encoding="utf-8",
    )

    metadata = write_seven_algorithm_study(
        _sample_terminal_report(),
        output_dir=tmp_path / "seven",
        bibliography_pack_dir=pack,
        strict_real_data=True,
    )

    study = json.loads((tmp_path / "seven" / "seven_quant_algorithms_study.json").read_text())
    assert metadata["research_only"] is True
    assert metadata["pdf_export"]["status"] == "PDF_EXPORT_UNAVAILABLE_INSTALL_RENDERER"
    assert (tmp_path / "seven" / "seven_quant_algorithms_paper.md").exists()
    assert (tmp_path / "seven" / "seven_quant_algorithms_paper.html").exists()
    assert len(study["tables"]["broker_readiness_decisions"]) == 7
    assert study["tables"]["bibliography_trace"][0]["algorithm_relevance"].startswith(
        "Black-Scholes"
    )


def _sample_terminal_report() -> dict[str, object]:
    rng = np.random.default_rng(23)
    index = pd.date_range("2020-01-01", periods=340, freq="B", tz="UTC")
    base = np.cumsum(rng.normal(0.0003, 0.01, size=len(index)))
    prices = {
        "AAA": 100.0 * np.exp(base + rng.normal(0.0, 0.01, len(index))),
        "BBB": 90.0 * np.exp(base + rng.normal(0.0, 0.01, len(index))),
        "CCC": 80.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index)))),
    }
    stocks = {}
    for asset, values in prices.items():
        series = pd.Series(values, index=index)
        returns = series.pct_change().dropna()
        stocks[asset] = {
            "price_series": [
                {"timestamp": ts.isoformat(), "price": float(value)}
                for ts, value in series.items()
            ],
            "simple_returns": [
                {"timestamp": ts.isoformat(), "return": float(value)}
                for ts, value in returns.items()
            ],
            "options_theoretical_analytics": {
                "spot": float(series.iloc[-1]),
                "strike": float(series.iloc[-1]),
                "volatility": 0.2,
                "black_scholes_call": 5.0,
                "black_scholes_put": 4.8,
                "put_call_parity_gap": 0.0,
                "model_status": "PARAMETRIC_EDUCATIONAL_MODEL",
                "call_greeks": {"delta": 0.5, "gamma": 0.01, "vega": 12.0},
                "payoff_profile": [
                    {"underlying_price": 80.0, "payoff": 0.0},
                    {"underlying_price": 100.0, "payoff": 0.0},
                    {"underlying_price": 120.0, "payoff": 20.0},
                ],
            },
            "monte_carlo": {
                "parametric_normal": _mc_payload(),
                "historical_bootstrap": _mc_payload(),
            },
        }
    return {
        "report_type": "professional_quant_terminal",
        "research_only": True,
        "warnings": [],
        "data": {
            "mode": "provider_test",
            "aligned_observations": len(index),
            "start_timestamp": index[0].isoformat(),
            "end_timestamp": index[-1].isoformat(),
            "warnings": [],
        },
        "stocks": stocks,
        "optimization": {
            "method": "long_only_grid_search",
            "grid_step": 0.5,
            "frontier": [
                {"annualized_return": 0.05, "annualized_volatility": 0.1, "sharpe_ratio": 0.5},
                {"annualized_return": 0.08, "annualized_volatility": 0.15, "sharpe_ratio": 0.53},
            ],
            "min_variance": {
                "annualized_return": 0.05,
                "annualized_volatility": 0.1,
                "sharpe_ratio": 0.5,
                "weights": {"AAA": 0.5, "BBB": 0.5, "CCC": 0.0},
            },
            "max_sharpe": {
                "annualized_return": 0.08,
                "annualized_volatility": 0.15,
                "sharpe_ratio": 0.53,
                "weights": {"AAA": 0.0, "BBB": 0.5, "CCC": 0.5},
            },
        },
        "ml_forecasting": {
            "AAA": {
                "status": "MODEL_NOT_BETTER_THAN_NAIVE_BASELINE",
                "test_observations": 50,
                "oos_r_squared": -0.1,
                "directional_accuracy": 0.51,
                "baseline_directional_accuracy": 0.5,
                "directional_accuracy_edge_vs_naive": 0.01,
                "information_coefficient": 0.02,
                "strategy_sharpe": 0.1,
                "status_reason": "weak edge",
                "prediction_rows": [
                    {"timestamp": "2021-01-01", "actual_return": 0.01, "model_prediction": 0.002}
                ],
            }
        },
    }


def _mc_payload() -> dict[str, object]:
    return {
        "path_count": 100,
        "horizon_days": 5,
        "terminal_mean": 1.01,
        "terminal_p05": 0.95,
        "terminal_p95": 1.07,
        "probability_of_loss": 0.45,
        "model_status": "PARAMETRIC_OR_BOOTSTRAP_SIMULATION",
        "fan_chart": [
            {"step": step, "p5": 0.95, "p25": 0.98, "p50": 1.0, "p75": 1.02, "p95": 1.05}
            for step in range(1, 6)
        ],
    }
