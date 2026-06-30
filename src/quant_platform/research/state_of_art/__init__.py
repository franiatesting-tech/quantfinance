"""State-of-the-art research diagnostics for final institutional studies."""

from quant_platform.research.state_of_art.covariance_shrinkage import (
    covariance_shrinkage_comparison,
    diagonal_shrinkage_covariance,
    ledoit_wolf_or_diagonal_covariance,
)
from quant_platform.research.state_of_art.execution_costs import (
    execution_cost_sensitivity,
    turnover_between_weights,
)
from quant_platform.research.state_of_art.factor_models import (
    FACTOR_MODEL_SPECS,
    factor_model_gap_table,
    fit_factor_model,
)
from quant_platform.research.state_of_art.final_research_gates import (
    RESEARCH_ONLY_DISCLAIMER,
    final_research_decision,
)
from quant_platform.research.state_of_art.model_confidence import evaluate_model_confidence
from quant_platform.research.state_of_art.multiple_testing import (
    deflated_sharpe_ratio_approximation,
    probabilistic_sharpe_ratio,
)
from quant_platform.research.state_of_art.portfolio_robustness import (
    portfolio_robustness_summary,
)
from quant_platform.research.state_of_art.tail_risk_backtesting import backtest_var

__all__ = [
    "FACTOR_MODEL_SPECS",
    "RESEARCH_ONLY_DISCLAIMER",
    "backtest_var",
    "covariance_shrinkage_comparison",
    "deflated_sharpe_ratio_approximation",
    "diagonal_shrinkage_covariance",
    "evaluate_model_confidence",
    "execution_cost_sensitivity",
    "factor_model_gap_table",
    "final_research_decision",
    "fit_factor_model",
    "ledoit_wolf_or_diagonal_covariance",
    "portfolio_robustness_summary",
    "probabilistic_sharpe_ratio",
    "turnover_between_weights",
]
