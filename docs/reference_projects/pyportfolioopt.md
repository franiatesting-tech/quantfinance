# PyPortfolioOpt Reference Notes

Status: external reference only, no runtime dependency.

Useful ideas for this platform:

- Portfolio APIs should expose expected returns, covariance estimates, constraints, objectives, and clean weight outputs separately.
- Baselines and diagnostics should exist before optimizer promotion.
- Weight cleaning and bounds need explicit policy.

Current decision:

- Do not add PyPortfolioOpt or CVXPY.
- Iteration 002 implements only non-optimized baselines: equal weight, inverse volatility, buy-and-hold, and simple momentum weights.
- Mean-variance and CVaR optimization remain blocked until covariance validation and solver policy are documented.
