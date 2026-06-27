# VaR Backtesting

Iteration 003 adds a minimal VaR exception module in `src/quant_platform/validation/var_backtesting.py`.

## Conventions

- Inputs are positive losses: `L_t = -r_{p,t}` after clipping or transforming consistently upstream.
- VaR forecasts are non-negative and must use the same horizon as losses.
- `alpha` is the confidence level, so the expected exception rate is `1 - alpha`.

## Implemented Formulas

Exception indicator:

```text
I_t = 1{L_t > VaR_t}
```

Observed exception rate:

```text
mean(I_t)
```

Expected exception rate:

```text
1 - alpha
```

Exact binomial probability:

```text
P(K = k) = C(n,k) p^k (1-p)^(n-k)
```

The optional two-sided p-value is a conservative exact implementation that sums outcomes whose exact probability is less than or equal to the observed outcome probability.

## Reference Status

`BIS_1996_Backtesting_Internal_Models_Market_Risk.pdf` remains the regulatory reference for later traffic-light and independence testing. Iteration 003 only covers exceptions and basic coverage mechanics without SciPy.
