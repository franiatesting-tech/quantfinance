# QuantLib Reference Notes

Status: external reference only, no runtime dependency.

Useful ideas for this platform:

- Explicit instrument, calendar, day-count, curve, and pricing-engine boundaries.
- Reproducible valuation settings rather than hidden global assumptions.
- Strong distinction between pricing models and market data inputs.

Current decision:

- Do not add QuantLib because pricing derivatives is outside the current foundation scope.
- Keep Black-Scholes, Heston, and advanced Monte Carlo as later-phase tasks.
- If pricing is introduced later, document calendars, day counts, model assumptions, and validation cases first.
