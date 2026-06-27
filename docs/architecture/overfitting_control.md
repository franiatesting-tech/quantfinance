# Overfitting Control

Iteration 003 adds a minimal local registry in `src/quant_platform/validation/overfitting_registry.py`.

## Purpose

The registry records every strategy trial so research does not silently select the best historical curve after many unreported attempts.

## Implemented Record

`StrategyTrialRecord` stores:

- `trial_id`
- `strategy_name`
- `hypothesis`
- `parameters`
- `train_period`
- `validation_period`
- `test_period`
- `created_at`
- `status`
- `metrics`
- `notes`

Records are appended to JSONL with `record_strategy_trial`. Existing records are not overwritten.

## Rules

- A hypothesis is mandatory.
- Missing validation or test periods are allowed only for `status="DRAFT"`.
- Draft records with missing validation/test periods are marked in notes.
- `summarize_trials` reports counts by strategy and status.

## Bibliographic Anchors

- Koshiyama/Firoozye: backtesting overfitting and covariance penalties.
- Rej: discounted backtest PnL.
- Gort et al.: crypto/DRL backtest overfitting risk.

No probabilistic overfitting adjustment is implemented yet; this is an audit trail foundation only.
