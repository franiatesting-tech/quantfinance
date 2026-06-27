from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.validation.overfitting_registry import (
    OverfittingRegistryError,
    StrategyTrialRecord,
    count_trials,
    load_strategy_trials,
    record_strategy_trial,
    summarize_trials,
)


def trial(trial_id: str = "trial-001", status: str = "COMPLETE") -> StrategyTrialRecord:
    return StrategyTrialRecord(
        trial_id=trial_id,
        strategy_name="equal_weight",
        hypothesis="Equal weight is a required benchmark.",
        parameters={"rebalance": "daily"},
        train_period={"start": "2024-01-01", "end": "2024-01-31"},
        validation_period={"start": "2024-02-01", "end": "2024-02-15"},
        test_period={"start": "2024-02-16", "end": "2024-02-29"},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        status=status,
        metrics={"sharpe_ratio": 1.0},
        notes="baseline",
    )


def test_record_and_load_strategy_trial(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "trials.jsonl"

    record_strategy_trial(trial(), path)
    records = load_strategy_trials(path)

    assert len(records) == 1
    assert records[0].trial_id == "trial-001"
    assert records[0].strategy_name == "equal_weight"


def test_count_trials(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "trials.jsonl"
    record_strategy_trial(trial("trial-001"), path)
    record_strategy_trial(trial("trial-002"), path)

    assert count_trials(path) == 2


def test_missing_hypothesis_fails(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "trials.jsonl"
    bad_record = StrategyTrialRecord(
        trial_id="bad",
        strategy_name="x",
        hypothesis="",
        parameters={},
        train_period={"start": "2024-01-01", "end": "2024-01-31"},
        validation_period={"start": "2024-02-01", "end": "2024-02-15"},
        test_period={"start": "2024-02-16", "end": "2024-02-29"},
        created_at="2024-01-01T00:00:00Z",
        status="COMPLETE",
        metrics={},
        notes="",
    )

    with pytest.raises(OverfittingRegistryError, match="hypothesis"):
        record_strategy_trial(bad_record, path)


def test_append_does_not_overwrite_records(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "trials.jsonl"
    record_strategy_trial(trial("trial-001"), path)
    record_strategy_trial(trial("trial-002"), path)

    records = load_strategy_trials(path)
    assert [record.trial_id for record in records] == ["trial-001", "trial-002"]


def test_summarize_trials_counts_by_strategy_and_status(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "trials.jsonl"
    record_strategy_trial(trial("trial-001", status="COMPLETE"), path)
    record_strategy_trial(trial("trial-002", status="DRAFT"), path)

    summary = summarize_trials(path)

    assert summary["total_trials"] == 2
    assert summary["by_strategy"] == {"equal_weight": 2}
    assert summary["by_status"] == {"COMPLETE": 1, "DRAFT": 1}


def test_draft_allows_missing_validation_and_test_period_with_note(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "trials.jsonl"
    draft = StrategyTrialRecord(
        trial_id="draft",
        strategy_name="momentum",
        hypothesis="Momentum may be useful as a benchmark.",
        parameters={"lookback": 3},
        train_period={"start": "2024-01-01", "end": "2024-01-31"},
        validation_period=None,
        test_period=None,
        created_at="2024-01-01T00:00:00Z",
        status="DRAFT",
        metrics={},
        notes="needs periods",
    )

    record_strategy_trial(draft, path)
    loaded = load_strategy_trials(path)[0]

    assert "DRAFT only" in loaded.notes
