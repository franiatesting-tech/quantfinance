"""Local JSONL registry for strategy trials and overfitting control."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class OverfittingRegistryError(ValueError):
    """Raised when a strategy trial record is invalid."""


@dataclass(frozen=True)
class StrategyTrialRecord:
    """One auditable strategy trial record."""

    trial_id: str
    strategy_name: str
    hypothesis: str
    parameters: dict[str, Any]
    train_period: Any
    validation_period: Any
    test_period: Any
    created_at: str | datetime
    status: str
    metrics: dict[str, Any]
    notes: str = ""


def _period_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == () or value == {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _record_to_dict(record: StrategyTrialRecord) -> dict[str, Any]:
    if not record.hypothesis or not record.hypothesis.strip():
        raise OverfittingRegistryError("hypothesis must be provided before recording a trial.")
    if _period_missing(record.validation_period) or _period_missing(record.test_period):
        if record.status != "DRAFT":
            raise OverfittingRegistryError(
                "validation_period and test_period are required unless status='DRAFT'."
            )

    values = asdict(record)
    if _period_missing(record.validation_period) or _period_missing(record.test_period):
        note = "Missing validation/test period: DRAFT only."
        values["notes"] = f"{record.notes} {note}".strip()
    return _json_ready(values)


def record_strategy_trial(record: StrategyTrialRecord, path: str | Path) -> None:
    """Append one strategy trial as a JSONL line without overwriting prior trials."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = _record_to_dict(record)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, sort_keys=True) + "\n")


def load_strategy_trials(path: str | Path) -> list[StrategyTrialRecord]:
    """Load strategy trial records from a JSONL file."""

    source = Path(path)
    if not source.exists():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = json.loads(line)
        records.append(StrategyTrialRecord(**values))
    return records


def count_trials(path: str | Path) -> int:
    """Return the number of recorded strategy trials."""

    return len(load_strategy_trials(path))


def summarize_trials(path: str | Path) -> dict[str, Any]:
    """Summarize trial counts by strategy and status."""

    records = load_strategy_trials(path)
    by_strategy: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for record in records:
        by_strategy[record.strategy_name] = by_strategy.get(record.strategy_name, 0) + 1
        by_status[record.status] = by_status.get(record.status, 0) + 1
    return {
        "total_trials": len(records),
        "by_strategy": by_strategy,
        "by_status": by_status,
    }
