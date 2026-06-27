"""Temporal train/validation/test splits for financial time series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


class TimeSplitError(ValueError):
    """Raised when temporal split inputs are invalid."""


@dataclass(frozen=True)
class TimeSplit:
    """Non-overlapping temporal train/validation/test boundary labels."""

    train_start: object
    train_end: object
    validation_start: object
    validation_end: object
    test_start: object
    test_end: object


def _clean_index(index: Sequence[object] | pd.Index) -> pd.Index:
    clean_index = index if isinstance(index, pd.Index) else pd.Index(index)
    if clean_index.empty:
        raise TimeSplitError("index must not be empty.")
    if clean_index.has_duplicates:
        raise TimeSplitError("index must not contain duplicates.")
    if not clean_index.is_monotonic_increasing:
        raise TimeSplitError("index must be sorted in increasing order.")
    return clean_index


def _positive_size(value: int, name: str) -> int:
    clean_value = int(value)
    if clean_value <= 0:
        raise TimeSplitError(f"{name} must be > 0.")
    return clean_value


def _build_split(
    index: pd.Index,
    start: int,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> TimeSplit:
    train_start_pos = start
    train_end_pos = start + train_size - 1
    validation_start_pos = train_end_pos + 1
    validation_end_pos = validation_start_pos + validation_size - 1
    test_start_pos = validation_end_pos + 1
    test_end_pos = test_start_pos + test_size - 1
    return TimeSplit(
        train_start=index[train_start_pos],
        train_end=index[train_end_pos],
        validation_start=index[validation_start_pos],
        validation_end=index[validation_end_pos],
        test_start=index[test_start_pos],
        test_end=index[test_end_pos],
    )


def single_time_split(
    index: Sequence[object] | pd.Index,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> TimeSplit:
    """Return one chronological train/validation/test split without overlap."""

    clean_index = _clean_index(index)
    clean_train_size = _positive_size(train_size, "train_size")
    clean_validation_size = _positive_size(validation_size, "validation_size")
    clean_test_size = _positive_size(test_size, "test_size")
    required = clean_train_size + clean_validation_size + clean_test_size
    if len(clean_index) < required:
        raise TimeSplitError("Not enough observations for requested split sizes.")
    return _build_split(
        clean_index,
        start=0,
        train_size=clean_train_size,
        validation_size=clean_validation_size,
        test_size=clean_test_size,
    )


def walk_forward_splits(
    index: Sequence[object] | pd.Index,
    train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
) -> list[TimeSplit]:
    """Return chronological walk-forward windows with a fixed step size."""

    clean_index = _clean_index(index)
    clean_train_size = _positive_size(train_size, "train_size")
    clean_validation_size = _positive_size(validation_size, "validation_size")
    clean_test_size = _positive_size(test_size, "test_size")
    clean_step_size = _positive_size(step_size, "step_size")
    required = clean_train_size + clean_validation_size + clean_test_size
    if len(clean_index) < required:
        raise TimeSplitError("Not enough observations for requested walk-forward sizes.")

    splits = []
    start = 0
    while start + required <= len(clean_index):
        splits.append(
            _build_split(
                clean_index,
                start=start,
                train_size=clean_train_size,
                validation_size=clean_validation_size,
                test_size=clean_test_size,
            )
        )
        start += clean_step_size
    return splits
