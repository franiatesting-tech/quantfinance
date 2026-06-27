from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.validation.splits import (
    TimeSplitError,
    single_time_split,
    walk_forward_splits,
)


def test_single_time_split_uses_chronological_boundaries() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")

    split = single_time_split(index, train_size=4, validation_size=3, test_size=2)

    assert split.train_start == index[0]
    assert split.train_end == index[3]
    assert split.validation_start == index[4]
    assert split.validation_end == index[6]
    assert split.test_start == index[7]
    assert split.test_end == index[8]
    assert split.train_end < split.validation_start <= split.validation_end < split.test_start


def test_walk_forward_splits_returns_multiple_windows() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")

    splits = walk_forward_splits(index, train_size=3, validation_size=2, test_size=2, step_size=2)

    assert len(splits) == 2
    assert splits[0].train_start == index[0]
    assert splits[1].train_start == index[2]


def test_split_rejects_unsorted_index() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"], utc=True)

    with pytest.raises(TimeSplitError, match="sorted"):
        single_time_split(index, train_size=1, validation_size=1, test_size=1)


def test_split_rejects_duplicate_index() -> None:
    index = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True)

    with pytest.raises(TimeSplitError, match="duplicates"):
        single_time_split(index, train_size=1, validation_size=1, test_size=1)


def test_split_rejects_invalid_sizes() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")

    with pytest.raises(TimeSplitError, match="train_size"):
        single_time_split(index, train_size=0, validation_size=1, test_size=1)


def test_split_rejects_insufficient_data() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")

    with pytest.raises(TimeSplitError, match="Not enough observations"):
        walk_forward_splits(index, train_size=2, validation_size=1, test_size=1, step_size=1)
