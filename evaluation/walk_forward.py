"""Walk-forward validation utilities for time-ordered experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class WalkForwardSplit:
    """A single walk-forward split with train, validation, and test windows."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_id: int


def generate_walk_forward_splits(
    frame: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
) -> list[WalkForwardSplit]:
    """Create strictly ordered walk-forward windows without overlap leakage."""

    splits: list[WalkForwardSplit] = []
    split_id = 0

    start = 0
    total_window = train_size + val_size + test_size
    while start + total_window <= len(frame):
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        splits.append(
            WalkForwardSplit(
                train=frame.iloc[start:train_end].copy(),
                validation=frame.iloc[train_end:val_end].copy(),
                test=frame.iloc[val_end:test_end].copy(),
                split_id=split_id,
            )
        )
        split_id += 1
        start += step_size

    if not splits:
        raise ValueError("No walk-forward splits could be created. Reduce window sizes or add more data.")

    return splits

