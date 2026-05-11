"""Tests for walk-forward validation."""

from __future__ import annotations

import pandas as pd

from evaluation.walk_forward import generate_walk_forward_splits


def test_generate_walk_forward_splits_respects_ordering() -> None:
    frame = pd.DataFrame({"value": range(40)})
    splits = generate_walk_forward_splits(frame, train_size=10, val_size=5, test_size=5, step_size=5)

    first = splits[0]
    assert len(first.train) == 10
    assert len(first.validation) == 5
    assert len(first.test) == 5
    assert first.train.index.max() < first.validation.index.min()
    assert first.validation.index.max() < first.test.index.min()

