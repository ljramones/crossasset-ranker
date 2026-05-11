from __future__ import annotations

import pandas as pd

from evaluation.drawdown_labels import (
    append_drawdown_label_grid,
    build_drawdown_event_labels,
    build_drawdown_label_grid,
    compute_future_max_drawdown,
    evaluate_candidate_drawdown_labels,
    get_drawdown_label_columns,
    summarize_candidate_label_grid_viability,
    summarize_label_grid_prevalence,
    summarize_label_prevalence,
    summarize_split_label_viability,
)
from evaluation.walk_forward import generate_walk_forward_splits


def test_compute_future_max_drawdown_uses_forward_window_only() -> None:
    prices = pd.Series([100.0, 95.0, 97.0, 90.0, 110.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))

    drawdown = compute_future_max_drawdown(prices, horizon=2)

    assert abs(drawdown.iloc[0] - (-0.05)) < 1e-9
    assert abs(drawdown.iloc[1] - (90.0 / 95.0 - 1.0)) < 1e-9
    assert abs(drawdown.iloc[2] - (90.0 / 97.0 - 1.0)) < 1e-9
    assert abs(drawdown.iloc[3] - (110.0 / 90.0 - 1.0)) < 1e-9
    assert pd.isna(drawdown.iloc[4])


def test_build_drawdown_event_labels_returns_nullable_binary_series() -> None:
    prices = pd.Series([100.0, 95.0, 97.0, 90.0, 110.0])

    labels = build_drawdown_event_labels(prices, horizon=2, threshold=-0.05)

    assert labels.name == "target_drawdown_event_2d_5pct"
    assert str(labels.dtype) == "Int64"
    assert labels.iloc[:4].tolist() == [1, 1, 1, 0]
    assert pd.isna(labels.iloc[4])


def test_build_drawdown_label_grid_creates_continuous_and_binary_columns() -> None:
    prices = pd.Series([100.0, 101.0, 98.0, 96.0, 99.0, 97.0])

    grid = build_drawdown_label_grid(prices, horizons=(2, 3), thresholds=(-0.03, -0.05))

    assert "future_max_drawdown_2d" in grid.columns
    assert "future_max_drawdown_3d" in grid.columns
    assert "target_drawdown_event_2d_3pct" in grid.columns
    assert "target_drawdown_event_2d_5pct" in grid.columns
    assert "target_drawdown_event_3d_3pct" in grid.columns


def test_append_drawdown_label_grid_preserves_original_frame_and_adds_labels() -> None:
    frame = pd.DataFrame(
        {
            "Adj Close": [100.0, 101.0, 98.0, 96.0, 99.0, 97.0],
            "feature_x": [1, 2, 3, 4, 5, 6],
        }
    )

    enriched = append_drawdown_label_grid(frame, horizons=(2,), thresholds=(-0.03,))

    assert "feature_x" in enriched.columns
    assert "future_max_drawdown_2d" in enriched.columns
    assert "target_drawdown_event_2d_3pct" in enriched.columns
    assert "future_max_drawdown_2d" not in frame.columns


def test_summarize_label_prevalence_counts_positive_negative_and_missing() -> None:
    labels = pd.Series(pd.array([1, 0, 1, pd.NA, 0], dtype="Int64"), name="target_drawdown_event_10d_3pct")

    summary = summarize_label_prevalence(labels)

    assert summary["label"] == "target_drawdown_event_10d_3pct"
    assert summary["total_rows"] == 5
    assert summary["valid_rows"] == 4
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 2
    assert summary["missing_count"] == 1
    assert abs(float(summary["positive_rate"]) - 0.5) < 1e-9
    assert bool(summary["all_one_class"]) is False


def test_summarize_label_grid_prevalence_discovers_target_columns() -> None:
    frame = pd.DataFrame(
        {
            "target_drawdown_event_10d_3pct": pd.array([1, 0, pd.NA, 1], dtype="Int64"),
            "target_drawdown_event_20d_5pct": pd.array([0, 0, 0, pd.NA], dtype="Int64"),
            "other_column": [1, 2, 3, 4],
        }
    )

    summary = summarize_label_grid_prevalence(frame)

    assert set(summary["label"]) == {
        "target_drawdown_event_10d_3pct",
        "target_drawdown_event_20d_5pct",
    }


def test_get_drawdown_label_columns_returns_only_drawdown_event_targets() -> None:
    frame = pd.DataFrame(
        {
            "target_drawdown_event_20d_5pct": [0, 1],
            "target_direction": [1, 0],
            "target_drawdown_event_10d_3pct": [1, 0],
        }
    )

    columns = get_drawdown_label_columns(frame)

    assert columns == [
        "target_drawdown_event_10d_3pct",
        "target_drawdown_event_20d_5pct",
    ]


def test_summarize_split_label_viability_reports_slice_level_viability() -> None:
    frame = pd.DataFrame(
        {
            "target_drawdown_event_10d_3pct": pd.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype="Int64"),
        }
    )
    splits = generate_walk_forward_splits(frame, train_size=4, val_size=3, test_size=3, step_size=10)

    summary = summarize_split_label_viability(
        splits,
        label_column="target_drawdown_event_10d_3pct",
        min_positive_count=1,
        min_negative_count=1,
        min_positive_rate=0.1,
        max_positive_rate=0.9,
    )

    assert len(summary) == 3
    assert set(summary["slice_name"]) == {"train", "validation", "test"}
    assert summary["is_viable"].all()


def test_summarize_candidate_label_grid_viability_aggregates_by_label_and_slice() -> None:
    frame = pd.DataFrame(
        {
            "target_drawdown_event_10d_3pct": pd.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype="Int64"),
            "target_drawdown_event_20d_5pct": pd.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype="Int64"),
        }
    )
    splits = generate_walk_forward_splits(frame, train_size=4, val_size=3, test_size=3, step_size=10)

    summary = summarize_candidate_label_grid_viability(
        splits,
        label_columns=["target_drawdown_event_10d_3pct", "target_drawdown_event_20d_5pct"],
        min_positive_count=1,
        min_negative_count=1,
        min_positive_rate=0.1,
        max_positive_rate=0.9,
    )

    viable_train = summary.loc[
        (summary["label"] == "target_drawdown_event_10d_3pct") & (summary["slice_name"] == "train"),
        "all_splits_viable",
    ].iloc[0]
    sparse_test = summary.loc[
        (summary["label"] == "target_drawdown_event_20d_5pct") & (summary["slice_name"] == "test"),
        "all_splits_viable",
    ].iloc[0]

    assert bool(viable_train) is True
    assert bool(sparse_test) is False


def test_evaluate_candidate_drawdown_labels_returns_combined_tables() -> None:
    frame = pd.DataFrame(
        {
            "Adj Close": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 96.0, 97.0, 98.0, 99.0],
        }
    )
    enriched = append_drawdown_label_grid(frame, horizons=(2,), thresholds=(-0.02, -0.04))
    splits = generate_walk_forward_splits(enriched, train_size=4, val_size=3, test_size=3, step_size=10)

    result = evaluate_candidate_drawdown_labels(
        enriched,
        splits,
        min_positive_count=1,
        min_negative_count=1,
        min_positive_rate=0.1,
        max_positive_rate=0.9,
    )

    assert set(result) == {"prevalence", "viability", "candidate_summary"}
    assert not result["prevalence"].empty
    assert not result["viability"].empty
    assert "fraction_viable_splits" in result["candidate_summary"].columns
