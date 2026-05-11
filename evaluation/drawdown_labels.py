"""Utilities for future drawdown-risk label construction and diagnostics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.walk_forward import WalkForwardSplit


def _format_threshold_token(threshold: float) -> str:
    """Format a drawdown threshold for stable column naming."""

    pct = abs(float(threshold) * 100.0)
    if float(pct).is_integer():
        token = str(int(pct))
    else:
        token = f"{pct:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{token}pct"


def _label_name(horizon: int, threshold: float) -> str:
    return f"target_drawdown_event_{int(horizon)}d_{_format_threshold_token(threshold)}"


def compute_future_max_drawdown(
    prices,
    *,
    horizon: int,
    name: str | None = None,
) -> pd.Series:
    """Compute forward max drawdown over the next ``horizon`` bars from prices.

    For each row ``t``, this returns the minimum forward return observed between
    ``t+1`` and ``t+horizon`` relative to the price at ``t``.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    series = pd.Series(prices, dtype=float)
    values = series.to_numpy(dtype=float)
    result = np.full(len(series), np.nan, dtype=float)

    for idx in range(len(values)):
        end = min(len(values), idx + horizon + 1)
        if idx + 1 >= end:
            continue
        start_price = values[idx]
        if np.isnan(start_price) or start_price == 0.0:
            continue
        window = values[idx + 1 : end]
        forward_returns = window / start_price - 1.0
        if np.all(np.isnan(forward_returns)):
            continue
        result[idx] = np.nanmin(forward_returns)

    column_name = name or f"future_max_drawdown_{int(horizon)}d"
    return pd.Series(result, index=series.index, name=column_name, dtype=float)


def build_drawdown_event_labels(
    prices,
    *,
    horizon: int,
    threshold: float,
    positive_label: int = 1,
    negative_label: int = 0,
) -> pd.Series:
    """Build a nullable binary drawdown-event label from future max drawdown."""

    if threshold >= 0.0:
        raise ValueError("threshold must be negative for drawdown events.")

    future_drawdown = compute_future_max_drawdown(prices, horizon=horizon)
    label_values: list[int | pd._libs.missing.NAType] = []
    for value in future_drawdown:
        if pd.isna(value):
            label_values.append(pd.NA)
        elif float(value) <= float(threshold):
            label_values.append(int(positive_label))
        else:
            label_values.append(int(negative_label))

    return pd.Series(
        pd.array(label_values, dtype="Int64"),
        index=future_drawdown.index,
        name=_label_name(horizon, threshold),
    )


def build_drawdown_label_grid(
    prices,
    *,
    horizons: Iterable[int] = (10, 20),
    thresholds: Iterable[float] = (-0.03, -0.05),
) -> pd.DataFrame:
    """Build a frame with future max drawdown helpers and candidate event labels."""

    horizon_values = tuple(int(h) for h in horizons)
    threshold_values = tuple(float(t) for t in thresholds)
    if not horizon_values:
        raise ValueError("At least one horizon is required.")
    if not threshold_values:
        raise ValueError("At least one threshold is required.")

    series = pd.Series(prices, dtype=float)
    frame = pd.DataFrame(index=series.index)

    for horizon in dict.fromkeys(horizon_values):
        frame[f"future_max_drawdown_{horizon}d"] = compute_future_max_drawdown(series, horizon=horizon)

    for horizon in horizon_values:
        for threshold in threshold_values:
            frame[_label_name(horizon, threshold)] = build_drawdown_event_labels(
                series,
                horizon=horizon,
                threshold=threshold,
            )

    return frame


def get_drawdown_label_columns(frame: pd.DataFrame) -> list[str]:
    """Return discovered drawdown-event label columns in stable sorted order."""

    return sorted([column for column in frame.columns if str(column).startswith("target_drawdown_event_")])


def append_drawdown_label_grid(
    frame: pd.DataFrame,
    *,
    price_column: str = "Adj Close",
    horizons: Iterable[int] = (10, 20),
    thresholds: Iterable[float] = (-0.03, -0.05),
) -> pd.DataFrame:
    """Return a copy of ``frame`` with drawdown helpers and label columns appended."""

    if price_column not in frame.columns:
        raise KeyError(f"Missing price column {price_column!r}.")
    label_grid = build_drawdown_label_grid(
        frame[price_column],
        horizons=horizons,
        thresholds=thresholds,
    )
    enriched = frame.copy()
    for column in label_grid.columns:
        enriched[column] = label_grid[column]
    return enriched


def summarize_label_prevalence(
    labels,
    *,
    label_name: str | None = None,
) -> pd.Series:
    """Return prevalence diagnostics for one binary candidate label."""

    series = pd.Series(labels)
    valid = series.dropna()
    positive_count = int((valid.astype(float) == 1.0).sum()) if not valid.empty else 0
    negative_count = int((valid.astype(float) == 0.0).sum()) if not valid.empty else 0
    valid_rows = int(valid.shape[0])
    total_rows = int(series.shape[0])
    missing_count = total_rows - valid_rows
    positive_rate = float(positive_count / valid_rows) if valid_rows else np.nan
    negative_rate = float(negative_count / valid_rows) if valid_rows else np.nan
    missing_rate = float(missing_count / total_rows) if total_rows else np.nan

    return pd.Series(
        {
            "label": label_name or getattr(series, "name", None) or "label",
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
            "negative_rate": negative_rate,
            "all_one_class": bool(valid_rows > 0 and (positive_count == 0 or negative_count == 0)),
        }
    )


def summarize_label_grid_prevalence(
    frame: pd.DataFrame,
    *,
    label_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize prevalence across multiple candidate label columns."""

    columns = list(label_columns) if label_columns is not None else [c for c in frame.columns if c.startswith("target_")]
    if not columns:
        raise ValueError("No label columns were provided or discovered.")

    return pd.DataFrame(
        [summarize_label_prevalence(frame[column], label_name=column).to_dict() for column in columns]
    ).sort_values(["positive_rate", "label"], na_position="last").reset_index(drop=True)


def summarize_split_label_viability(
    splits: list[WalkForwardSplit],
    *,
    label_column: str,
    min_positive_count: int = 5,
    min_negative_count: int = 5,
    min_positive_rate: float = 0.01,
    max_positive_rate: float = 0.99,
) -> pd.DataFrame:
    """Return per-split, per-slice viability diagnostics for one label column."""

    rows: list[dict[str, object]] = []
    for split in splits:
        for slice_name, frame in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            if label_column not in frame.columns:
                raise KeyError(f"Missing label column {label_column!r} in split {split.split_id} {slice_name} slice.")
            summary = summarize_label_prevalence(frame[label_column], label_name=label_column)
            positive_rate = summary["positive_rate"]
            within_bounds = bool(
                pd.notna(positive_rate) and float(min_positive_rate) <= float(positive_rate) <= float(max_positive_rate)
            )
            has_min_positive_count = int(summary["positive_count"]) >= int(min_positive_count)
            has_min_negative_count = int(summary["negative_count"]) >= int(min_negative_count)
            is_viable = bool(within_bounds and has_min_positive_count and has_min_negative_count)
            rows.append(
                {
                    "split_id": split.split_id,
                    "slice_name": slice_name,
                    **summary.to_dict(),
                    "min_positive_count_required": int(min_positive_count),
                    "min_negative_count_required": int(min_negative_count),
                    "min_positive_rate_required": float(min_positive_rate),
                    "max_positive_rate_required": float(max_positive_rate),
                    "has_min_positive_count": has_min_positive_count,
                    "has_min_negative_count": has_min_negative_count,
                    "within_rate_bounds": within_bounds,
                    "is_viable": is_viable,
                }
            )
    return pd.DataFrame(rows).sort_values(["split_id", "slice_name"]).reset_index(drop=True)


def summarize_candidate_label_grid_viability(
    splits: list[WalkForwardSplit],
    *,
    label_columns: Iterable[str],
    min_positive_count: int = 5,
    min_negative_count: int = 5,
    min_positive_rate: float = 0.01,
    max_positive_rate: float = 0.99,
) -> pd.DataFrame:
    """Aggregate walk-forward viability diagnostics across multiple label columns."""

    rows: list[dict[str, object]] = []
    for label_column in label_columns:
        detail = summarize_split_label_viability(
            splits,
            label_column=label_column,
            min_positive_count=min_positive_count,
            min_negative_count=min_negative_count,
            min_positive_rate=min_positive_rate,
            max_positive_rate=max_positive_rate,
        )
        for slice_name, subset in detail.groupby("slice_name", sort=False):
            subset = subset.reset_index(drop=True)
            rows.append(
                {
                    "label": label_column,
                    "slice_name": slice_name,
                    "num_splits": int(len(subset)),
                    "num_viable_splits": int(subset["is_viable"].sum()),
                    "fraction_viable_splits": float(subset["is_viable"].mean()) if len(subset) else np.nan,
                    "mean_positive_rate": float(subset["positive_rate"].mean()) if len(subset) else np.nan,
                    "min_positive_rate": float(subset["positive_rate"].min()) if len(subset) else np.nan,
                    "max_positive_rate": float(subset["positive_rate"].max()) if len(subset) else np.nan,
                    "mean_positive_count": float(subset["positive_count"].mean()) if len(subset) else np.nan,
                    "min_positive_count": int(subset["positive_count"].min()) if len(subset) else 0,
                    "mean_negative_count": float(subset["negative_count"].mean()) if len(subset) else np.nan,
                    "min_negative_count": int(subset["negative_count"].min()) if len(subset) else 0,
                    "all_splits_viable": bool(len(subset) > 0 and subset["is_viable"].all()),
                }
            )
    return pd.DataFrame(rows).sort_values(["label", "slice_name"]).reset_index(drop=True)


def evaluate_candidate_drawdown_labels(
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    *,
    label_columns: Iterable[str] | None = None,
    min_positive_count: int = 5,
    min_negative_count: int = 5,
    min_positive_rate: float = 0.01,
    max_positive_rate: float = 0.99,
) -> dict[str, pd.DataFrame]:
    """Return combined prevalence and walk-forward viability summaries."""

    discovered_columns = list(label_columns) if label_columns is not None else get_drawdown_label_columns(frame)
    if not discovered_columns:
        raise ValueError("No drawdown label columns were provided or discovered.")

    prevalence = summarize_label_grid_prevalence(frame, label_columns=discovered_columns)
    viability = summarize_candidate_label_grid_viability(
        splits,
        label_columns=discovered_columns,
        min_positive_count=min_positive_count,
        min_negative_count=min_negative_count,
        min_positive_rate=min_positive_rate,
        max_positive_rate=max_positive_rate,
    )

    merged = prevalence.merge(
        viability[viability["slice_name"] == "test"].drop(columns=["slice_name"]),
        how="left",
        left_on="label",
        right_on="label",
    ).sort_values(["fraction_viable_splits", "positive_rate", "label"], ascending=[False, True, True], na_position="last")

    return {
        "prevalence": prevalence,
        "viability": viability,
        "candidate_summary": merged.reset_index(drop=True),
    }
