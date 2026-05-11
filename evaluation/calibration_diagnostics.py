"""Calibration diagnostics for saved classifier OOF predictions."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


BinStrategy = Literal["quantile", "uniform"]


def assign_probability_bins(
    probabilities,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> pd.Series:
    """Assign probability bins for calibration analysis."""

    probs = pd.Series(probabilities, dtype=float)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if probs.empty:
        return pd.Series(dtype="Int64", name="bin_id")

    clipped = probs.clip(0.0, 1.0)
    if strategy == "quantile":
        try:
            bins = pd.qcut(clipped, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series(0, index=clipped.index, dtype="Int64")
    elif strategy == "uniform":
        bins = pd.cut(
            clipped,
            bins=np.linspace(0.0, 1.0, n_bins + 1),
            labels=False,
            include_lowest=True,
        )
    else:
        raise ValueError(f"Unsupported strategy {strategy!r}.")

    if not isinstance(bins, pd.Series):
        bins = pd.Series(bins, index=clipped.index)
    if bins.isna().all():
        bins = pd.Series(0, index=clipped.index, dtype="Int64")
    return bins.astype("Int64").rename("bin_id")


def build_calibration_table(
    y_true,
    probabilities,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> pd.DataFrame:
    """Return bin-level calibration summary."""

    truth = pd.Series(y_true).astype(float)
    probs = pd.Series(probabilities, index=truth.index).astype(float).clip(0.0, 1.0)
    bins = assign_probability_bins(probs, n_bins=n_bins, strategy=strategy)

    frame = pd.DataFrame({"target": truth, "prediction_probability": probs, "bin_id": bins})
    frame = frame.dropna(subset=["target", "prediction_probability", "bin_id"])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "bin_id",
                "count",
                "mean_predicted_probability",
                "observed_event_rate",
                "signed_gap",
                "abs_gap",
            ]
        )

    table = (
        frame.groupby("bin_id", observed=False)
        .agg(
            count=("target", "size"),
            mean_predicted_probability=("prediction_probability", "mean"),
            observed_event_rate=("target", "mean"),
        )
        .reset_index()
    )
    table["signed_gap"] = table["observed_event_rate"] - table["mean_predicted_probability"]
    table["abs_gap"] = table["signed_gap"].abs()
    return table


def expected_calibration_error(calibration_table: pd.DataFrame) -> float:
    """Return weighted absolute calibration gap."""

    if calibration_table.empty:
        return 0.0
    weights = calibration_table["count"] / calibration_table["count"].sum()
    return float((weights * calibration_table["abs_gap"]).sum())


def maximum_calibration_error(calibration_table: pd.DataFrame) -> float:
    """Return maximum absolute calibration gap across bins."""

    if calibration_table.empty:
        return 0.0
    return float(calibration_table["abs_gap"].max())


def summarize_probability_diagnostics(
    y_true,
    probabilities,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> dict[str, float]:
    """Return pooled probability diagnostics from saved OOF predictions."""

    truth = pd.Series(y_true).astype(int)
    probs = pd.Series(probabilities, index=truth.index).astype(float).clip(0.0, 1.0)
    table = build_calibration_table(truth, probs, n_bins=n_bins, strategy=strategy)

    try:
        ap = float(average_precision_score(truth, probs))
    except ValueError:
        ap = float("nan")
    try:
        auc = float(roc_auc_score(truth, probs)) if truth.nunique() > 1 else 0.5
    except ValueError:
        auc = 0.5

    return {
        "n_rows": float(len(truth)),
        "base_event_rate": float(truth.mean()) if len(truth) else float("nan"),
        "mean_prediction_probability": float(probs.mean()) if len(probs) else float("nan"),
        "prediction_probability_std": float(probs.std(ddof=0)) if len(probs) else float("nan"),
        "brier_score": float(brier_score_loss(truth, probs)) if len(truth) else float("nan"),
        "average_precision": ap,
        "auc_roc": auc,
        "expected_calibration_error": expected_calibration_error(table),
        "maximum_calibration_error": maximum_calibration_error(table),
        "num_bins_realized": float(len(table)),
    }


def summarize_fold_probability_diagnostics(
    frame: pd.DataFrame,
    *,
    model_name: str,
    split_column: str = "split_id",
    target_column: str = "target",
    probability_column: str = "prediction_probability",
    n_bins: int = 10,
    strategy: BinStrategy = "quantile",
) -> pd.DataFrame:
    """Return per-fold probability diagnostics."""

    if split_column not in frame.columns:
        raise KeyError(f"Missing split column {split_column!r}.")
    rows: list[dict[str, float | str]] = []
    for split_id, subset in frame.groupby(split_column, sort=True):
        summary = summarize_probability_diagnostics(
            subset[target_column],
            subset[probability_column],
            n_bins=n_bins,
            strategy=strategy,
        )
        rows.append({"model_name": model_name, "split_id": split_id, **summary})
    return pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)
