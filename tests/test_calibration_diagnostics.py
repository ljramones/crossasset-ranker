from __future__ import annotations

import pandas as pd

from evaluation.calibration_diagnostics import (
    assign_probability_bins,
    build_calibration_table,
    expected_calibration_error,
    maximum_calibration_error,
    summarize_fold_probability_diagnostics,
    summarize_probability_diagnostics,
)


def test_assign_probability_bins_supports_quantile_and_uniform() -> None:
    probabilities = pd.Series([0.1, 0.2, 0.4, 0.7, 0.9])

    quantile_bins = assign_probability_bins(probabilities, n_bins=3, strategy="quantile")
    uniform_bins = assign_probability_bins(probabilities, n_bins=3, strategy="uniform")

    assert len(quantile_bins) == len(probabilities)
    assert len(uniform_bins) == len(probabilities)
    assert quantile_bins.notna().all()
    assert uniform_bins.notna().all()


def test_build_calibration_table_returns_bin_level_gaps() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1, 0])
    probabilities = pd.Series([0.1, 0.2, 0.4, 0.7, 0.8, 0.9])

    table = build_calibration_table(y_true, probabilities, n_bins=3, strategy="quantile")

    assert {"bin_id", "count", "mean_predicted_probability", "observed_event_rate", "signed_gap", "abs_gap"} == set(table.columns)
    assert table["count"].sum() == len(y_true)


def test_build_calibration_table_handles_constant_probabilities() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = pd.Series([0.4, 0.4, 0.4, 0.4])

    table = build_calibration_table(y_true, probabilities, n_bins=5, strategy="quantile")

    assert not table.empty
    assert table["count"].sum() == len(y_true)


def test_expected_and_maximum_calibration_error_are_non_negative() -> None:
    table = pd.DataFrame(
        {
            "count": [2, 2, 2],
            "abs_gap": [0.1, 0.2, 0.05],
        }
    )

    assert expected_calibration_error(table) > 0.0
    assert maximum_calibration_error(table) == 0.2


def test_summarize_probability_diagnostics_returns_expected_fields() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1, 0, 1, 0])
    probabilities = pd.Series([0.1, 0.2, 0.35, 0.7, 0.8, 0.65, 0.9, 0.3])

    summary = summarize_probability_diagnostics(y_true, probabilities, n_bins=4, strategy="quantile")

    assert {
        "n_rows",
        "base_event_rate",
        "mean_prediction_probability",
        "prediction_probability_std",
        "brier_score",
        "average_precision",
        "auc_roc",
        "expected_calibration_error",
        "maximum_calibration_error",
        "num_bins_realized",
    } == set(summary)


def test_summarize_fold_probability_diagnostics_returns_one_row_per_split() -> None:
    frame = pd.DataFrame(
        {
            "split_id": [0, 0, 0, 1, 1, 1],
            "target": [0, 1, 0, 1, 1, 0],
            "prediction_probability": [0.1, 0.8, 0.2, 0.7, 0.6, 0.3],
        }
    )

    summary = summarize_fold_probability_diagnostics(frame, model_name="logistic", n_bins=3, strategy="quantile")

    assert len(summary) == 2
    assert set(summary["split_id"]) == {0, 1}
    assert set(summary["model_name"]) == {"logistic"}
