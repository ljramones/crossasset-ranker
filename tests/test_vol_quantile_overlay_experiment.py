"""Synthetic tests for the standalone volatility-quantile overlay runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.walk_forward import WalkForwardSplit
from experiments.vol_quantile_overlay_experiment import (
    build_fold_details_frame,
    build_matched_nulls_frame,
    run_fold_local_vol_quantile_overlay_experiment,
)
from scripts.run_vol_quantile_overlay_experiment import (
    _build_markdown_report,
    _build_output_paths,
    main as cli_main,
)


def _build_synthetic_split() -> tuple[pd.DataFrame, list[WalkForwardSplit]]:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "forward_simple_return_1d": [0.01, 0.012, -0.03, -0.025, -0.04, 0.02, 0.02, -0.03, 0.015, 0.02],
            "benchmark_return_1d": [0.002, 0.002, -0.004, -0.003, -0.006, 0.003, 0.003, -0.004, 0.002, 0.003],
            "target_direction": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        },
        index=index,
    )
    split = WalkForwardSplit(
        train=frame.iloc[:4].copy(),
        validation=frame.iloc[4:7].copy(),
        test=frame.iloc[7:10].copy(),
        split_id=0,
    )
    return frame, [split]


def test_vol_quantile_overlay_runner_selects_validation_parameters_and_reports_nulls() -> None:
    frame, splits = _build_synthetic_split()

    result = run_fold_local_vol_quantile_overlay_experiment(
        frame=frame,
        splits=splits,
        realized_vol_window=2,
        quantiles=(0.67, 0.8),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )

    split_result = result.split_results[0]
    top_row = split_result.validation_grid.iloc[0]
    assert split_result.selected_parameters.quantile == top_row["quantile"]
    assert split_result.selected_parameters.risk_multiplier == top_row["risk_multiplier"]
    assert "same_average_exposure_random" in split_result.matched_nulls["null_summaries"]
    assert "same_turnover_random" in split_result.matched_nulls["null_summaries"]
    assert "same_vol_state_exposure_random" in split_result.matched_nulls["null_summaries"]
    assert "overlay_vs_base_information_ratio" in result.summary.columns


def test_vol_quantile_overlay_runner_emits_audit_artifact_without_double_shift() -> None:
    frame, splits = _build_synthetic_split()

    result = run_fold_local_vol_quantile_overlay_experiment(
        frame=frame,
        splits=splits,
        realized_vol_window=2,
        quantiles=(0.67, 0.8),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )

    artifact = result.audit_artifact_frame
    required_columns = {
        "date",
        "split_id",
        "model_name",
        "asset_return",
        "benchmark_return",
        "raw_signal",
        "executed_position",
        "strategy_gross_return",
        "strategy_net_return",
        "turnover",
        "transaction_cost",
        "base_position",
        "vol_state",
        "high_vol_indicator",
        "vol_quantile",
        "vol_cutoff_value",
        "overlay_threshold_state",
        "overlay_risk_multiplier",
        "overlay_mode",
        "is_cut_day",
    }
    assert required_columns <= set(artifact.columns)
    assert artifact["executed_position"].tolist() == artifact["raw_signal"].tolist()


def test_vol_quantile_overlay_builds_fold_details_and_matched_null_frames() -> None:
    frame, splits = _build_synthetic_split()

    result = run_fold_local_vol_quantile_overlay_experiment(
        frame=frame,
        splits=splits,
        realized_vol_window=2,
        quantiles=(0.67, 0.8),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )
    fold_details = build_fold_details_frame(result)
    matched_nulls = build_matched_nulls_frame(result)

    assert len(fold_details) == 1
    assert {
        "split_id",
        "selected_quantile",
        "selected_cutoff_value",
        "selected_threshold_state",
        "selected_risk_multiplier",
        "decision_metric",
        "validation_score",
        "test_overlay_vs_base_information_ratio",
    } <= set(fold_details.columns)
    assert {
        "split_id",
        "null_name",
        "decision_metric",
        "canonical_value",
        "mean_null_value",
        "percentile_95_null_value",
        "p_value",
        "n_runs",
        "passes_p_value_gate",
    } <= set(matched_nulls.columns)
    assert "same_vol_state_exposure_random" in set(matched_nulls["null_name"])


def test_vol_quantile_overlay_markdown_report_contains_expected_sections() -> None:
    frame, splits = _build_synthetic_split()
    result = run_fold_local_vol_quantile_overlay_experiment(
        frame=frame,
        splits=splits,
        realized_vol_window=2,
        quantiles=(0.67, 0.8),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )
    report = _build_markdown_report(
        summary_frame=result.summary,
        fold_details_frame=build_fold_details_frame(result),
        matched_nulls_frame=build_matched_nulls_frame(result),
        metadata={
            "generated_at_utc": "20260510_120000",
            "input_csv": "/tmp/input.csv",
            "asset_name": "SPY",
            "n_splits": 1,
            "null_runs": 5,
            "decision_grade": False,
            "notes": ["synthetic"],
        },
    )

    assert "# Volatility-Quantile Overlay Experiment Report" in report
    assert "## Summary" in report
    assert "## Fold Details" in report
    assert "## Matched Null Diagnostics" in report
    assert "same_vol_state_exposure_random" in report


def test_vol_quantile_overlay_cli_dry_run_does_not_load_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("pd.read_csv should not be called in --dry-run mode.")

    monkeypatch.setattr("scripts.run_vol_quantile_overlay_experiment.pd.read_csv", _boom)

    result = cli_main(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result.name == "DRY_RUN_ONLY"
    assert list(tmp_path.iterdir()) == []


def test_vol_quantile_overlay_cli_execute_requires_runtime_args() -> None:
    with pytest.raises(SystemExit):
        cli_main(["--execute"])


def test_vol_quantile_overlay_output_paths_are_timestamped_and_complete(tmp_path: Path) -> None:
    paths_a = _build_output_paths(tmp_path, timestamp="20260510_120000")
    paths_b = _build_output_paths(tmp_path, timestamp="20260510_120001")

    assert paths_a["summary"] != paths_b["summary"]
    assert paths_a["audit_artifacts"] != paths_b["audit_artifacts"]
    assert set(paths_a) == {
        "summary",
        "fold_details",
        "audit_artifacts",
        "matched_nulls",
        "report_markdown",
        "metadata",
    }
