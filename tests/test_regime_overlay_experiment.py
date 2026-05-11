"""Synthetic tests for the standalone regime-overlay experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from evaluation.walk_forward import WalkForwardSplit
from experiments.regime_overlay_experiment import (
    build_fold_details_frame,
    build_matched_nulls_frame,
    run_fold_local_regime_overlay_experiment,
)
from regime.regime_detection import RegimePrediction
from scripts.run_regime_overlay_experiment import (
    _build_markdown_report,
    _build_output_paths,
    main as cli_main,
)


@dataclass
class _FakeDetector:
    fit_indices: list[list[pd.Timestamp]] = field(default_factory=list)
    predict_indices: list[list[pd.Timestamp]] = field(default_factory=list)
    predict_live_safe_indices: list[list[pd.Timestamp]] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> None:
        self.fit_indices.append(list(frame.index))

    def predict(self, frame: pd.DataFrame) -> RegimePrediction:
        self.predict_indices.append(list(frame.index))
        return self._build_prediction(frame)

    def predict_live_safe(self, frame: pd.DataFrame) -> RegimePrediction:
        self.predict_live_safe_indices.append(list(frame.index))
        return self._build_prediction(frame)

    @staticmethod
    def _build_prediction(frame: pd.DataFrame) -> RegimePrediction:
        labels = frame["expected_regime_id"].astype(int).rename("regime_id")
        probabilities = pd.DataFrame(
            {
                "regime_prob_0": frame["regime_prob_0"].astype(float),
                "regime_prob_1": frame["regime_prob_1"].astype(float),
                "regime_prob_2": frame["regime_prob_2"].astype(float),
            },
            index=frame.index,
        )
        return RegimePrediction(labels=labels, probabilities=probabilities)


def _build_synthetic_split() -> tuple[pd.DataFrame, list[WalkForwardSplit], _FakeDetector]:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "forward_simple_return_1d": [0.01, 0.012, -0.03, -0.025, -0.04, 0.02, 0.02, -0.03, 0.015, 0.02],
            "benchmark_return_1d": [0.002, 0.002, -0.004, -0.003, -0.006, 0.003, 0.003, -0.004, 0.002, 0.003],
            "target_direction": [1, 1, 0, 0, 0, 1, 1, 0, 1, 1],
            "expected_regime_id": [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            "regime_prob_0": [0.85, 0.80, 0.10, 0.10, 0.15, 0.80, 0.80, 0.05, 0.80, 0.80],
            "regime_prob_1": [0.10, 0.15, 0.85, 0.80, 0.80, 0.10, 0.10, 0.90, 0.10, 0.10],
            "regime_prob_2": [0.05] * 10,
        },
        index=index,
    )
    split = WalkForwardSplit(
        train=frame.iloc[:4].copy(),
        validation=frame.iloc[4:7].copy(),
        test=frame.iloc[7:10].copy(),
        split_id=0,
    )
    detector = _FakeDetector()
    return frame, [split], detector


def test_overlay_runner_uses_train_fit_and_live_safe_on_validation_and_test() -> None:
    frame, splits, detector = _build_synthetic_split()

    result = run_fold_local_regime_overlay_experiment(
        frame=frame,
        splits=splits,
        detector_factory=lambda: detector,
        realized_vol_window=2,
        thresholds=(0.5, 0.9),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )

    assert len(result.split_results) == 1
    assert detector.fit_indices == [list(splits[0].train.index)]
    assert detector.predict_indices[0] == list(splits[0].train.index)
    assert detector.predict_live_safe_indices == [
        list(splits[0].validation.index),
        list(splits[0].test.index),
    ]


def test_overlay_runner_selects_validation_parameters_and_reports_matched_nulls() -> None:
    frame, splits, detector = _build_synthetic_split()

    result = run_fold_local_regime_overlay_experiment(
        frame=frame,
        splits=splits,
        detector_factory=lambda: detector,
        realized_vol_window=2,
        thresholds=(0.5, 0.9),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )

    split_result = result.split_results[0]
    assert split_result.dangerous_regime_id == 1
    top_row = split_result.validation_grid.iloc[0]
    assert split_result.selected_parameters.threshold == top_row["threshold"]
    assert split_result.selected_parameters.risk_multiplier == top_row["risk_multiplier"]
    assert "same_average_exposure_random" in split_result.matched_nulls["null_summaries"]
    assert "same_turnover_random" in split_result.matched_nulls["null_summaries"]
    assert "same_regime_exposure_random" in split_result.matched_nulls["null_summaries"]
    assert "overlay_vs_base_information_ratio" in result.summary.columns


def test_overlay_runner_emits_executed_position_audit_artifact_without_double_shift() -> None:
    frame, splits, detector = _build_synthetic_split()

    result = run_fold_local_regime_overlay_experiment(
        frame=frame,
        splits=splits,
        detector_factory=lambda: detector,
        realized_vol_window=2,
        thresholds=(0.5, 0.9),
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
        "regime_id",
        "regime_prob_0",
        "regime_prob_1",
        "regime_prob_2",
        "base_position",
        "dangerous_regime_id",
        "dangerous_regime_probability",
        "overlay_threshold",
        "overlay_risk_multiplier",
        "overlay_mode",
        "is_cut_day",
    }
    assert required_columns <= set(artifact.columns)
    assert artifact["executed_position"].tolist() == artifact["raw_signal"].tolist()
    expected_gross = artifact["executed_position"] * artifact["asset_return"]
    pd.testing.assert_series_equal(
        artifact["strategy_gross_return"].reset_index(drop=True),
        expected_gross.reset_index(drop=True),
        check_names=False,
    )


def test_cli_dry_run_does_not_load_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("pd.read_csv should not be called in --dry-run mode.")

    monkeypatch.setattr("scripts.run_regime_overlay_experiment.pd.read_csv", _boom)

    result = cli_main(
        [
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--model-type",
            "hmm",
        ]
    )

    assert result.name == "DRY_RUN_ONLY"
    assert list(tmp_path.iterdir()) == []


def test_cli_execute_requires_required_runtime_args() -> None:
    with pytest.raises(SystemExit):
        cli_main(["--execute"])


def test_cli_output_paths_are_timestamped_and_non_overlapping(tmp_path: Path) -> None:
    paths_a = _build_output_paths(tmp_path, timestamp="20260508_120000")
    paths_b = _build_output_paths(tmp_path, timestamp="20260508_120001")

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
    assert paths_a["summary"].name.startswith("regime_overlay_experiment_summary_")
    assert paths_a["audit_artifacts"].name.startswith("regime_overlay_experiment_audit_artifacts_")
    assert paths_a["fold_details"].name.startswith("regime_overlay_experiment_fold_details_")
    assert paths_a["matched_nulls"].name.startswith("regime_overlay_experiment_matched_nulls_")


def test_overlay_runner_builds_separate_fold_details_and_matched_null_frames() -> None:
    frame, splits, detector = _build_synthetic_split()

    result = run_fold_local_regime_overlay_experiment(
        frame=frame,
        splits=splits,
        detector_factory=lambda: detector,
        realized_vol_window=2,
        thresholds=(0.5, 0.9),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )

    fold_details = build_fold_details_frame(result)
    matched_nulls = build_matched_nulls_frame(result)

    assert len(fold_details) == 1
    assert {
        "split_id",
        "dangerous_regime_id",
        "selected_threshold",
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
    assert "same_regime_exposure_random" in set(matched_nulls["null_name"])


def test_markdown_report_includes_summary_fold_details_and_matched_null_sections() -> None:
    frame, splits, detector = _build_synthetic_split()

    result = run_fold_local_regime_overlay_experiment(
        frame=frame,
        splits=splits,
        detector_factory=lambda: detector,
        realized_vol_window=2,
        thresholds=(0.5, 0.9),
        risk_multipliers=(0.0, 0.5),
        null_n_runs=5,
        include_block_bootstrap=False,
    )
    report = _build_markdown_report(
        summary_frame=result.summary,
        fold_details_frame=build_fold_details_frame(result),
        matched_nulls_frame=build_matched_nulls_frame(result),
        metadata={
            "generated_at_utc": "20260508_120000",
            "input_csv": "/tmp/input.csv",
            "model_type": "hmm",
            "asset_name": "SPY",
            "n_splits": 1,
            "null_runs": 5,
            "decision_grade": False,
            "notes": ["synthetic"],
        },
    )

    assert "# Regime Overlay Experiment Report" in report
    assert "## Summary" in report
    assert "## Fold Details" in report
    assert "## Matched Null Diagnostics" in report
    assert "same_regime_exposure_random" in report
