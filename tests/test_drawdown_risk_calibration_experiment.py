from __future__ import annotations

import sys

import pandas as pd

from evaluation.drawdown_labels import append_drawdown_label_grid
from evaluation.walk_forward import generate_walk_forward_splits
from experiments.drawdown_risk_calibration_experiment import (
    build_drawdown_calibration_fold_details,
    build_drawdown_calibration_summary,
    evaluate_drawdown_calibration_on_split,
    run_drawdown_risk_calibration_experiment,
)
from scripts.run_drawdown_risk_calibration_experiment import (
    build_parser,
    determine_decision_grade,
    main as cli_main,
)


def _build_synthetic_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=24, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "Adj Close": [100, 99, 98, 97, 96, 95, 94, 95, 96, 97, 98, 99, 101, 100, 99, 98, 97, 99, 101, 102, 101, 100, 99, 98],
            "forward_simple_return_1d": [
                -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                -0.01, -0.01, -0.01, -0.01, 0.02, 0.02, 0.01, -0.01, -0.01, -0.01, -0.01, 0.0,
            ],
            "benchmark_return_1d": [0.002] * 24,
            "feature_a": [-2.5, -2.0, -1.8, -1.6, -1.4, -1.2, -0.8, -0.5, -0.2, 0.1, 0.4, 0.8, 1.4, 1.2, 1.0, 0.7, 0.4, 0.0, -0.4, -0.8, -1.0, -1.2, -1.4, -1.6],
            "feature_b": [0.7, 0.8, 0.8, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.7, 0.5, 0.4, 0.3, 0.3, 0.4, 0.5, 0.6],
            "target_direction": [0] * 24,
        }
    )
    return append_drawdown_label_grid(frame, horizons=(3,), thresholds=(-0.02,))


def test_evaluate_drawdown_calibration_on_split_returns_raw_and_calibrated_metrics() -> None:
    frame = _build_synthetic_frame()
    splits = generate_walk_forward_splits(frame, train_size=10, val_size=6, test_size=4, step_size=4)

    result = evaluate_drawdown_calibration_on_split(
        base_model=__import__("evaluation.drawdown_classification", fromlist=["RegularizedLinearDrawdownClassifier"]).RegularizedLinearDrawdownClassifier(),
        base_model_name="regularized_linear",
        calibrator=__import__("evaluation.probability_calibration", fromlist=["PlattCalibrator"]).PlattCalibrator(),
        calibration_method="platt",
        split=splits[0],
        feature_columns=["feature_a", "feature_b"],
        target_column="target_drawdown_event_3d_2pct",
        asset_name="SPY",
        n_bins=5,
    )

    assert "auc_roc" in result.validation_raw_metrics
    assert "auc_roc" in result.validation_calibrated_metrics
    assert "auc_roc" in result.test_raw_metrics
    assert "auc_roc" in result.test_calibrated_metrics
    assert "raw_prediction_probability" in result.oof_artifact.columns
    assert "calibrated_prediction_probability" in result.oof_artifact.columns
    assert not result.calibration_bins.empty


def test_run_drawdown_risk_calibration_experiment_returns_complete_bundle() -> None:
    frame = _build_synthetic_frame()
    splits = generate_walk_forward_splits(frame, train_size=10, val_size=6, test_size=4, step_size=4)

    result = run_drawdown_risk_calibration_experiment(
        frame=frame,
        splits=splits,
        feature_columns=["feature_a", "feature_b"],
        target_column="target_drawdown_event_3d_2pct",
        base_model_name="logistic",
        calibration_method="isotonic",
        asset_name="SPY",
        n_bins=5,
    )

    assert result.base_model_name == "logistic"
    assert result.calibration_method == "isotonic"
    assert not result.summary.empty
    assert not result.fold_details.empty
    assert not result.oof_artifacts.empty
    assert not result.calibration_bins.empty
    assert "mean_test_calibrated_auc_roc" in result.summary.columns
    assert "mean_test_brier_improvement" in result.summary.columns


def test_fold_detail_and_summary_builders_work() -> None:
    frame = _build_synthetic_frame()
    splits = generate_walk_forward_splits(frame, train_size=10, val_size=6, test_size=4, step_size=4)
    split_result = evaluate_drawdown_calibration_on_split(
        base_model=__import__("evaluation.drawdown_classification", fromlist=["LogisticDrawdownClassifier"]).LogisticDrawdownClassifier(),
        base_model_name="logistic",
        calibrator=__import__("evaluation.probability_calibration", fromlist=["IdentityCalibrator"]).IdentityCalibrator(),
        calibration_method="identity",
        split=splits[0],
        feature_columns=["feature_a", "feature_b"],
        target_column="target_drawdown_event_3d_2pct",
        asset_name="SPY",
    )

    fold_details = build_drawdown_calibration_fold_details([split_result])
    summary = build_drawdown_calibration_summary(
        fold_details,
        base_model_name="logistic",
        calibration_method="identity",
        target_column="target_drawdown_event_3d_2pct",
    )

    assert "validation_raw_auc_roc" in fold_details.columns
    assert "test_calibrated_auc_roc" in fold_details.columns
    assert "mean_test_calibrated_auc_roc" in summary.columns


def test_cli_dry_run_exits_without_loading_or_writing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--dry-run"])

    cli_main()

    output = capsys.readouterr().out
    assert "dry run only" in output.lower()
    assert "No data will be loaded" in output


def test_determine_decision_grade_respects_run_purpose_and_flag() -> None:
    assert determine_decision_grade(run_purpose="plumbing", decision_grade_flag=False) is False
    assert determine_decision_grade(run_purpose="diagnostic", decision_grade_flag=False) is False
    assert determine_decision_grade(run_purpose="decision_grade", decision_grade_flag=False) is True
    assert determine_decision_grade(run_purpose="plumbing", decision_grade_flag=True) is True


def test_cli_parser_supports_calibration_method_and_run_purpose() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--execute",
            "--input-csv",
            "x.csv",
            "--output-dir",
            "out",
            "--base-model-name",
            "logistic",
            "--calibration-method",
            "platt",
            "--run-purpose",
            "plumbing",
        ]
    )

    assert args.base_model_name == "logistic"
    assert args.calibration_method == "platt"
    assert args.run_purpose == "plumbing"

