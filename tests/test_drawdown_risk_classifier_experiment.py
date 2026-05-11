from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from evaluation.drawdown_labels import append_drawdown_label_grid
from evaluation.walk_forward import generate_walk_forward_splits
from experiments.drawdown_risk_classifier_experiment import (
    build_drawdown_classifier_fold_details,
    build_drawdown_classifier_summary,
    evaluate_drawdown_classifier_on_split,
    run_drawdown_risk_classifier_experiment,
)
from scripts.run_drawdown_risk_classifier_experiment import (
    build_parser,
    determine_decision_grade,
    infer_feature_columns,
    main as cli_main,
)


def _build_synthetic_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "Adj Close": [100, 99, 98, 97, 96, 95, 96, 97, 98, 99, 100, 101, 99, 98, 97, 99, 101, 100, 102, 103],
            "forward_simple_return_1d": [0.01, -0.01, -0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.02, -0.01, -0.01, 0.02, 0.02, -0.01, 0.02, 0.01, 0.0],
            "benchmark_return_1d": [0.002] * 20,
            "feature_a": [-2, -1, -1, -1, -1, -0.5, -0.2, 0.1, 0.3, 0.4, 0.8, 1.5, 1.2, 1.0, 0.9, 0.5, -0.4, -0.5, -0.6, -0.7],
            "feature_b": [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1],
            "target_direction": [0] * 20,
        }
    )
    enriched = append_drawdown_label_grid(frame, horizons=(3,), thresholds=(-0.02,))
    return enriched


def test_evaluate_drawdown_classifier_on_split_returns_metrics_and_artifacts() -> None:
    frame = _build_synthetic_frame()
    splits = generate_walk_forward_splits(frame, train_size=8, val_size=4, test_size=4, step_size=4)
    feature_columns = ["feature_a", "feature_b"]

    result = evaluate_drawdown_classifier_on_split(
        model_name="logistic",
        model=__import__("evaluation.drawdown_classification", fromlist=["LogisticDrawdownClassifier"]).LogisticDrawdownClassifier(),
        split=splits[0],
        feature_columns=feature_columns,
        target_column="target_drawdown_event_3d_2pct",
        asset_name="SPY",
    )

    assert result.split_id == 0
    assert "auc_roc" in result.validation_metrics
    assert "auc_roc" in result.test_metrics
    assert "prediction_probability" in result.oof_artifact.columns
    assert "target" in result.oof_artifact.columns


def test_run_drawdown_risk_classifier_experiment_returns_summary_fold_details_and_oof() -> None:
    frame = _build_synthetic_frame()
    splits = generate_walk_forward_splits(frame, train_size=8, val_size=4, test_size=4, step_size=4)
    feature_columns = ["feature_a", "feature_b"]

    result = run_drawdown_risk_classifier_experiment(
        frame=frame,
        splits=splits,
        feature_columns=feature_columns,
        target_column="target_drawdown_event_3d_2pct",
        model_name="logistic",
        asset_name="SPY",
    )

    assert result.model_name == "logistic"
    assert result.target_column == "target_drawdown_event_3d_2pct"
    assert not result.summary.empty
    assert not result.fold_details.empty
    assert not result.oof_artifacts.empty
    assert "mean_test_auc_roc" in result.summary.columns
    assert "mean_validation_auc_roc" in result.summary.columns


def test_fold_detail_and_summary_builders_work_from_split_results() -> None:
    frame = _build_synthetic_frame()
    splits = generate_walk_forward_splits(frame, train_size=8, val_size=4, test_size=4, step_size=4)
    feature_columns = ["feature_a", "feature_b"]
    split_result = evaluate_drawdown_classifier_on_split(
        model_name="always_negative",
        model=__import__("evaluation.drawdown_classification", fromlist=["AlwaysNegativeClassifier"]).AlwaysNegativeClassifier(),
        split=splits[0],
        feature_columns=feature_columns,
        target_column="target_drawdown_event_3d_2pct",
        asset_name="SPY",
    )

    fold_details = build_drawdown_classifier_fold_details([split_result])
    summary = build_drawdown_classifier_summary(fold_details, model_name="always_negative", target_column="target_drawdown_event_3d_2pct")

    assert "validation_auc_roc" in fold_details.columns
    assert "test_auc_roc" in fold_details.columns
    assert "mean_test_auc_roc" in summary.columns


def test_infer_feature_columns_excludes_targets_prices_and_returns() -> None:
    frame = _build_synthetic_frame()

    columns = infer_feature_columns(frame)

    assert "feature_a" in columns
    assert "feature_b" in columns
    assert "Adj Close" not in columns
    assert "target_drawdown_event_3d_2pct" not in columns


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


def test_cli_parser_supports_run_purpose_and_decision_grade() -> None:
    parser = build_parser()
    args = parser.parse_args(["--execute", "--input-csv", "x.csv", "--output-dir", "out", "--run-purpose", "plumbing"])

    assert args.run_purpose == "plumbing"
    assert args.decision_grade is False
