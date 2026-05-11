"""Standalone walk-forward calibration experiment for drawdown-risk classifiers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from evaluation.audit_artifacts import build_standard_audit_artifact_frame
from evaluation.calibration_diagnostics import build_calibration_table
from evaluation.drawdown_classification import (
    DrawdownClassifier,
    build_simple_drawdown_classifier,
    compute_drawdown_classification_metrics,
)
from evaluation.probability_calibration import (
    ProbabilityCalibrator,
    build_probability_calibrator,
)
from evaluation.walk_forward import WalkForwardSplit


@dataclass(slots=True)
class DrawdownCalibrationSplitResult:
    split_id: int
    base_model_name: str
    calibration_method: str
    validation_raw_metrics: dict[str, float]
    validation_calibrated_metrics: dict[str, float]
    test_raw_metrics: dict[str, float]
    test_calibrated_metrics: dict[str, float]
    oof_artifact: pd.DataFrame
    calibration_bins: pd.DataFrame


@dataclass(slots=True)
class DrawdownCalibrationExperimentResult:
    base_model_name: str
    calibration_method: str
    target_column: str
    summary: pd.DataFrame
    fold_details: pd.DataFrame
    oof_artifacts: pd.DataFrame
    calibration_bins: pd.DataFrame


def _prepare_xy(frame: pd.DataFrame, *, feature_columns: list[str], target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in frame.columns:
        raise KeyError(f"Missing target column {target_column!r}.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    subset = frame[feature_columns + [target_column]].dropna().copy()
    X = subset[feature_columns].astype(float)
    y = subset[target_column].astype(int)
    return X, y


def evaluate_drawdown_calibration_on_split(
    *,
    base_model: DrawdownClassifier,
    base_model_name: str,
    calibrator: ProbabilityCalibrator,
    calibration_method: str,
    split: WalkForwardSplit,
    feature_columns: list[str],
    target_column: str,
    asset_name: str = "SPY",
    n_bins: int = 10,
) -> DrawdownCalibrationSplitResult:
    """Fit classifier on train, fit calibrator on validation, score raw vs calibrated test probabilities."""

    X_train, y_train = _prepare_xy(split.train, feature_columns=feature_columns, target_column=target_column)
    X_validation, y_validation = _prepare_xy(split.validation, feature_columns=feature_columns, target_column=target_column)
    X_test, y_test = _prepare_xy(split.test, feature_columns=feature_columns, target_column=target_column)

    base_model.fit(X_train, y_train)

    validation_raw_probability = base_model.predict_proba(X_validation)
    validation_raw_prediction = (validation_raw_probability >= 0.5).astype(int).rename("prediction")

    calibrator.fit(validation_raw_probability, y_validation)
    validation_calibrated_probability = calibrator.predict_proba(validation_raw_probability)
    validation_calibrated_prediction = (validation_calibrated_probability >= 0.5).astype(int).rename("prediction")

    test_raw_probability = base_model.predict_proba(X_test)
    test_raw_prediction = (test_raw_probability >= 0.5).astype(int).rename("prediction")
    test_calibrated_probability = calibrator.predict_proba(test_raw_probability)
    test_calibrated_prediction = (test_calibrated_probability >= 0.5).astype(int).rename("prediction")

    validation_raw_metrics = compute_drawdown_classification_metrics(
        y_validation,
        validation_raw_probability,
        validation_raw_prediction,
    )
    validation_calibrated_metrics = compute_drawdown_classification_metrics(
        y_validation,
        validation_calibrated_probability,
        validation_calibrated_prediction,
    )
    test_raw_metrics = compute_drawdown_classification_metrics(
        y_test,
        test_raw_probability,
        test_raw_prediction,
    )
    test_calibrated_metrics = compute_drawdown_classification_metrics(
        y_test,
        test_calibrated_probability,
        test_calibrated_prediction,
    )

    artifact_frame = split.test.loc[X_test.index].copy()
    oof_artifact = build_standard_audit_artifact_frame(
        frame=artifact_frame,
        label=f"{base_model_name}__{calibration_method}",
        prediction=test_calibrated_prediction,
        probability=test_calibrated_probability,
        split_id=split.split_id,
        transaction_cost_bps=0.0,
        asset_name=asset_name,
        target_column=target_column,
        return_column="forward_simple_return_1d",
        benchmark_column="benchmark_return_1d",
    )
    oof_artifact["base_model_name"] = base_model_name
    oof_artifact["calibration_method"] = calibration_method
    oof_artifact["raw_prediction_probability"] = pd.Series(test_raw_probability, index=X_test.index).to_numpy()
    oof_artifact["calibrated_prediction_probability"] = pd.Series(test_calibrated_probability, index=X_test.index).to_numpy()

    bins = build_calibration_table(y_test, test_calibrated_probability, n_bins=n_bins, strategy="quantile")
    bins["split_id"] = split.split_id
    bins["base_model_name"] = base_model_name
    bins["calibration_method"] = calibration_method

    return DrawdownCalibrationSplitResult(
        split_id=split.split_id,
        base_model_name=base_model_name,
        calibration_method=calibration_method,
        validation_raw_metrics=validation_raw_metrics,
        validation_calibrated_metrics=validation_calibrated_metrics,
        test_raw_metrics=test_raw_metrics,
        test_calibrated_metrics=test_calibrated_metrics,
        oof_artifact=oof_artifact,
        calibration_bins=bins,
    )


def build_drawdown_calibration_fold_details(split_results: list[DrawdownCalibrationSplitResult]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for result in split_results:
        row: dict[str, float | str | int] = {
            "split_id": result.split_id,
            "base_model_name": result.base_model_name,
            "calibration_method": result.calibration_method,
        }
        row.update({f"validation_raw_{key}": value for key, value in result.validation_raw_metrics.items()})
        row.update({f"validation_calibrated_{key}": value for key, value in result.validation_calibrated_metrics.items()})
        row.update({f"test_raw_{key}": value for key, value in result.test_raw_metrics.items()})
        row.update({f"test_calibrated_{key}": value for key, value in result.test_calibrated_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)


def build_drawdown_calibration_summary(
    fold_details: pd.DataFrame,
    *,
    base_model_name: str,
    calibration_method: str,
    target_column: str,
) -> pd.DataFrame:
    metric_columns = [
        "directional_accuracy",
        "auc_roc",
        "balanced_accuracy",
        "precision",
        "recall",
        "brier_score",
        "positive_prediction_rate",
        "base_event_rate",
    ]
    row: dict[str, float | str | int] = {
        "base_model_name": base_model_name,
        "calibration_method": calibration_method,
        "target_column": target_column,
        "n_splits": int(len(fold_details)),
    }
    for prefix in ["validation_raw", "validation_calibrated", "test_raw", "test_calibrated"]:
        for metric in metric_columns:
            column = f"{prefix}_{metric}"
            if column in fold_details.columns:
                row[f"mean_{column}"] = float(fold_details[column].mean())
                row[f"median_{column}"] = float(fold_details[column].median())
    if "test_calibrated_auc_roc" in fold_details.columns:
        row["positive_test_calibrated_auc_folds"] = int((fold_details["test_calibrated_auc_roc"] > 0.5).sum())
        row["std_test_calibrated_auc_roc"] = float(fold_details["test_calibrated_auc_roc"].std(ddof=0))
    if "test_calibrated_brier_score" in fold_details.columns and "test_raw_brier_score" in fold_details.columns:
        row["mean_test_brier_improvement"] = float(
            (fold_details["test_raw_brier_score"] - fold_details["test_calibrated_brier_score"]).mean()
        )
    return pd.DataFrame([row])


def run_drawdown_risk_calibration_experiment(
    *,
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_columns: list[str],
    target_column: str,
    base_model_name: str,
    calibration_method: str,
    asset_name: str = "SPY",
    base_model: DrawdownClassifier | None = None,
    calibrator: ProbabilityCalibrator | None = None,
    n_bins: int = 10,
) -> DrawdownCalibrationExperimentResult:
    """Run a standalone fold-safe calibration experiment across walk-forward splits."""

    base_template = base_model or build_simple_drawdown_classifier(base_model_name)
    calibrator_template = calibrator or build_probability_calibrator(calibration_method)
    split_results: list[DrawdownCalibrationSplitResult] = []

    for split in splits:
        split_results.append(
            evaluate_drawdown_calibration_on_split(
                base_model=deepcopy(base_template),
                base_model_name=base_model_name,
                calibrator=deepcopy(calibrator_template),
                calibration_method=calibration_method,
                split=split,
                feature_columns=feature_columns,
                target_column=target_column,
                asset_name=asset_name,
                n_bins=n_bins,
            )
        )

    fold_details = build_drawdown_calibration_fold_details(split_results)
    summary = build_drawdown_calibration_summary(
        fold_details,
        base_model_name=base_model_name,
        calibration_method=calibration_method,
        target_column=target_column,
    )
    oof_artifacts = pd.concat([result.oof_artifact for result in split_results], ignore_index=True)
    calibration_bins = pd.concat([result.calibration_bins for result in split_results], ignore_index=True)

    return DrawdownCalibrationExperimentResult(
        base_model_name=base_model_name,
        calibration_method=calibration_method,
        target_column=target_column,
        summary=summary,
        fold_details=fold_details,
        oof_artifacts=oof_artifacts,
        calibration_bins=calibration_bins,
    )

