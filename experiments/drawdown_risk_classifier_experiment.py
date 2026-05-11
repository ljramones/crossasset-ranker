"""Standalone walk-forward drawdown-risk classifier baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import pandas as pd

from evaluation.audit_artifacts import build_standard_audit_artifact_frame
from evaluation.drawdown_classification import (
    DrawdownClassifier,
    build_simple_drawdown_classifier,
    compute_drawdown_classification_metrics,
)
from evaluation.walk_forward import WalkForwardSplit


@dataclass(slots=True)
class DrawdownClassifierSplitResult:
    split_id: int
    model_name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    oof_artifact: pd.DataFrame


@dataclass(slots=True)
class DrawdownClassifierExperimentResult:
    model_name: str
    target_column: str
    summary: pd.DataFrame
    fold_details: pd.DataFrame
    oof_artifacts: pd.DataFrame


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


def evaluate_drawdown_classifier_on_split(
    *,
    model: DrawdownClassifier,
    model_name: str,
    split: WalkForwardSplit,
    feature_columns: list[str],
    target_column: str,
    asset_name: str = "SPY",
) -> DrawdownClassifierSplitResult:
    """Fit on train, report validation and test metrics separately."""

    X_train, y_train = _prepare_xy(split.train, feature_columns=feature_columns, target_column=target_column)
    X_validation, y_validation = _prepare_xy(split.validation, feature_columns=feature_columns, target_column=target_column)
    X_test, y_test = _prepare_xy(split.test, feature_columns=feature_columns, target_column=target_column)

    model.fit(X_train, y_train)
    validation_probability = model.predict_proba(X_validation)
    validation_prediction = model.predict(X_validation)
    probability = model.predict_proba(X_test)
    prediction = model.predict(X_test)

    validation_metrics = compute_drawdown_classification_metrics(y_validation, validation_probability, validation_prediction)
    validation_metrics["split_id"] = float(split.split_id)
    test_metrics = compute_drawdown_classification_metrics(y_test, probability, prediction)
    test_metrics["split_id"] = float(split.split_id)

    artifact_frame = split.test.loc[X_test.index].copy()
    oof_artifact = build_standard_audit_artifact_frame(
        frame=artifact_frame,
        label=model_name,
        prediction=prediction,
        probability=probability,
        split_id=split.split_id,
        transaction_cost_bps=0.0,
        asset_name=asset_name,
        target_column=target_column,
        return_column="forward_simple_return_1d",
        benchmark_column="benchmark_return_1d",
    )

    return DrawdownClassifierSplitResult(
        split_id=split.split_id,
        model_name=model_name,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        oof_artifact=oof_artifact,
    )


def build_drawdown_classifier_fold_details(split_results: list[DrawdownClassifierSplitResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split_id": result.split_id,
                "model_name": result.model_name,
                **{f"validation_{key}": value for key, value in result.validation_metrics.items() if key != "split_id"},
                **{f"test_{key}": value for key, value in result.test_metrics.items() if key != "split_id"},
            }
            for result in split_results
        ]
    ).sort_values("split_id").reset_index(drop=True)


def build_drawdown_classifier_summary(
    fold_details: pd.DataFrame,
    *,
    model_name: str,
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
    row = {
        "model_name": model_name,
        "target_column": target_column,
        "n_splits": int(len(fold_details)),
    }
    for column in metric_columns:
        test_column = f"test_{column}"
        validation_column = f"validation_{column}"
        if test_column in fold_details.columns:
            row[f"mean_test_{column}"] = float(fold_details[test_column].mean())
            row[f"median_test_{column}"] = float(fold_details[test_column].median())
        if validation_column in fold_details.columns:
            row[f"mean_validation_{column}"] = float(fold_details[validation_column].mean())
    if "test_auc_roc" in fold_details.columns:
        row["positive_test_auc_folds"] = int((fold_details["test_auc_roc"] > 0.5).sum())
        row["std_test_auc_roc"] = float(fold_details["test_auc_roc"].std(ddof=0))
    return pd.DataFrame([row])


def run_drawdown_risk_classifier_experiment(
    *,
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    asset_name: str = "SPY",
    model: DrawdownClassifier | None = None,
) -> DrawdownClassifierExperimentResult:
    """Run a standalone simple classifier baseline across walk-forward splits."""

    classifier_template = model or build_simple_drawdown_classifier(model_name)
    split_results: list[DrawdownClassifierSplitResult] = []

    for split in splits:
        split_results.append(
            evaluate_drawdown_classifier_on_split(
                model=deepcopy(classifier_template),
                model_name=model_name,
                split=split,
                feature_columns=feature_columns,
                target_column=target_column,
                asset_name=asset_name,
            )
        )

    fold_details = build_drawdown_classifier_fold_details(split_results)
    summary = build_drawdown_classifier_summary(
        fold_details,
        model_name=model_name,
        target_column=target_column,
    )
    oof_artifacts = pd.concat([result.oof_artifact for result in split_results], ignore_index=True)

    return DrawdownClassifierExperimentResult(
        model_name=model_name,
        target_column=target_column,
        summary=summary,
        fold_details=fold_details,
        oof_artifacts=oof_artifacts,
    )
