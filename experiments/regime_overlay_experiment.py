"""Standalone fold-local regime-overlay experiment runner.

This module is intentionally isolated from the main model-zoo workflow. It
operates on an already-prepared feature frame plus explicit walk-forward splits,
and it relies on the caller to supply a regime-detector factory. That keeps the
experiment easy to test with synthetic data and prevents accidental calls into
full training/evaluation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from evaluation.metrics import compute_signal_turnover
from evaluation.null_baselines import run_matched_null_suite
from evaluation.regime_overlay import (
    apply_hard_regime_overlay,
    build_hard_overlay_parameter_grid,
    build_vol_target_positions,
    characterize_regimes,
    evaluate_overlay_vs_baseline,
    extract_dangerous_regime_probability,
    identify_dangerous_regime,
)
from evaluation.walk_forward import WalkForwardSplit


DetectorFactory = Callable[[], Any]


@dataclass(slots=True)
class OverlayParameterSelection:
    """Validation-selected overlay parameters for one split."""

    split_id: int
    dangerous_regime_id: int | float
    threshold: float
    risk_multiplier: float
    decision_metric: str
    validation_score: float
    validation_metrics: dict[str, float]


@dataclass(slots=True)
class OverlaySplitResult:
    """Outputs for one walk-forward split."""

    split_id: int
    dangerous_regime_id: int | float
    selected_parameters: OverlayParameterSelection
    validation_grid: pd.DataFrame
    test_evaluation: dict[str, Any]
    matched_nulls: dict[str, Any]
    audit_artifact: pd.DataFrame
    train_regime_summary: pd.DataFrame


@dataclass(slots=True)
class OverlayExperimentResult:
    """Aggregate experiment result across all splits."""

    split_results: list[OverlaySplitResult]
    summary: pd.DataFrame
    audit_artifact_frame: pd.DataFrame


def build_fold_details_frame(result: OverlayExperimentResult) -> pd.DataFrame:
    """Flatten split-level selection and evaluation details into one table."""

    rows: list[dict[str, Any]] = []
    for split_result in result.split_results:
        row: dict[str, Any] = {
            "split_id": split_result.split_id,
            "dangerous_regime_id": split_result.dangerous_regime_id,
            "selected_threshold": split_result.selected_parameters.threshold,
            "selected_risk_multiplier": split_result.selected_parameters.risk_multiplier,
            "decision_metric": split_result.selected_parameters.decision_metric,
            "validation_score": split_result.selected_parameters.validation_score,
            "train_regime_rows": int(len(split_result.train_regime_summary)),
        }
        for key, value in split_result.selected_parameters.validation_metrics.items():
            row[f"validation_{key}"] = value
        for key, value in split_result.test_evaluation["base_metrics"].items():
            row[f"test_base_{key}"] = value
        for key, value in split_result.test_evaluation["overlay_metrics"].items():
            row[f"test_overlay_{key}"] = value
        for key, value in split_result.test_evaluation["overlay_vs_base_metrics"].items():
            row[f"test_overlay_vs_base_{key}"] = value
        for key, value in split_result.test_evaluation["position_change_summary"].items():
            row[f"position_change_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_matched_nulls_frame(result: OverlayExperimentResult) -> pd.DataFrame:
    """Flatten matched-null summaries into one row per split and null family."""

    rows: list[dict[str, Any]] = []
    for split_result in result.split_results:
        decision_metric = split_result.matched_nulls["decision_metric"]
        for null_name, payload in split_result.matched_nulls["null_summaries"].items():
            summary = payload["summary"]
            rows.append(
                {
                    "split_id": split_result.split_id,
                    "dangerous_regime_id": split_result.dangerous_regime_id,
                    "null_name": null_name,
                    "decision_metric": decision_metric,
                    "canonical_value": summary.canonical_value,
                    "mean_null_value": summary.mean_null_value,
                    "percentile_95_null_value": summary.percentile_95_null_value,
                    "p_value": summary.p_value,
                    "n_runs": summary.n_runs,
                    "passes_p_value_gate": bool(summary.p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def _coerce_float_series(values, *, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.astype(float).copy()
        if not series.index.equals(index):
            series = series.reindex(index)
        series.name = name
        return series
    return pd.Series(values, index=index, name=name, dtype=float)


def _concat_split_frames(split: WalkForwardSplit) -> pd.DataFrame:
    train = split.train.copy()
    validation = split.validation.copy()
    test = split.test.copy()
    train["slice_name"] = "train"
    validation["slice_name"] = "validation"
    test["slice_name"] = "test"
    combined = pd.concat([train, validation, test], axis=0)
    combined["split_id"] = split.split_id
    return combined


def _slice_with_predictions(
    frame: pd.DataFrame,
    *,
    prediction,
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["regime_id"] = prediction.labels.astype(int)
    for column in prediction.probabilities.columns:
        annotated[column] = prediction.probabilities[column].astype(float)
    return annotated


def _build_overlay_audit_artifact_frame(
    *,
    frame: pd.DataFrame,
    split_id: int,
    model_name: str,
    asset_name: str | None,
    raw_signal: pd.Series,
    prediction_probability: pd.Series,
    executed_position: pd.Series,
    benchmark_column: str,
    return_column: str,
    target_column: str | None,
    transaction_cost_bps: float,
    base_position: pd.Series,
    dangerous_regime_probability: pd.Series,
    dangerous_regime_id: int | float,
    threshold: float,
    risk_multiplier: float,
) -> pd.DataFrame:
    """Build an audit-ready artifact frame without shifting positions again."""

    artifact = frame.copy()
    index = artifact.index
    signal = _coerce_float_series(raw_signal, index=index, name="raw_signal")
    probability = _coerce_float_series(prediction_probability, index=index, name="prediction_probability")
    position = _coerce_float_series(executed_position, index=index, name="executed_position")
    base = _coerce_float_series(base_position, index=index, name="base_position")
    danger_probability = _coerce_float_series(
        dangerous_regime_probability,
        index=index,
        name="dangerous_regime_probability",
    )
    asset_returns = artifact[return_column].astype(float).rename("asset_return")
    benchmark_returns = artifact[benchmark_column].astype(float).rename("benchmark_return")
    turnover = compute_signal_turnover(position).rename("turnover")
    transaction_cost = (turnover * (transaction_cost_bps / 10_000.0)).rename("transaction_cost")
    gross = (position * asset_returns).rename("strategy_gross_return")
    net = (gross - transaction_cost).rename("strategy_net_return")

    output = pd.DataFrame(
        {
            "date": artifact.index if "date" not in artifact.columns else artifact["date"].values,
            "split_id": split_id,
            "model_name": model_name,
            "asset": asset_name if asset_name is not None else np.nan,
            "asset_return": asset_returns.values,
            "benchmark_return": benchmark_returns.values,
            "raw_signal": signal.values,
            "prediction_probability": probability.values,
            "target": artifact[target_column].values if target_column and target_column in artifact.columns else np.nan,
            "executed_position": position.values,
            "strategy_gross_return": gross.values,
            "strategy_net_return": net.values,
            "turnover": turnover.values,
            "transaction_cost": transaction_cost.values,
            "regime_id": artifact["regime_id"].values if "regime_id" in artifact.columns else np.nan,
            "base_position": base.values,
            "dangerous_regime_id": dangerous_regime_id,
            "dangerous_regime_probability": danger_probability.values,
            "overlay_threshold": threshold,
            "overlay_risk_multiplier": risk_multiplier,
            "overlay_mode": "hard_veto",
            "is_cut_day": position.lt(base).astype(int).values,
        }
    )

    for column in artifact.columns:
        if column.startswith("regime_prob_") and column not in output.columns:
            output[column] = artifact[column].values

    return output


def _select_overlay_parameters(
    *,
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    base_positions: pd.Series,
    dangerous_regime_probability: pd.Series,
    parameter_grid: list[dict[str, float]],
    transaction_cost_bps: float,
    annualization: int,
    decision_metric: str,
) -> tuple[OverlayParameterSelection, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_evaluation: dict[str, Any] | None = None
    best_value = float("-inf")

    for params in parameter_grid:
        overlay_positions = apply_hard_regime_overlay(
            base_positions,
            dangerous_regime_probability,
            threshold=float(params["threshold"]),
            risk_multiplier=float(params["risk_multiplier"]),
        )
        evaluation = evaluate_overlay_vs_baseline(
            asset_returns=asset_returns,
            benchmark_returns=benchmark_returns,
            base_positions=base_positions,
            overlay_positions=overlay_positions,
            transaction_cost_bps=transaction_cost_bps,
            annualization=annualization,
        )
        validation_metrics = evaluation["overlay_vs_base_metrics"]
        score = float(validation_metrics.get(decision_metric, float("-inf")))
        row = {
            "threshold": float(params["threshold"]),
            "risk_multiplier": float(params["risk_multiplier"]),
            "validation_score": score,
            "information_ratio": float(validation_metrics.get("information_ratio", 0.0)),
            "active_calmar": float(validation_metrics.get("active_calmar", 0.0)),
            "annualized_active_return": float(validation_metrics.get("annualized_active_return", 0.0)),
            "fraction_in_market": float(validation_metrics.get("fraction_in_market", 0.0)),
            "daily_turnover": float(validation_metrics.get("daily_turnover", 0.0)),
        }
        rows.append(row)
        if score > best_value:
            best_value = score
            best_row = row
            best_evaluation = evaluation

    if best_row is None or best_evaluation is None:
        raise ValueError("Overlay parameter selection received an empty parameter grid.")

    selection = OverlayParameterSelection(
        split_id=-1,
        dangerous_regime_id=-1,
        threshold=float(best_row["threshold"]),
        risk_multiplier=float(best_row["risk_multiplier"]),
        decision_metric=decision_metric,
        validation_score=float(best_row["validation_score"]),
        validation_metrics={
            key: float(best_evaluation["overlay_vs_base_metrics"].get(key, 0.0))
            for key in [
                "information_ratio",
                "active_calmar",
                "annualized_active_return",
                "active_max_drawdown",
                "daily_turnover",
                "fraction_in_market",
            ]
        },
    )
    return selection, pd.DataFrame(rows).sort_values(
        ["validation_score", "threshold", "risk_multiplier"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _summarize_split_result(split_result: OverlaySplitResult) -> dict[str, float | int]:
    base_metrics = split_result.test_evaluation["base_metrics"]
    overlay_metrics = split_result.test_evaluation["overlay_metrics"]
    relative_metrics = split_result.test_evaluation["overlay_vs_base_metrics"]
    same_exposure = split_result.matched_nulls["null_summaries"]["same_average_exposure_random"]["summary"]
    same_turnover = split_result.matched_nulls["null_summaries"]["same_turnover_random"]["summary"]

    row: dict[str, float | int] = {
        "split_id": split_result.split_id,
        "dangerous_regime_id": float(split_result.dangerous_regime_id),
        "threshold": split_result.selected_parameters.threshold,
        "risk_multiplier": split_result.selected_parameters.risk_multiplier,
        "base_information_ratio": float(base_metrics["information_ratio"]),
        "overlay_information_ratio": float(overlay_metrics["information_ratio"]),
        "overlay_vs_base_information_ratio": float(relative_metrics["information_ratio"]),
        "overlay_vs_base_active_calmar": float(relative_metrics["active_calmar"]),
        "overlay_vs_base_annualized_active_return": float(relative_metrics["annualized_active_return"]),
        "fraction_in_market": float(overlay_metrics["fraction_in_market"]),
        "daily_turnover": float(overlay_metrics["daily_turnover"]),
        "same_average_exposure_p_value": float(same_exposure.p_value),
        "same_turnover_p_value": float(same_turnover.p_value),
    }
    if "same_regime_exposure_random" in split_result.matched_nulls["null_summaries"]:
        row["same_regime_exposure_p_value"] = float(
            split_result.matched_nulls["null_summaries"]["same_regime_exposure_random"]["summary"].p_value
        )
    return row


def run_fold_local_regime_overlay_experiment(
    *,
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    detector_factory: DetectorFactory,
    benchmark_column: str = "benchmark_return_1d",
    return_column: str = "forward_simple_return_1d",
    target_column: str | None = "target_direction",
    asset_name: str | None = None,
    model_name: str = "hmm_regime_overlay_hard_veto",
    target_vol: float = 0.10,
    realized_vol_window: int = 20,
    annualization: int = 252,
    min_position: float = 0.0,
    max_position: float = 1.0,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7),
    risk_multipliers: tuple[float, ...] = (0.0, 0.25, 0.5),
    transaction_cost_bps: float = 2.0,
    decision_metric: str = "information_ratio",
    null_n_runs: int = 100,
    null_seed: int = 42,
    include_block_bootstrap: bool = True,
) -> OverlayExperimentResult:
    """Run a fold-local regime overlay experiment without model-zoo dependencies."""

    if not splits:
        raise ValueError("At least one walk-forward split is required.")
    if return_column not in frame.columns:
        raise KeyError(f"Missing required return column {return_column!r}.")
    if benchmark_column not in frame.columns:
        raise KeyError(f"Missing required benchmark column {benchmark_column!r}.")

    parameter_grid = build_hard_overlay_parameter_grid(
        thresholds=list(thresholds),
        risk_multipliers=list(risk_multipliers),
    )
    split_results: list[OverlaySplitResult] = []

    for split in splits:
        detector = detector_factory()
        detector.fit(split.train)

        train_prediction = detector.predict(split.train)
        validation_prediction = detector.predict_live_safe(split.validation)
        test_prediction = detector.predict_live_safe(split.test)

        train_frame = _slice_with_predictions(split.train, prediction=train_prediction)
        validation_frame = _slice_with_predictions(split.validation, prediction=validation_prediction)
        test_frame = _slice_with_predictions(split.test, prediction=test_prediction)

        train_regime_summary = characterize_regimes(
            train_frame[return_column],
            train_frame["regime_id"],
            annualization=annualization,
        )
        dangerous_regime_id = identify_dangerous_regime(train_regime_summary)

        combined_frame = _concat_split_frames(
            WalkForwardSplit(
                train=train_frame,
                validation=validation_frame,
                test=test_frame,
                split_id=split.split_id,
            )
        )
        combined_positions = build_vol_target_positions(
            combined_frame[return_column],
            target_vol=target_vol,
            realized_vol_window=realized_vol_window,
            annualization=annualization,
            min_position=min_position,
            max_position=max_position,
            split_ids=combined_frame["split_id"],
        )
        combined_frame["base_position"] = combined_positions.values

        validation_probability = extract_dangerous_regime_probability(
            validation_frame.filter(regex=r"^regime_prob_"),
            dangerous_regime_id,
        )
        test_probability = extract_dangerous_regime_probability(
            test_frame.filter(regex=r"^regime_prob_"),
            dangerous_regime_id,
        )

        validation_base = combined_frame.loc[combined_frame["slice_name"] == "validation", "base_position"]
        selection, validation_grid = _select_overlay_parameters(
            asset_returns=validation_frame[return_column].astype(float),
            benchmark_returns=validation_frame[benchmark_column].astype(float),
            base_positions=validation_base.astype(float),
            dangerous_regime_probability=validation_probability,
            parameter_grid=parameter_grid,
            transaction_cost_bps=transaction_cost_bps,
            annualization=annualization,
            decision_metric=decision_metric,
        )
        selection = OverlayParameterSelection(
            split_id=split.split_id,
            dangerous_regime_id=dangerous_regime_id,
            threshold=selection.threshold,
            risk_multiplier=selection.risk_multiplier,
            decision_metric=selection.decision_metric,
            validation_score=selection.validation_score,
            validation_metrics=selection.validation_metrics,
        )

        test_base = combined_frame.loc[combined_frame["slice_name"] == "test", "base_position"].astype(float)
        test_overlay = apply_hard_regime_overlay(
            test_base,
            test_probability,
            threshold=selection.threshold,
            risk_multiplier=selection.risk_multiplier,
        )
        test_evaluation = evaluate_overlay_vs_baseline(
            asset_returns=test_frame[return_column].astype(float),
            benchmark_returns=test_frame[benchmark_column].astype(float),
            base_positions=test_base,
            overlay_positions=test_overlay,
            transaction_cost_bps=transaction_cost_bps,
            annualization=annualization,
        )
        base_net_returns = (
            test_base.astype(float) * test_frame[return_column].astype(float)
            - compute_signal_turnover(test_base.astype(float)) * (transaction_cost_bps / 10_000.0)
        )
        matched_nulls = run_matched_null_suite(
            positions=test_overlay,
            returns=test_frame[return_column].astype(float),
            benchmark_returns=base_net_returns,
            regime_labels=test_frame["regime_id"],
            n_runs=null_n_runs,
            seed=null_seed + split.split_id,
            annualization_factor=annualization,
            transaction_cost_bps=transaction_cost_bps,
            decision_metric=decision_metric,
            include_block_bootstrap=include_block_bootstrap,
        )

        audit_artifact = _build_overlay_audit_artifact_frame(
            frame=test_frame,
            split_id=split.split_id,
            model_name=model_name,
            asset_name=asset_name,
            raw_signal=test_overlay,
            prediction_probability=test_probability,
            executed_position=test_overlay,
            benchmark_column=benchmark_column,
            return_column=return_column,
            target_column=target_column,
            transaction_cost_bps=transaction_cost_bps,
            base_position=test_base,
            dangerous_regime_probability=test_probability,
            dangerous_regime_id=dangerous_regime_id,
            threshold=selection.threshold,
            risk_multiplier=selection.risk_multiplier,
        )

        split_results.append(
            OverlaySplitResult(
                split_id=split.split_id,
                dangerous_regime_id=dangerous_regime_id,
                selected_parameters=selection,
                validation_grid=validation_grid,
                test_evaluation=test_evaluation,
                matched_nulls=matched_nulls,
                audit_artifact=audit_artifact,
                train_regime_summary=train_regime_summary,
            )
        )

    summary = pd.DataFrame([_summarize_split_result(split_result) for split_result in split_results])
    audit_artifact_frame = pd.concat(
        [split_result.audit_artifact for split_result in split_results],
        axis=0,
        ignore_index=True,
    )
    return OverlayExperimentResult(
        split_results=split_results,
        summary=summary,
        audit_artifact_frame=audit_artifact_frame,
    )
