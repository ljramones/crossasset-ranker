"""Standalone fold-local volatility-quantile overlay experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import compute_signal_turnover
from evaluation.regime_overlay import (
    build_hard_overlay_parameter_grid,
    build_vol_target_positions,
    evaluate_overlay_vs_baseline,
)
from evaluation.volatility_overlay import (
    apply_volatility_quantile_overlay,
    assign_volatility_states,
    compute_trailing_realized_volatility,
    derive_train_volatility_cutoffs,
    identify_high_vol_state_ids,
    run_volatility_matched_null_suite,
)
from evaluation.walk_forward import WalkForwardSplit


@dataclass(slots=True)
class VolQuantileOverlayParameterSelection:
    split_id: int
    quantile: float
    cutoff_value: float
    threshold_state: int
    risk_multiplier: float
    decision_metric: str
    validation_score: float
    validation_metrics: dict[str, float]


@dataclass(slots=True)
class VolQuantileOverlaySplitResult:
    split_id: int
    selected_parameters: VolQuantileOverlayParameterSelection
    validation_grid: pd.DataFrame
    test_evaluation: dict[str, Any]
    matched_nulls: dict[str, Any]
    audit_artifact: pd.DataFrame


@dataclass(slots=True)
class VolQuantileOverlayExperimentResult:
    split_results: list[VolQuantileOverlaySplitResult]
    summary: pd.DataFrame
    audit_artifact_frame: pd.DataFrame


def build_fold_details_frame(result: VolQuantileOverlayExperimentResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_result in result.split_results:
        row: dict[str, Any] = {
            "split_id": split_result.split_id,
            "selected_quantile": split_result.selected_parameters.quantile,
            "selected_cutoff_value": split_result.selected_parameters.cutoff_value,
            "selected_threshold_state": split_result.selected_parameters.threshold_state,
            "selected_risk_multiplier": split_result.selected_parameters.risk_multiplier,
            "decision_metric": split_result.selected_parameters.decision_metric,
            "validation_score": split_result.selected_parameters.validation_score,
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


def build_matched_nulls_frame(result: VolQuantileOverlayExperimentResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_result in result.split_results:
        decision_metric = split_result.matched_nulls["decision_metric"]
        for null_name, payload in split_result.matched_nulls["null_summaries"].items():
            summary = payload["summary"]
            rows.append(
                {
                    "split_id": split_result.split_id,
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


def _build_overlay_audit_artifact_frame(
    *,
    frame: pd.DataFrame,
    split_id: int,
    model_name: str,
    asset_name: str | None,
    raw_signal: pd.Series,
    executed_position: pd.Series,
    benchmark_column: str,
    return_column: str,
    target_column: str | None,
    transaction_cost_bps: float,
    base_position: pd.Series,
    vol_state: pd.Series,
    high_vol_indicator: pd.Series,
    quantile: float,
    cutoff_value: float,
    threshold_state: int,
    risk_multiplier: float,
) -> pd.DataFrame:
    artifact = frame.copy()
    index = artifact.index
    signal = _coerce_float_series(raw_signal, index=index, name="raw_signal")
    position = _coerce_float_series(executed_position, index=index, name="executed_position")
    base = _coerce_float_series(base_position, index=index, name="base_position")
    states = pd.Series(vol_state, index=index, name="vol_state", dtype=int)
    high_indicator = pd.Series(high_vol_indicator, index=index, name="high_vol_indicator", dtype=int)
    asset_returns = artifact[return_column].astype(float).rename("asset_return")
    benchmark_returns = artifact[benchmark_column].astype(float).rename("benchmark_return")
    turnover = compute_signal_turnover(position).rename("turnover")
    transaction_cost = (turnover * (transaction_cost_bps / 10_000.0)).rename("transaction_cost")
    gross = (position * asset_returns).rename("strategy_gross_return")
    net = (gross - transaction_cost).rename("strategy_net_return")

    return pd.DataFrame(
        {
            "date": artifact.index if "date" not in artifact.columns else artifact["date"].values,
            "split_id": split_id,
            "model_name": model_name,
            "asset": asset_name if asset_name is not None else np.nan,
            "asset_return": asset_returns.values,
            "benchmark_return": benchmark_returns.values,
            "raw_signal": signal.values,
            "prediction_probability": high_indicator.values,
            "target": artifact[target_column].values if target_column and target_column in artifact.columns else np.nan,
            "executed_position": position.values,
            "strategy_gross_return": gross.values,
            "strategy_net_return": net.values,
            "turnover": turnover.values,
            "transaction_cost": transaction_cost.values,
            "base_position": base.values,
            "vol_state": states.values,
            "high_vol_indicator": high_indicator.values,
            "vol_quantile": quantile,
            "vol_cutoff_value": cutoff_value,
            "overlay_threshold_state": threshold_state,
            "overlay_risk_multiplier": risk_multiplier,
            "overlay_mode": "vol_quantile_hard_veto",
            "is_cut_day": position.lt(base).astype(int).values,
        }
    )


def _select_overlay_parameters(
    *,
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    base_positions: pd.Series,
    validation_realized_volatility: pd.Series,
    train_realized_volatility: pd.Series,
    quantiles: tuple[float, ...],
    risk_multipliers: tuple[float, ...],
    transaction_cost_bps: float,
    annualization: int,
    decision_metric: str,
) -> tuple[VolQuantileOverlayParameterSelection, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    best_row: dict[str, float] | None = None
    best_evaluation: dict[str, Any] | None = None
    best_value = float("-inf")

    for quantile in quantiles:
        cutoffs = derive_train_volatility_cutoffs(train_realized_volatility, quantiles=(float(quantile),))
        validation_states = assign_volatility_states(validation_realized_volatility, cutoffs=cutoffs)
        high_state_ids = identify_high_vol_state_ids(validation_states)
        threshold_state = min(high_state_ids)
        parameter_grid = build_hard_overlay_parameter_grid(
            thresholds=[float(threshold_state)],
            risk_multipliers=list(risk_multipliers),
        )
        for params in parameter_grid:
            overlay_positions = apply_volatility_quantile_overlay(
                base_positions,
                validation_states,
                high_state_ids=high_state_ids,
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
                "quantile": float(quantile),
                "cutoff_value": float(next(iter(cutoffs.values()))),
                "threshold_state": float(threshold_state),
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
        raise ValueError("Volatility overlay parameter selection produced no candidates.")

    selection = VolQuantileOverlayParameterSelection(
        split_id=-1,
        quantile=float(best_row["quantile"]),
        cutoff_value=float(best_row["cutoff_value"]),
        threshold_state=int(best_row["threshold_state"]),
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
        ["validation_score", "quantile", "risk_multiplier"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _summarize_split_result(split_result: VolQuantileOverlaySplitResult) -> dict[str, float | int]:
    base_metrics = split_result.test_evaluation["base_metrics"]
    overlay_metrics = split_result.test_evaluation["overlay_metrics"]
    relative_metrics = split_result.test_evaluation["overlay_vs_base_metrics"]
    same_exposure = split_result.matched_nulls["null_summaries"]["same_average_exposure_random"]["summary"]
    same_turnover = split_result.matched_nulls["null_summaries"]["same_turnover_random"]["summary"]
    same_vol_state = split_result.matched_nulls["null_summaries"]["same_vol_state_exposure_random"]["summary"]
    return {
        "split_id": split_result.split_id,
        "quantile": split_result.selected_parameters.quantile,
        "cutoff_value": split_result.selected_parameters.cutoff_value,
        "threshold_state": split_result.selected_parameters.threshold_state,
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
        "same_vol_state_exposure_p_value": float(same_vol_state.p_value),
    }


def run_fold_local_vol_quantile_overlay_experiment(
    *,
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    benchmark_column: str = "benchmark_return_1d",
    return_column: str = "forward_simple_return_1d",
    target_column: str | None = "target_direction",
    asset_name: str | None = None,
    model_name: str = "vol_quantile_overlay_hard_veto",
    target_vol: float = 0.10,
    realized_vol_window: int = 20,
    annualization: int = 252,
    min_position: float = 0.0,
    max_position: float = 1.0,
    quantiles: tuple[float, ...] = (0.67, 0.75, 0.8),
    risk_multipliers: tuple[float, ...] = (0.0, 0.25, 0.5),
    transaction_cost_bps: float = 2.0,
    decision_metric: str = "information_ratio",
    null_n_runs: int = 100,
    null_seed: int = 42,
    include_block_bootstrap: bool = True,
) -> VolQuantileOverlayExperimentResult:
    """Run a fold-local volatility-quantile overlay experiment."""

    if not splits:
        raise ValueError("At least one walk-forward split is required.")
    if return_column not in frame.columns:
        raise KeyError(f"Missing required return column {return_column!r}.")
    if benchmark_column not in frame.columns:
        raise KeyError(f"Missing required benchmark column {benchmark_column!r}.")

    split_results: list[VolQuantileOverlaySplitResult] = []
    for split in splits:
        combined_frame = _concat_split_frames(split)
        combined_base_positions = build_vol_target_positions(
            combined_frame[return_column],
            target_vol=target_vol,
            realized_vol_window=realized_vol_window,
            annualization=annualization,
            min_position=min_position,
            max_position=max_position,
            split_ids=combined_frame["split_id"],
        )
        combined_realized_vol = compute_trailing_realized_volatility(
            combined_frame[return_column],
            window=realized_vol_window,
            annualization=annualization,
            split_ids=combined_frame["split_id"],
        )
        combined_frame["base_position"] = combined_base_positions.values
        combined_frame["trailing_realized_volatility"] = combined_realized_vol.values

        train_vol = combined_frame.loc[combined_frame["slice_name"] == "train", "trailing_realized_volatility"].astype(float)
        validation_vol = combined_frame.loc[
            combined_frame["slice_name"] == "validation",
            "trailing_realized_volatility",
        ].astype(float)
        validation_base = combined_frame.loc[combined_frame["slice_name"] == "validation", "base_position"].astype(float)
        validation_returns = split.validation[return_column].astype(float)
        validation_benchmark = split.validation[benchmark_column].astype(float)

        selection, validation_grid = _select_overlay_parameters(
            asset_returns=validation_returns,
            benchmark_returns=validation_benchmark,
            base_positions=validation_base,
            validation_realized_volatility=validation_vol,
            train_realized_volatility=train_vol,
            quantiles=quantiles,
            risk_multipliers=risk_multipliers,
            transaction_cost_bps=transaction_cost_bps,
            annualization=annualization,
            decision_metric=decision_metric,
        )
        selection = VolQuantileOverlayParameterSelection(
            split_id=split.split_id,
            quantile=selection.quantile,
            cutoff_value=selection.cutoff_value,
            threshold_state=selection.threshold_state,
            risk_multiplier=selection.risk_multiplier,
            decision_metric=selection.decision_metric,
            validation_score=selection.validation_score,
            validation_metrics=selection.validation_metrics,
        )

        test_vol = combined_frame.loc[combined_frame["slice_name"] == "test", "trailing_realized_volatility"].astype(float)
        test_states = assign_volatility_states(
            test_vol,
            cutoffs={f"q_{selection.quantile:.4f}": selection.cutoff_value},
        )
        high_state_ids = identify_high_vol_state_ids(test_states, min_state=selection.threshold_state)
        test_base = combined_frame.loc[combined_frame["slice_name"] == "test", "base_position"].astype(float)
        test_overlay = apply_volatility_quantile_overlay(
            test_base,
            test_states,
            high_state_ids=high_state_ids,
            risk_multiplier=selection.risk_multiplier,
        )
        test_evaluation = evaluate_overlay_vs_baseline(
            asset_returns=split.test[return_column].astype(float),
            benchmark_returns=split.test[benchmark_column].astype(float),
            base_positions=test_base,
            overlay_positions=test_overlay,
            transaction_cost_bps=transaction_cost_bps,
            annualization=annualization,
        )
        base_net_returns = (
            test_base.astype(float) * split.test[return_column].astype(float)
            - compute_signal_turnover(test_base.astype(float)) * (transaction_cost_bps / 10_000.0)
        )
        matched_nulls = run_volatility_matched_null_suite(
            positions=test_overlay,
            returns=split.test[return_column].astype(float),
            benchmark_returns=base_net_returns,
            vol_state_labels=test_states,
            n_runs=null_n_runs,
            seed=null_seed + split.split_id,
            annualization_factor=annualization,
            transaction_cost_bps=transaction_cost_bps,
            decision_metric=decision_metric,
            include_block_bootstrap=include_block_bootstrap,
        )

        high_indicator = test_states.isin(high_state_ids).astype(int)
        audit_artifact = _build_overlay_audit_artifact_frame(
            frame=split.test,
            split_id=split.split_id,
            model_name=model_name,
            asset_name=asset_name,
            raw_signal=test_overlay,
            executed_position=test_overlay,
            benchmark_column=benchmark_column,
            return_column=return_column,
            target_column=target_column,
            transaction_cost_bps=transaction_cost_bps,
            base_position=test_base,
            vol_state=test_states,
            high_vol_indicator=high_indicator,
            quantile=selection.quantile,
            cutoff_value=selection.cutoff_value,
            threshold_state=selection.threshold_state,
            risk_multiplier=selection.risk_multiplier,
        )
        split_results.append(
            VolQuantileOverlaySplitResult(
                split_id=split.split_id,
                selected_parameters=selection,
                validation_grid=validation_grid,
                test_evaluation=test_evaluation,
                matched_nulls=matched_nulls,
                audit_artifact=audit_artifact,
            )
        )

    summary = pd.DataFrame([_summarize_split_result(split_result) for split_result in split_results])
    audit_artifact_frame = pd.concat(
        [split_result.audit_artifact for split_result in split_results],
        axis=0,
        ignore_index=True,
    )
    return VolQuantileOverlayExperimentResult(
        split_results=split_results,
        summary=summary,
        audit_artifact_frame=audit_artifact_frame,
    )
