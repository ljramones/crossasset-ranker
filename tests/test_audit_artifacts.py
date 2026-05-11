"""Tests for standardized audit artifact schema helpers."""

from __future__ import annotations

import pandas as pd

from evaluation.audit_artifacts import STANDARD_AUDIT_COLUMNS, build_standard_audit_artifact_frame
from scripts.generate_matched_null_report_from_oof_artifacts import inspect_oof_artifact_columns


def test_build_standard_audit_artifact_frame_emits_required_columns() -> None:
    frame = pd.DataFrame(
        {
            "target_direction": [1, 0, 1],
            "forward_simple_return_1d": [0.01, -0.02, 0.03],
            "benchmark_return_1d": [0.008, -0.01, 0.02],
            "regime_id": [0, 1, 1],
            "regime_prob_0": [0.7, 0.2, 0.1],
            "regime_prob_1": [0.2, 0.7, 0.8],
            "regime_prob_2": [0.1, 0.1, 0.1],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    prediction = pd.Series([1, 1, 0], index=frame.index)
    probability = pd.Series([0.6, 0.55, 0.45], index=frame.index)

    artifact = build_standard_audit_artifact_frame(
        frame=frame,
        label="demo_model",
        prediction=prediction,
        probability=probability,
        split_id=7,
        transaction_cost_bps=2.0,
        asset_name="SPY",
    )

    for column in STANDARD_AUDIT_COLUMNS:
        if column.startswith("regime_prob_") or column == "asset":
            assert column in artifact.columns
        else:
            assert column in artifact.columns
    assert artifact["executed_position"].tolist() == [0.0, 1.0, 1.0]
    assert artifact["raw_signal"].tolist() == [1.0, 1.0, 0.0]
    assert artifact["prediction_probability"].tolist() == [0.6, 0.55, 0.45]
    assert artifact["model_name"].tolist() == ["demo_model"] * 3
    assert artifact["asset"].tolist() == ["SPY"] * 3


def test_artifact_report_inspection_recognizes_standardized_schema() -> None:
    artifact = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "split_id": [1, 1],
            "model_name": ["demo_model", "demo_model"],
            "asset": ["SPY", "SPY"],
            "asset_return": [0.01, -0.02],
            "benchmark_return": [0.008, -0.01],
            "raw_signal": [1.0, 0.0],
            "prediction_probability": [0.6, 0.4],
            "target": [1, 0],
            "executed_position": [0.0, 1.0],
            "strategy_gross_return": [0.0, -0.02],
            "strategy_net_return": [0.0, -0.0202],
            "turnover": [0.0, 1.0],
            "transaction_cost": [0.0, 0.0002],
            "regime_id": [0, 1],
        }
    )

    resolution = inspect_oof_artifact_columns(artifact)

    assert resolution.asset_return_col == "asset_return"
    assert resolution.benchmark_return_col == "benchmark_return"
    assert resolution.executed_position_col == "executed_position"
    assert resolution.prediction_col == "raw_signal"
    assert resolution.probability_col == "prediction_probability"
    assert resolution.model_col == "model_name"
    assert resolution.regime_col == "regime_id"
