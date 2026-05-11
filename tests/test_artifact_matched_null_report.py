"""Tests for the artifact-only matched-null report helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_matched_null_report_from_oof_artifacts import (
    ArtifactColumnResolution,
    build_markdown_report,
    evaluate_oof_artifact,
    generate_report,
    inspect_oof_artifact_columns,
    reconstruct_executed_positions,
)


def test_inspect_oof_artifact_columns_resolves_saved_regime_stacking_shape() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "split_id": [1, 1, 1],
            "regime_id": [0, 1, 1],
            "forward_simple_return_1d": [0.01, -0.005, 0.004],
            "benchmark_return_1d": [0.008, -0.004, 0.003],
            "probability": [0.51, 0.48, 0.54],
            "prediction": [1, 0, 1],
        }
    )

    resolution = inspect_oof_artifact_columns(frame)

    assert resolution.date_col == "date"
    assert resolution.split_col == "split_id"
    assert resolution.asset_return_col == "forward_simple_return_1d"
    assert resolution.benchmark_return_col == "benchmark_return_1d"
    assert resolution.prediction_col == "prediction"
    assert resolution.regime_col == "regime_id"
    assert resolution.can_evaluate is True


def test_reconstruct_executed_positions_is_split_local_when_prediction_only() -> None:
    frame = pd.DataFrame(
        {
            "split_id": [1, 1, 1, 2, 2, 2],
            "prediction": [1, 1, 0, 1, 0, 1],
        }
    )
    resolution = ArtifactColumnResolution(
        date_col=None,
        split_col="split_id",
        asset_return_col="forward_simple_return_1d",
        benchmark_return_col="benchmark_return_1d",
        executed_position_col=None,
        prediction_col="prediction",
        probability_col=None,
        regime_col=None,
        model_col=None,
    )

    executed, source = reconstruct_executed_positions(frame, resolution)

    assert executed.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert "split-local shift" in source


def test_evaluate_oof_artifact_omits_regime_null_when_regime_labels_missing(tmp_path: Path) -> None:
    path = tmp_path / "artifact.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "split_id": [1] * 4 + [2] * 4,
            "forward_simple_return_1d": [0.01, -0.005, 0.007, -0.004, 0.006, -0.002, 0.004, -0.003],
            "benchmark_return_1d": [0.008, -0.004, 0.005, -0.003, 0.004, -0.001, 0.003, -0.002],
            "prediction": [1, 1, 0, 1, 0, 1, 1, 0],
        }
    ).to_csv(path, index=False)

    result = evaluate_oof_artifact(path, n_runs=5, seed=7, transaction_cost_bps=2.0)

    assert result.canonical_metrics is not None
    assert result.null_summaries is not None
    assert "same_regime_exposure_random" not in result.null_summaries
    assert any("same-regime-exposure null was omitted" in note for note in result.notes)


def test_generate_report_writes_new_markdown_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "split_id": [1] * 4 + [2] * 4,
            "regime_id": [0, 0, 1, 1, 0, 0, 1, 1],
            "forward_simple_return_1d": [0.01, -0.005, 0.007, -0.004, 0.006, -0.002, 0.004, -0.003],
            "benchmark_return_1d": [0.008, -0.004, 0.005, -0.003, 0.004, -0.001, 0.003, -0.002],
            "probability": [0.51, 0.52, 0.47, 0.55, 0.44, 0.58, 0.62, 0.39],
            "prediction": [1, 1, 0, 1, 0, 1, 1, 0],
        }
    ).to_csv(artifact, index=False)

    output_path = generate_report(
        [artifact],
        output_dir=tmp_path,
        n_runs=5,
        seed=11,
        transaction_cost_bps=2.0,
        decision_metric="information_ratio",
    )

    contents = output_path.read_text(encoding="utf-8")
    assert output_path.exists()
    assert "Partial Artifact-Only Matched-Null Audit Report" in contents
    assert "Matched Null Diagnostics" in contents
    assert "same_regime_exposure_random" in contents
