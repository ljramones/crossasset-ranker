"""Synthetic tests for matched-null audit integration helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit import integrity_audit as audit


def test_run_matched_null_audit_uses_executed_positions_without_extra_shift() -> None:
    returns = pd.Series([0.01, -0.005, 0.004, 0.003, -0.002, 0.005], name="returns")
    benchmark = pd.Series([0.0, 0.001, 0.0, 0.001, 0.0, 0.001], name="benchmark")
    executed_position = pd.Series([0.0, 1.0, 1.0, 0.0, -1.0, -1.0], name="position")
    regimes = pd.Series([0, 0, 1, 1, 2, 2], name="regime_id")

    summary = audit._run_matched_null_audit(
        model_name="synthetic_model",
        asset_returns=returns,
        canonical_positions=executed_position,
        benchmark_returns=benchmark,
        regime_labels=regimes,
        transaction_cost_bps=2.0,
        n_runs=5,
        random_state=17,
        decision_metric="information_ratio",
        significance_threshold=0.05,
    )

    assert summary["matched_null_canonical_fraction_in_market"] == executed_position.ne(0.0).mean()
    assert summary["matched_null_canonical_position_flip_count"] >= 1.0
    assert "matched_null__same_average_exposure_random__p_value" in summary
    assert "matched_null__same_regime_exposure_random__decision" in summary


def test_run_matched_null_audit_omits_same_regime_null_without_regime_labels() -> None:
    returns = pd.Series([0.01, -0.005, 0.007, -0.004, 0.006, -0.002, 0.004, -0.003], name="returns")
    benchmark = pd.Series([0.008, -0.004, 0.005, -0.003, 0.004, -0.001, 0.003, -0.002], name="benchmark")
    executed_position = pd.Series([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], name="position")

    summary = audit._run_matched_null_audit(
        model_name="synthetic_model_no_regime",
        asset_returns=returns,
        canonical_positions=executed_position,
        benchmark_returns=benchmark,
        regime_labels=None,
        transaction_cost_bps=2.0,
        n_runs=5,
        random_state=19,
        decision_metric="information_ratio",
        significance_threshold=0.05,
    )

    assert "matched_null__same_average_exposure_random__p_value" in summary
    assert "matched_null__same_turnover_random__p_value" in summary
    assert "matched_null__same_regime_exposure_random__p_value" not in summary


def test_collect_single_model_prediction_frame_shifts_within_split_only() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "split_id": [0, 0, 0, 1, 1, 1],
            "forward_simple_return_1d": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
            "benchmark_return_1d": [0.0] * 6,
            "prediction": [1, 1, 0, 1, 0, 1],
        }
    )

    collected = audit._collect_single_model_prediction_frame(frame)

    assert collected["executed_position"].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]


def test_write_comparative_null_report_includes_matched_null_sections(tmp_path: Path) -> None:
    comparative = pd.DataFrame(
        [
            {
                "model": "demo_model",
                "model_type": "single_model",
                "runs": 10,
                "canonical_information_ratio": 0.12,
                "mean_shuffled_information_ratio": 0.01,
                "percentile_95_shuffled_information_ratio": 0.08,
                "canonical_excess_net_sharpe": 0.05,
                "mean_shuffled_excess_net_sharpe": 0.00,
                "percentile_95_shuffled_excess_net_sharpe": 0.04,
                "p_value": 0.02,
                "significance_threshold": 0.05,
                "decision": "PASS",
                "canonical_net_sharpe": 1.1,
                "canonical_net_sortino": 1.3,
                "canonical_calmar": 1.4,
                "mean_shuffled_net_sharpe": 0.8,
                "percentile_95_shuffled_net_sharpe": 1.0,
                "canonical_average_long_exposure": 0.7,
                "canonical_average_position_size": 0.75,
                "canonical_fraction_positive_predictions": 0.72,
                "decision_metric": "information_ratio",
                "matched_null_runs": 5,
                "matched_null_canonical_information_ratio": 0.12,
                "matched_null_canonical_fraction_in_market": 0.75,
                "matched_null_canonical_daily_turnover": 0.2,
                "matched_null_canonical_position_flip_count": 3.0,
                "matched_null__same_average_exposure_random__p_value": 0.04,
                "matched_null__same_turnover_random__p_value": 0.10,
                "matched_null__same_exposure_and_turnover_random__p_value": 0.08,
                "matched_null__same_regime_exposure_random__p_value": 0.03,
                "matched_null__block_bootstrap_same_exposure_random__p_value": 0.09,
                "matched_null__same_average_exposure_random__decision": "PASS",
                "matched_null__same_turnover_random__decision": "FAIL",
                "matched_null__same_exposure_and_turnover_random__decision": "FAIL",
                "matched_null__same_regime_exposure_random__decision": "PASS",
                "matched_null__block_bootstrap_same_exposure_random__decision": "FAIL",
            }
        ]
    )

    path = tmp_path / "comparative_null_test_report.md"
    audit._write_comparative_null_report(path, comparative)
    contents = path.read_text(encoding="utf-8")

    assert "## Matched Null Decision Table" in contents
    assert "## Matched Null Decisions" in contents
    assert "same regime exposure" in contents.lower()


def test_write_integrity_report_includes_matched_null_section(tmp_path: Path) -> None:
    comparative = pd.DataFrame(
        [
            {
                "model": "demo_model",
                "model_type": "single_model",
                "runs": 10,
                "canonical_information_ratio": 0.12,
                "mean_shuffled_information_ratio": 0.01,
                "percentile_95_shuffled_information_ratio": 0.08,
                "canonical_excess_net_sharpe": 0.05,
                "mean_shuffled_excess_net_sharpe": 0.00,
                "percentile_95_shuffled_excess_net_sharpe": 0.04,
                "p_value": 0.02,
                "significance_threshold": 0.05,
                "decision": "PASS",
                "canonical_net_sharpe": 1.1,
                "mean_shuffled_net_sharpe": 0.8,
                "percentile_95_shuffled_net_sharpe": 1.0,
                "canonical_average_long_exposure": 0.7,
                "canonical_average_position_size": 0.75,
                "canonical_fraction_positive_predictions": 0.72,
                "matched_null_canonical_information_ratio": 0.12,
                "matched_null_canonical_fraction_in_market": 0.75,
                "matched_null_canonical_daily_turnover": 0.2,
                "matched_null__same_average_exposure_random__p_value": 0.04,
                "matched_null__same_turnover_random__p_value": 0.10,
                "matched_null__same_exposure_and_turnover_random__p_value": 0.08,
                "matched_null__same_regime_exposure_random__p_value": 0.03,
                "matched_null__block_bootstrap_same_exposure_random__p_value": 0.09,
            }
        ]
    )
    experiment = type(
        "ExperimentStub",
        (),
        {
            "config": {"project": {"run_profile": "test"}},
            "splits": [object(), object()],
        },
    )()
    checks = [
        audit.AuditCheck(
            name="synthetic_check",
            status="PASS",
            detail="Synthetic check for report rendering.",
            evidence={"foo": "bar"},
        )
    ]

    path = tmp_path / "integrity_audit_report.md"
    audit._write_report(path, checks=checks, experiment=experiment, comparative_results=comparative)
    contents = path.read_text(encoding="utf-8")

    assert "### Matched Null Decision Table" in contents
    assert "Benchmark-relative matched-null diagnostics use executed positions" in contents
    assert "same turnover p-value" in contents.lower()
