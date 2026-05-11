"""Synthetic tests for regime overlay foundation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.regime_overlay import (
    apply_hard_regime_overlay,
    apply_soft_regime_overlay,
    build_hard_overlay_parameter_grid,
    build_vol_target_positions,
    characterize_regimes,
    compute_regime_danger_scores,
    evaluate_overlay_vs_baseline,
    evaluate_overlay_strategy,
    extract_dangerous_regime_probability,
    identify_dangerous_regime,
    summarize_overlay_position_change,
)


def test_build_vol_target_positions_resets_inside_splits() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.015, -0.01, 0.012, 0.008, -0.03, 0.025],
        index=pd.date_range("2020-01-01", periods=8, freq="D"),
    )
    split_ids = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=returns.index)

    positions = build_vol_target_positions(
        returns,
        target_vol=0.10,
        realized_vol_window=2,
        annualization=252,
        min_position=0.0,
        max_position=1.0,
        split_ids=split_ids,
    )

    assert positions.iloc[0] == 0.0
    assert positions.iloc[1] == 0.0
    assert positions.iloc[4] == 0.0
    assert positions.iloc[5] == 0.0
    assert positions.between(0.0, 1.0).all()


def test_characterize_regimes_and_identify_dangerous_regime_are_fold_local() -> None:
    returns = pd.Series([0.01, 0.012, -0.03, -0.025, 0.008, 0.009, -0.04, -0.015])
    regimes = pd.Series([0, 0, 1, 1, 2, 2, 1, 1])

    summary = characterize_regimes(returns, regimes, annualization=252)
    scored = compute_regime_danger_scores(summary)
    dangerous = identify_dangerous_regime(summary)

    assert {"regime_id", "average_return", "realized_volatility", "danger_score"} <= set(scored.columns)
    assert dangerous == 1


def test_extract_dangerous_regime_probability_resolves_standard_column() -> None:
    probabilities = pd.DataFrame(
        {
            "regime_prob_0": [0.7, 0.2],
            "regime_prob_1": [0.2, 0.6],
            "regime_prob_2": [0.1, 0.2],
        }
    )

    extracted = extract_dangerous_regime_probability(probabilities, 1)

    assert extracted.tolist() == [0.2, 0.6]


def test_apply_hard_regime_overlay_cuts_only_when_probability_exceeds_threshold() -> None:
    base = pd.Series([1.0, 0.8, 0.6, 0.4])
    probability = pd.Series([0.4, 0.7, 0.2, 0.9])

    overlay = apply_hard_regime_overlay(base, probability, threshold=0.6, risk_multiplier=0.25)

    assert overlay.tolist() == [1.0, 0.2, 0.6, 0.1]


def test_apply_soft_regime_overlay_scales_positions_smoothly() -> None:
    base = pd.Series([1.0, 0.8, 0.6])
    probability = pd.Series([0.0, 0.5, 1.0])

    overlay = apply_soft_regime_overlay(base, probability, cut_strength=0.5)

    assert overlay.tolist() == pytest.approx([1.0, 0.6, 0.3])


def test_build_hard_overlay_parameter_grid_returns_all_pairs() -> None:
    grid = build_hard_overlay_parameter_grid(
        thresholds=[0.5, 0.6],
        risk_multipliers=[0.0, 0.25],
    )

    assert grid == [
        {"threshold": 0.5, "risk_multiplier": 0.0},
        {"threshold": 0.5, "risk_multiplier": 0.25},
        {"threshold": 0.6, "risk_multiplier": 0.0},
        {"threshold": 0.6, "risk_multiplier": 0.25},
    ]


def test_summarize_overlay_position_change_reports_turnover_and_cut_days() -> None:
    base = pd.Series([0.0, 1.0, 1.0, 1.0, 0.5])
    overlay = pd.Series([0.0, 1.0, 0.5, 0.5, 0.25])

    summary = summarize_overlay_position_change(base, overlay)

    assert summary["fraction_cut_days"] == pytest.approx(0.6)
    assert "overlay_daily_turnover" in summary
    assert summary["average_overlay_position"] < summary["average_base_position"]


def test_evaluate_overlay_strategy_returns_active_metrics() -> None:
    positions = pd.Series([0.0, 1.0, 1.0, 0.5, 0.5])
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.01])
    benchmark = pd.Series([0.0, 0.005, 0.0, 0.001, 0.002])

    metrics = evaluate_overlay_strategy(
        asset_returns=returns,
        benchmark_returns=benchmark,
        positions=positions,
        transaction_cost_bps=2.0,
        annualization=252,
    )

    assert {"information_ratio", "active_calmar", "fraction_in_market", "daily_turnover"} <= set(metrics)


def test_evaluate_overlay_vs_baseline_returns_overlay_relative_metrics() -> None:
    base_positions = pd.Series([0.0, 1.0, 1.0, 1.0, 0.8])
    overlay_positions = pd.Series([0.0, 1.0, 0.5, 0.5, 0.4])
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.01])
    benchmark = pd.Series([0.0, 0.005, 0.0, 0.001, 0.002])

    evaluation = evaluate_overlay_vs_baseline(
        asset_returns=returns,
        benchmark_returns=benchmark,
        base_positions=base_positions,
        overlay_positions=overlay_positions,
        transaction_cost_bps=2.0,
        annualization=252,
    )

    assert {"base_metrics", "overlay_metrics", "overlay_vs_base_metrics", "position_change_summary"} <= set(evaluation)
    assert "information_ratio" in evaluation["overlay_vs_base_metrics"]
    assert evaluation["position_change_summary"]["average_overlay_position"] < evaluation["position_change_summary"]["average_base_position"]
