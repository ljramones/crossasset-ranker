"""Synthetic tests for volatility-quantile overlay utilities."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from evaluation.volatility_overlay import (
    apply_volatility_quantile_overlay,
    assign_volatility_states,
    build_vol_state_from_quantile,
    compute_trailing_realized_volatility,
    derive_train_volatility_cutoffs,
    identify_high_vol_state_ids,
    run_volatility_matched_null_suite,
    same_vol_state_exposure_random,
)


def test_compute_trailing_realized_volatility_resets_inside_splits() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.015, -0.01, 0.012, 0.008, -0.03, 0.025],
        index=pd.date_range("2020-01-01", periods=8, freq="D"),
    )
    split_ids = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=returns.index)

    realized = compute_trailing_realized_volatility(returns, window=2, split_ids=split_ids)

    assert realized.iloc[0] != realized.iloc[0]  # NaN
    assert realized.iloc[1] != realized.iloc[1]
    assert realized.iloc[4] != realized.iloc[4]
    assert realized.iloc[5] != realized.iloc[5]


def test_derive_train_volatility_cutoffs_and_assign_states() -> None:
    realized = pd.Series([0.10, 0.12, 0.15, 0.20, 0.22, 0.30])
    cutoffs = derive_train_volatility_cutoffs(realized, quantiles=(0.5, 0.8))
    states = assign_volatility_states(realized, cutoffs=cutoffs)

    assert set(cutoffs) == {"q_0.5000", "q_0.8000"}
    assert states.min() == 0
    assert states.max() == 2


def test_identify_high_vol_state_ids_defaults_to_highest_state() -> None:
    states = pd.Series([0, 0, 1, 1, 2, 2])
    assert identify_high_vol_state_ids(states) == [2]
    assert identify_high_vol_state_ids(states, min_state=1) == [1, 2]


def test_build_vol_state_from_quantile_returns_labels_and_cutoffs() -> None:
    realized = pd.Series([0.10, 0.11, 0.12, 0.20, 0.21])
    labels, cutoffs = build_vol_state_from_quantile(realized, quantile=0.8)

    assert len(labels) == len(realized)
    assert list(cutoffs) == ["q_0.8000"]


def test_apply_volatility_quantile_overlay_cuts_only_high_states() -> None:
    base = pd.Series([1.0, 0.8, 0.6, 0.4])
    states = pd.Series([0, 1, 1, 0])

    overlay = apply_volatility_quantile_overlay(base, states, high_state_ids=[1], risk_multiplier=0.25)

    assert overlay.tolist() == [1.0, 0.2, 0.15, 0.4]


def test_apply_volatility_quantile_overlay_returns_base_when_no_high_states_present() -> None:
    base = pd.Series([1.0, 0.8, 0.6, 0.4])
    states = pd.Series([0, 0, 0, 0])

    overlay = apply_volatility_quantile_overlay(base, states, high_state_ids=[], risk_multiplier=0.0)

    assert overlay.tolist() == base.tolist()


def test_same_vol_state_exposure_random_preserves_per_state_multiset() -> None:
    position = pd.Series([0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.5, 1.0], name="position")
    states = pd.Series([0, 0, 1, 1, 1, 2, 2, 2], name="vol_state")

    randomized = same_vol_state_exposure_random(position, states, seed=11)

    for state_id in sorted(states.unique()):
        original_values = position.loc[states == state_id].tolist()
        randomized_values = randomized.loc[states == state_id].tolist()
        assert Counter(original_values) == Counter(randomized_values)


def test_run_volatility_matched_null_suite_includes_same_vol_state_null() -> None:
    position = pd.Series([0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0], name="position")
    returns = pd.Series([0.01, -0.005, 0.004, 0.003, -0.002, 0.005, 0.002, -0.001], name="returns")
    benchmark = pd.Series([0.0] * len(position), name="benchmark")
    states = pd.Series([0, 0, 1, 1, 1, 2, 2, 2], name="vol_state")

    suite = run_volatility_matched_null_suite(
        positions=position,
        returns=returns,
        benchmark_returns=benchmark,
        vol_state_labels=states,
        n_runs=5,
        seed=23,
        include_block_bootstrap=False,
    )

    assert "same_vol_state_exposure_random" in suite["null_summaries"]
    assert suite["null_summaries"]["same_vol_state_exposure_random"]["summary"].n_runs == 5
