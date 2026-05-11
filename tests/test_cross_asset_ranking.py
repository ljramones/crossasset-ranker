"""Tests for the cross-asset ranking helpers in evaluation/cross_asset_ranking.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.cross_asset_ranking import (
    add_cross_sectional_ranks,
    apply_rebalance_schedule,
    build_cross_asset_panel,
    build_equal_weight_allocations,
    build_lambdarank_groups,
    build_top_k_allocations,
    compute_allocation_returns,
    compute_forward_return,
    compute_forward_risk_adjusted_return,
    compute_trailing_realized_vol,
    empirical_pvalue_one_sided_ge,
    make_lambdarank_relevance_labels,
    normalize_features_per_asset_train_only,
    random_top_k_allocations,
    select_cross_asset_feature_columns,
)


def _synthetic_asset_frame(
    *,
    n_rows: int = 60,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 0,
    start: str = "2020-01-02",
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_rows, freq="B")
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, size=n_rows)
    return pd.DataFrame(
        {
            "date": dates,
            "return_1d": returns,
            "return_5d": pd.Series(returns).rolling(5).sum().values,
            "return_20d": pd.Series(returns).rolling(20).sum().values,
            "vol_ratio": rng.normal(1.0, 0.1, size=n_rows),
        }
    )


def test_compute_forward_return_uses_future_only_and_nans_tail() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.0, 0.01], index=pd.date_range("2020-01-01", periods=6))

    forward = compute_forward_return(returns, horizon=2)

    expected_first = (1 + 0.02) * (1 + (-0.01)) - 1
    assert forward.iloc[0] == pytest.approx(expected_first)
    assert pd.isna(forward.iloc[-2])
    assert pd.isna(forward.iloc[-1])


def test_compute_trailing_realized_vol_uses_only_past() -> None:
    returns = pd.Series(np.linspace(0.01, 0.05, 10))

    trailing = compute_trailing_realized_vol(returns, window=3, annualization=252)

    assert pd.isna(trailing.iloc[0])
    assert pd.isna(trailing.iloc[1])
    assert pd.isna(trailing.iloc[2])
    assert trailing.iloc[3] == pytest.approx(returns.iloc[:3].std(ddof=1) * np.sqrt(252))


def test_compute_forward_risk_adjusted_return_handles_zero_vol() -> None:
    returns = pd.Series([0.0] * 10 + [0.02] * 10)

    target = compute_forward_risk_adjusted_return(returns, forward_horizon=2, vol_window=5, annualization=252)

    assert target.notna().any(), "Some rows should produce a valid target."
    early_indices = target.index[:5]
    assert target.loc[early_indices].isna().any(), "Early rows with zero trailing vol must yield NaN target."


def test_cross_sectional_ranks_are_correct_per_date() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "forward_20d_risk_adjusted_return": [0.5, 1.0, -0.2, 0.1, -0.1, 0.05],
        }
    )

    ranked = add_cross_sectional_ranks(panel)

    day1 = ranked[ranked["date"] == pd.Timestamp("2020-01-01")].set_index("asset")
    assert day1.loc["B", "cross_sectional_rank"] == 1.0
    assert day1.loc["A", "cross_sectional_rank"] == 2.0
    assert day1.loc["C", "cross_sectional_rank"] == 3.0
    assert int(day1.loc["B", "is_top_1"]) == 1
    assert int(day1.loc["A", "is_top_2"]) == 1
    assert int(day1.loc["C", "is_top_2"]) == 0


def test_feature_selector_excludes_future_target_rank_and_raw_columns() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "asset": ["A"],
            "Open": [1.0],
            "Close": [1.0],
            "Adj Close": [1.0],
            "Volume": [1.0],
            "BenchmarkClose": [1.0],
            "VIXClose": [1.0],
            "return_1d": [0.0],
            "return_5d": [0.0],
            "vol_ratio": [1.0],
            "forward_20d_return": [0.0],
            "forward_20d_risk_adjusted_return": [0.0],
            "trailing_20d_realized_vol": [0.1],
            "future_max_drawdown_20d": [0.0],
            "target_drawdown_event_20d_3pct": [0.0],
            "cross_sectional_rank": [1.0],
            "is_top_1": [1],
            "split_id": [0],
        }
    )

    features = select_cross_asset_feature_columns(panel, target_col="forward_20d_risk_adjusted_return")

    assert "return_1d" in features
    assert "return_5d" in features
    assert "vol_ratio" in features
    for forbidden in (
        "Open",
        "Close",
        "Adj Close",
        "Volume",
        "BenchmarkClose",
        "VIXClose",
        "forward_20d_return",
        "forward_20d_risk_adjusted_return",
        "trailing_20d_realized_vol",
        "future_max_drawdown_20d",
        "target_drawdown_event_20d_3pct",
        "cross_sectional_rank",
        "is_top_1",
        "split_id",
        "date",
        "asset",
    ):
        assert forbidden not in features


def test_top_k_allocation_selects_correct_assets_and_equal_weights() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "asset": ["A", "B", "C", "D"],
            "model_score": [0.1, 0.5, -0.2, 0.4],
        }
    )

    alloc = build_top_k_allocations(scored, score_col="model_score", k=2)

    selected = alloc[alloc["weight"] > 0].sort_values("asset")
    assert sorted(selected["asset"].tolist()) == ["B", "D"]
    np.testing.assert_allclose(selected["weight"].to_numpy(), np.array([0.5, 0.5]))
    not_selected = alloc[alloc["weight"] == 0.0]
    assert sorted(not_selected["asset"].tolist()) == ["A", "C"]


def test_allocation_returns_use_one_bar_lag_and_apply_costs() -> None:
    dates = pd.date_range("2020-01-01", periods=4)
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "return_1d": [0.0, 0.10, 0.10, 0.10] + [0.0, -0.05, -0.05, -0.05],
        }
    )
    allocations = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "weight": [1.0, 1.0, 1.0, 1.0] + [0.0, 0.0, 0.0, 0.0],
        }
    )

    gross, net, turnover = compute_allocation_returns(
        allocations,
        panel,
        transaction_cost_bps=10.0,
    )

    # Day 0: lagged weight is 0 (no prior weight), so gross/net = 0; turnover = 1.0 (allocation appearing).
    assert gross.iloc[0] == pytest.approx(0.0)
    assert turnover.iloc[0] == pytest.approx(1.0)
    # Day 1: lagged weight from day 0 = 1.0 in A; A's return on day 1 = 0.10. Gross = 0.10.
    assert gross.iloc[1] == pytest.approx(0.10)
    # Net = gross - turnover * cost. Turnover on day 1 = 0 (no weight change), so net = gross.
    assert net.iloc[1] == pytest.approx(0.10)


def test_random_top_k_allocations_are_deterministic_under_seed() -> None:
    panel = pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=3)) * 4,
            "asset": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
            "return_1d": np.zeros(12),
        }
    )

    runs_a = random_top_k_allocations(panel, k=2, n_runs=5, random_state=42)
    runs_b = random_top_k_allocations(panel, k=2, n_runs=5, random_state=42)

    assert list(runs_a.keys()) == list(runs_b.keys())
    for label in runs_a:
        pd.testing.assert_frame_equal(runs_a[label], runs_b[label])

    selected_per_date = (
        runs_a["run_0000"][runs_a["run_0000"]["weight"] > 0]
        .groupby("date")["asset"]
        .nunique()
    )
    assert (selected_per_date == 2).all(), "Each date must select exactly k assets in a top-k random null."


def test_build_cross_asset_panel_inner_joins_dates() -> None:
    frame_a = _synthetic_asset_frame(n_rows=40, seed=1, start="2020-01-02")
    frame_b = _synthetic_asset_frame(n_rows=40, seed=2, start="2020-01-09")  # starts later

    panel = build_cross_asset_panel({"A": frame_a, "B": frame_b})

    assert sorted(panel["asset"].unique().tolist()) == ["A", "B"]
    panel_dates_a = panel[panel["asset"] == "A"]["date"]
    panel_dates_b = panel[panel["asset"] == "B"]["date"]
    pd.testing.assert_series_equal(
        panel_dates_a.reset_index(drop=True),
        panel_dates_b.reset_index(drop=True),
    )
    assert panel["date"].min() >= frame_b["date"].min()


def test_empirical_pvalue_smoothed_one_sided() -> None:
    distribution = [0.0, 0.1, 0.2, 0.3, 0.4]
    assert empirical_pvalue_one_sided_ge(0.5, distribution) == pytest.approx(1 / 6)
    assert empirical_pvalue_one_sided_ge(0.0, distribution) == pytest.approx(6 / 6)
    assert empirical_pvalue_one_sided_ge(0.25, distribution) == pytest.approx(3 / 6)


def _daily_top_k_alloc_for_rebalance_tests(*, n_dates: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a panel + scored top-2 daily allocation that visibly varies across dates."""
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    rng = np.random.default_rng(0)
    rows: list[pd.DataFrame] = []
    for asset, mu in zip(["A", "B", "C", "D"], [0.001, 0.0, -0.0005, 0.0008]):
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "return_1d": rng.normal(mu, 0.005, size=n_dates),
                    "model_score": rng.normal(0.0, 1.0, size=n_dates),
                }
            )
        )
    panel = pd.concat(rows, axis=0, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    daily_alloc = build_top_k_allocations(panel, score_col="model_score", k=2)
    return panel, daily_alloc


def test_apply_rebalance_schedule_identity_when_every_one() -> None:
    _, daily = _daily_top_k_alloc_for_rebalance_tests()
    rebalanced = apply_rebalance_schedule(daily, rebalance_every=1)
    pd.testing.assert_frame_equal(daily.reset_index(drop=True), rebalanced.reset_index(drop=True))


def test_apply_rebalance_schedule_holds_weights_between_rebalance_dates() -> None:
    _, daily = _daily_top_k_alloc_for_rebalance_tests(n_dates=15)
    rebalanced = apply_rebalance_schedule(daily, rebalance_every=5)

    weight_matrix = rebalanced.pivot(index="date", columns="asset", values="weight").sort_index()
    # Day 0 is a rebalance day. Days 1, 2, 3, 4 must equal day 0.
    np.testing.assert_allclose(
        weight_matrix.iloc[1:5].to_numpy(),
        np.tile(weight_matrix.iloc[0].to_numpy(), (4, 1)),
    )
    # Day 5 is the next rebalance day — it can change.
    # Days 6-9 must equal day 5.
    np.testing.assert_allclose(
        weight_matrix.iloc[6:10].to_numpy(),
        np.tile(weight_matrix.iloc[5].to_numpy(), (4, 1)),
    )


def test_apply_rebalance_schedule_reduces_turnover() -> None:
    panel, daily = _daily_top_k_alloc_for_rebalance_tests(n_dates=40)
    weekly = apply_rebalance_schedule(daily, rebalance_every=5)
    monthly = apply_rebalance_schedule(daily, rebalance_every=20)

    _, _, daily_turnover = compute_allocation_returns(daily, panel, transaction_cost_bps=2.0)
    _, _, weekly_turnover = compute_allocation_returns(weekly, panel, transaction_cost_bps=2.0)
    _, _, monthly_turnover = compute_allocation_returns(monthly, panel, transaction_cost_bps=2.0)

    # Skip day 0 — every schedule incurs the same "appearance" turnover there.
    assert weekly_turnover.iloc[1:].sum() < daily_turnover.iloc[1:].sum()
    assert monthly_turnover.iloc[1:].sum() < weekly_turnover.iloc[1:].sum()


def test_apply_rebalance_schedule_preserves_one_bar_lag() -> None:
    """Day-1 gross return must come from day-0 weights regardless of rebalance schedule.

    Construction: A returns +10% on day 1, 0 elsewhere; B returns +10% on day 2, 0
    elsewhere. Daily allocation alternates A and B. With one-bar lag, day-1 strategy
    return = lagged-weight-from-day-0 × day-1-asset-returns. Day 0's weight is
    A=1 (B=0) in both daily and rebalance_every=2 schedules, so day-1 must be 0.10
    in both. (Day 2 will diverge — that's expected.)
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "return_1d": [0.0, 0.10, 0.0, 0.0] + [0.0, 0.0, 0.10, 0.0],
        }
    )
    daily_alloc = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "weight": [1.0, 0.0, 1.0, 0.0] + [0.0, 1.0, 0.0, 1.0],
        }
    )
    rebalanced = apply_rebalance_schedule(daily_alloc, rebalance_every=2)

    daily_gross, _, _ = compute_allocation_returns(daily_alloc, panel, transaction_cost_bps=0.0)
    reb_gross, _, _ = compute_allocation_returns(rebalanced, panel, transaction_cost_bps=0.0)

    assert daily_gross.iloc[1] == pytest.approx(0.10)
    assert reb_gross.iloc[1] == pytest.approx(0.10)
    # Day-2: daily picks up B at +10% (weight on day 1 was B); rebalanced still
    # holds A (frozen day-0 weight), and A returns 0 on day 2. They MUST differ.
    assert daily_gross.iloc[2] == pytest.approx(0.10)
    assert reb_gross.iloc[2] == pytest.approx(0.0)


def _normalization_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Two assets on different scales; train block is a clean tail of dates."""
    train_dates = pd.date_range("2020-01-01", periods=10, freq="B")
    test_dates = pd.date_range(train_dates[-1] + pd.Timedelta(days=1), periods=4, freq="B")
    all_dates = train_dates.append(test_dates)
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for asset, scale in [("A", 1.0), ("B", 100.0)]:
        for d in all_dates:
            rows.append(
                {
                    "date": d,
                    "asset": asset,
                    # in-train rows: feature ~ N(0, scale); test rows: huge outliers
                    "return_1d": (
                        scale * rng.normal(0.0, 1.0)
                        if d in train_dates
                        else scale * 50.0
                    ),
                    "vol_ratio": rng.uniform(0.5, 1.5),
                    "constant_feature": 7.0,  # zero-std test
                    "forward_20d_return": rng.normal(0.0, 1.0),  # must NOT be normalized
                    "is_top_1": int(rng.integers(0, 2)),  # rank-like: must NOT be normalized
                }
            )
    panel = pd.DataFrame(rows).sort_values(["date", "asset"]).reset_index(drop=True)
    return panel, train_dates


def test_normalize_features_per_asset_uses_train_only_stats_per_asset() -> None:
    panel, train_dates = _normalization_panel()
    normalized = normalize_features_per_asset_train_only(
        panel,
        train_dates=train_dates,
        feature_columns=["return_1d", "vol_ratio", "constant_feature"],
    )

    train_normalized = normalized[normalized["date"].isin(train_dates)]
    a_train = train_normalized[train_normalized["asset"] == "A"]["return_1d"]
    b_train = train_normalized[train_normalized["asset"] == "B"]["return_1d"]

    # Train-block z-scores should be roughly mean 0 with std order ~1 per asset.
    assert abs(a_train.mean()) < 1e-9
    assert abs(b_train.mean()) < 1e-9
    assert 0.5 < a_train.std() < 2.0
    assert 0.5 < b_train.std() < 2.0


def test_normalize_features_outliers_in_test_do_not_affect_train_zscore() -> None:
    panel, train_dates = _normalization_panel()
    # Same panel but inflate the test outliers further — train z-scores must not change.
    panel_outlier = panel.copy()
    panel_outlier.loc[~panel_outlier["date"].isin(train_dates), "return_1d"] = 1e6

    a = normalize_features_per_asset_train_only(
        panel,
        train_dates=train_dates,
        feature_columns=["return_1d"],
    )
    b = normalize_features_per_asset_train_only(
        panel_outlier,
        train_dates=train_dates,
        feature_columns=["return_1d"],
    )
    a_train = a[a["date"].isin(train_dates)].sort_values(["date", "asset"]).reset_index(drop=True)
    b_train = b[b["date"].isin(train_dates)].sort_values(["date", "asset"]).reset_index(drop=True)
    pd.testing.assert_series_equal(a_train["return_1d"], b_train["return_1d"])


def test_normalize_features_does_not_touch_target_or_rank_columns() -> None:
    panel, train_dates = _normalization_panel()
    normalized = normalize_features_per_asset_train_only(
        panel,
        train_dates=train_dates,
        feature_columns=["return_1d", "vol_ratio", "constant_feature"],
    )

    pd.testing.assert_series_equal(
        panel["forward_20d_return"].reset_index(drop=True),
        normalized["forward_20d_return"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        panel["is_top_1"].reset_index(drop=True),
        normalized["is_top_1"].reset_index(drop=True),
    )


def test_normalize_features_zero_std_safely_maps_to_zero() -> None:
    panel, train_dates = _normalization_panel()
    normalized = normalize_features_per_asset_train_only(
        panel,
        train_dates=train_dates,
        feature_columns=["constant_feature"],
    )
    assert normalized["constant_feature"].notna().all()
    np.testing.assert_allclose(normalized["constant_feature"].to_numpy(), 0.0)


def test_normalize_features_preserves_row_order() -> None:
    panel, train_dates = _normalization_panel()
    normalized = normalize_features_per_asset_train_only(
        panel,
        train_dates=train_dates,
        feature_columns=["return_1d", "vol_ratio"],
    )
    pd.testing.assert_frame_equal(
        panel[["date", "asset"]].reset_index(drop=True),
        normalized[["date", "asset"]].reset_index(drop=True),
    )


def test_build_equal_weight_allocations_distributes_across_universe() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "return_1d": [0.0] * 6,
        }
    )

    alloc = build_equal_weight_allocations(panel)

    np.testing.assert_allclose(alloc["weight"].to_numpy(), np.full(len(alloc), 1.0 / 3.0))
    weights_per_date = alloc.groupby("date")["weight"].sum()
    np.testing.assert_allclose(weights_per_date.to_numpy(), np.ones(2))


def test_make_lambdarank_relevance_labels_assigns_per_date_integer_ranks() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "forward_20d_risk_adjusted_return": [0.1, 0.5, -0.2, 0.3, 0.05, 0.2],
        }
    )

    relevance = make_lambdarank_relevance_labels(panel)

    day1 = panel[panel["date"] == pd.Timestamp("2020-01-01")].copy()
    day1["rel"] = relevance.loc[day1.index]
    by_asset = day1.set_index("asset")["rel"].astype(int).to_dict()
    assert by_asset == {"B": 2, "A": 1, "C": 0}

    day2 = panel[panel["date"] == pd.Timestamp("2020-01-02")].copy()
    day2["rel"] = relevance.loc[day2.index]
    by_asset_2 = day2.set_index("asset")["rel"].astype(int).to_dict()
    assert by_asset_2 == {"A": 2, "C": 1, "B": 0}


def test_make_lambdarank_relevance_labels_handles_nan_targets_safely() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "forward_20d_risk_adjusted_return": [0.1, np.nan, -0.2, np.nan, np.nan, np.nan],
        }
    )

    relevance = make_lambdarank_relevance_labels(panel)

    day1_indices = panel[panel["date"] == pd.Timestamp("2020-01-01")].index
    rel_day1 = {panel.loc[i, "asset"]: relevance.loc[i] for i in day1_indices}
    assert rel_day1["A"] == 1.0
    assert rel_day1["C"] == 0.0
    assert pd.isna(rel_day1["B"])
    day2_indices = panel[panel["date"] == pd.Timestamp("2020-01-02")].index
    for i in day2_indices:
        assert pd.isna(relevance.loc[i])


def test_make_lambdarank_relevance_labels_breaks_ties_deterministically() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3),
            "asset": ["A", "B", "C"],
            "forward_20d_risk_adjusted_return": [0.5, 0.5, 0.5],
        }
    )

    rel_a = make_lambdarank_relevance_labels(panel)
    rel_b = make_lambdarank_relevance_labels(panel)

    pd.testing.assert_series_equal(rel_a, rel_b)
    assert sorted(rel_a.dropna().astype(int).tolist()) == [0, 1, 2]


def test_build_lambdarank_groups_returns_per_date_sizes() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 2 + ["2020-01-03"] * 4),
            "asset": ["A", "B", "C", "A", "B", "A", "B", "C", "D"],
        }
    )

    groups = build_lambdarank_groups(panel)
    assert groups == [3, 2, 4]
    assert sum(groups) == len(panel)


def test_build_lambdarank_groups_empty_panel() -> None:
    assert build_lambdarank_groups(pd.DataFrame(columns=["date", "asset"])) == []
