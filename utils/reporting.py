"""Reporting helpers for summarizing comparison results."""

from __future__ import annotations

import pandas as pd


def get_default_ranking_columns(frame: pd.DataFrame) -> list[str]:
    """Return the active-skill-first ranking columns available in one frame."""

    preferred = [
        "information_ratio",
        "active_calmar",
        "annualized_active_return",
        "net_sharpe",
        "sharpe",
    ]
    return [column for column in preferred if column in frame.columns]


def get_primary_ranking_label(frame: pd.DataFrame) -> str:
    """Return the label of the first available ranking metric."""

    ranking_columns = get_default_ranking_columns(frame)
    return ranking_columns[0] if ranking_columns else "model"


def summarize_results(
    results: list[dict[str, float]],
    ranking_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert raw evaluation dictionaries into a sorted comparison table."""

    frame = pd.DataFrame(results)
    if frame.empty:
        return frame

    if "average_net_exposure" not in frame.columns:
        if "average_long_exposure" in frame.columns and "average_short_exposure" in frame.columns:
            frame["average_net_exposure"] = frame["average_long_exposure"] - frame["average_short_exposure"]
        elif "average_long_exposure" in frame.columns:
            frame["average_net_exposure"] = frame["average_long_exposure"]

    if "fraction_in_market" not in frame.columns and "trade_frequency" in frame.columns:
        frame["fraction_in_market"] = frame["trade_frequency"]

    metric_order = [
        "model",
        "directional_accuracy",
        "auc_roc",
        "information_ratio",
        "annualized_active_return",
        "active_volatility",
        "tracking_error",
        "active_max_drawdown",
        "active_calmar",
        "correlation_to_benchmark",
        "beta_to_benchmark",
        "alpha_after_beta",
        "sharpe",
        "net_sharpe",
        "benchmark_sharpe",
        "excess_net_sharpe",
        "sharpe_cost_delta",
        "sortino",
        "net_sortino",
        "sortino_cost_delta",
        "calmar",
        "max_drawdown",
        "fraction_in_market",
        "trade_frequency",
        "average_net_exposure",
        "average_long_exposure",
        "average_short_exposure",
        "average_abs_position",
        "average_position_size",
        "fraction_positive_predictions",
        "position_flip_count",
        "daily_turnover",
        "annualized_turnover",
        "average_holding_period_days",
        "round_trip_count",
        "cost_drag",
        "annualized_cost_drag",
        "cost_per_unit_active_return",
        "gross_total_return",
        "net_total_return",
        "total_return",
    ]
    existing = [column for column in metric_order if column in frame.columns]
    extras = [column for column in frame.columns if column not in existing]
    ranking = ranking_columns or get_default_ranking_columns(frame)
    if not ranking:
        return frame[existing + extras].reset_index(drop=True)
    ascending = [False] * len(ranking)
    return frame[existing + extras].sort_values(ranking, ascending=ascending).reset_index(drop=True)
