"""Asset eligibility selection for selective multi-asset portfolios."""

from __future__ import annotations

import pandas as pd


def select_eligible_assets(
    comparison: pd.DataFrame,
    *,
    cost_bps: float,
    champion_model: str,
    min_excess_sharpe: float,
) -> pd.DataFrame:
    """Return one row per asset with eligibility based on excess net Sharpe vs buy-and-hold."""

    subset = comparison.loc[comparison["cost_bps"].eq(float(cost_bps))].copy()
    benchmark = subset[subset["model"].str.startswith("buy_and_hold_")][["ticker", "net_sharpe"]].rename(
        columns={"net_sharpe": "buy_hold_net_sharpe"}
    )
    champion = subset.loc[subset["model"].eq(champion_model), ["ticker", "net_sharpe", "net_sortino", "calmar"]].rename(
        columns={
            "net_sharpe": "champion_net_sharpe",
            "net_sortino": "champion_net_sortino",
            "calmar": "champion_calmar",
        }
    )
    merged = champion.merge(benchmark, on="ticker", how="left")
    merged["excess_vs_bh"] = merged["champion_net_sharpe"] - merged["buy_hold_net_sharpe"]
    merged["eligible"] = merged["excess_vs_bh"] > float(min_excess_sharpe)
    merged["cost_bps"] = float(cost_bps)
    return merged.sort_values("excess_vs_bh", ascending=False).reset_index(drop=True)
