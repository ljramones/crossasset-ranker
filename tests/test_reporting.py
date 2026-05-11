"""Tests for reporting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.reporting import get_primary_ranking_label, summarize_results


def test_summarize_results_prefers_information_ratio_over_net_sharpe() -> None:
    frame = summarize_results(
        [
            {"model": "high_sharpe_low_ir", "net_sharpe": 2.0, "information_ratio": 0.1},
            {"model": "lower_sharpe_high_ir", "net_sharpe": 1.0, "information_ratio": 0.3},
        ]
    )

    assert frame.iloc[0]["model"] == "lower_sharpe_high_ir"


def test_summarize_results_derives_fraction_in_market_and_average_net_exposure() -> None:
    frame = summarize_results(
        [
            {
                "model": "sample",
                "trade_frequency": 0.8,
                "average_long_exposure": 0.8,
                "average_short_exposure": 0.1,
                "information_ratio": 0.2,
            }
        ]
    )

    assert "fraction_in_market" in frame.columns
    assert "average_net_exposure" in frame.columns
    assert float(frame.loc[0, "fraction_in_market"]) == 0.8
    assert float(frame.loc[0, "average_net_exposure"]) == pytest.approx(0.7)


def test_primary_ranking_label_is_active_skill_first() -> None:
    frame = pd.DataFrame([{"model": "sample", "information_ratio": 0.2, "net_sharpe": 1.0}])

    assert get_primary_ranking_label(frame) == "information_ratio"
