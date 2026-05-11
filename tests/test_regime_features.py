"""Tests for regime-derived feature helpers."""

from __future__ import annotations

import pandas as pd

from features.regime_features import add_regime_features


def test_add_regime_features_creates_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "regime_id": [0, 1, 2],
            "regime_prob_0": [0.8, 0.1, 0.2],
            "regime_prob_1": [0.1, 0.8, 0.2],
            "regime_prob_2": [0.1, 0.1, 0.6],
            "momentum_norm": [0.2, -0.1, 0.05],
            "vol_ratio": [1.1, 0.9, 1.4],
        }
    )

    feature_columns = add_regime_features(frame)

    expected = {
        "regime_id",
        "regime_prob_0",
        "regime_prob_1",
        "regime_prob_2",
        "momentum_x_regime_bull",
        "momentum_x_regime_high_vol",
        "vol_ratio_x_regime_high_vol",
    }
    assert expected.issubset(set(feature_columns))
    assert expected.issubset(set(frame.columns))
