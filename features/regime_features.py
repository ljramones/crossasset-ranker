"""Regime-derived feature generation helpers."""

from __future__ import annotations

import pandas as pd


def add_regime_features(frame: pd.DataFrame) -> list[str]:
    """Add regime labels, probabilities, and selected interaction terms."""

    if "regime_id" not in frame.columns:
        raise ValueError("`regime_id` must be present before adding regime features.")

    feature_columns = [
        "regime_id",
        "regime_prob_0",
        "regime_prob_1",
        "regime_prob_2",
    ]
    for column in feature_columns[1:]:
        if column not in frame.columns:
            raise ValueError(f"Missing regime probability column {column!r}.")

    frame["momentum_x_regime_bull"] = frame["momentum_norm"] * frame["regime_prob_0"]
    frame["momentum_x_regime_high_vol"] = frame["momentum_norm"] * frame["regime_prob_2"]
    frame["vol_ratio_x_regime_high_vol"] = frame["vol_ratio"] * frame["regime_prob_2"]

    feature_columns.extend(
        [
            "momentum_x_regime_bull",
            "momentum_x_regime_high_vol",
            "vol_ratio_x_regime_high_vol",
        ]
    )
    return feature_columns
