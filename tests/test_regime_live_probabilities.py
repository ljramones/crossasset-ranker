"""Tests for live-safe regime probability inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.regime_detection import MarketRegimeDetector, RegimeDetectionConfig


class _IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class _FakeGMMModel:
    def predict(self, scaled):
        scaled = np.asarray(scaled, dtype=float)
        return (scaled[:, 0] > 0.0).astype(int)

    def predict_proba(self, scaled):
        scaled = np.asarray(scaled, dtype=float)
        probabilities = np.zeros((len(scaled), 3), dtype=float)
        positive = scaled[:, 0] > 0.0
        probabilities[:, 0] = (~positive).astype(float)
        probabilities[:, 1] = positive.astype(float)
        return probabilities


class _FakeHMMModel:
    def predict(self, scaled):
        scaled = np.asarray(scaled, dtype=float)
        return np.full(len(scaled), fill_value=min(len(scaled) - 1, 2), dtype=int)

    def predict_proba(self, scaled):
        scaled = np.asarray(scaled, dtype=float)
        # Depend explicitly on prefix length so earlier-row full-slice
        # probabilities differ from expanding-prefix probabilities.
        prefix_len = len(scaled)
        probabilities = np.zeros((prefix_len, 3), dtype=float)
        last_probability = min(0.1 * prefix_len, 0.9)
        probabilities[:, 0] = 1.0 - last_probability
        probabilities[:, 2] = last_probability
        return probabilities


def _build_detector(*, backend: str, model) -> MarketRegimeDetector:
    detector = MarketRegimeDetector(
        RegimeDetectionConfig(
            model_type=backend,
            n_regimes=3,
            inference_columns=("a", "b", "c"),
        )
    )
    detector.backend = backend
    detector.model = model
    detector.scaler = _IdentityScaler()
    detector.feature_columns = ["a", "b", "c"]
    detector.regime_mapping_ = {0: 0, 1: 1, 2: 2}
    return detector


def test_predict_live_safe_matches_predict_for_gmm_backend() -> None:
    frame = pd.DataFrame(
        {
            "a": [-1.0, 1.0, -2.0],
            "b": [0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    detector = _build_detector(backend="gmm", model=_FakeGMMModel())

    standard = detector.predict(frame)
    live_safe = detector.predict_live_safe(frame)

    pd.testing.assert_series_equal(standard.labels, live_safe.labels)
    pd.testing.assert_frame_equal(standard.probabilities, live_safe.probabilities)


def test_predict_live_safe_uses_expanding_prefix_for_hmm_backend() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [0.0, 0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )
    detector = _build_detector(backend="hmm", model=_FakeHMMModel())

    standard = detector.predict(frame)
    live_safe = detector.predict_live_safe(frame)

    # Full-slice posterior uses the same prefix length for all rows, whereas
    # live-safe inference should evolve as the prefix grows.
    assert standard.probabilities["regime_prob_2"].tolist() == [0.4, 0.4, 0.4, 0.4]
    assert live_safe.probabilities["regime_prob_2"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert live_safe.labels.tolist() == [0, 1, 2, 2]


def test_predict_live_safe_preserves_output_shape_and_columns() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 2.0],
            "b": [0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    detector = _build_detector(backend="hmm", model=_FakeHMMModel())

    prediction = detector.predict_live_safe(frame)

    assert list(prediction.labels.index) == list(frame.index)
    assert list(prediction.probabilities.index) == list(frame.index)
    assert list(prediction.probabilities.columns) == ["regime_prob_0", "regime_prob_1", "regime_prob_2"]


def test_predict_live_safe_respects_non_identity_canonical_mapping() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    detector = _build_detector(backend="hmm", model=_FakeHMMModel())
    detector.regime_mapping_ = {0: 2, 1: 0, 2: 1}

    prediction = detector.predict_live_safe(frame)

    # Raw label path is [0, 1, 2]; canonical mapping should remap to [2, 0, 1].
    assert prediction.labels.tolist() == [2, 0, 1]
    # Raw probability mass accumulates in column 2; remapped canonical column 1 should receive it.
    assert prediction.probabilities["regime_prob_1"].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_predict_live_safe_missing_rows_do_not_backfill_from_future_rows() -> None:
    frame = pd.DataFrame(
        {
            "a": [np.nan, 1.0, 2.0],
            "b": [0.0, 0.0, 0.0],
            "c": [0.0, 0.0, 0.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    detector = _build_detector(backend="hmm", model=_FakeHMMModel())

    standard = detector.predict(frame)
    live_safe = detector.predict_live_safe(frame)

    # Existing vectorized path preserves prior behavior and may backfill the
    # leading invalid row from the next valid observation.
    assert standard.labels.iloc[0] == 1
    assert standard.probabilities.iloc[0].tolist() == pytest.approx([0.8, 0.0, 0.2])

    # Live-safe path must not use future rows to fill earlier invalid rows.
    assert live_safe.labels.iloc[0] == 0
    assert live_safe.probabilities.iloc[0].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    # Valid rows remain canonically remapped and prefix-safe.
    assert live_safe.probabilities.iloc[1]["regime_prob_2"] == pytest.approx(0.1)
    assert live_safe.probabilities.iloc[2]["regime_prob_2"] == pytest.approx(0.2)
