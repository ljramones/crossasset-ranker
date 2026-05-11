from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.probability_calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_probability_calibrator,
    clip_probabilities,
    logit_probabilities,
)


def test_clip_probabilities_limits_values_to_open_interval() -> None:
    values = clip_probabilities([0.0, 0.2, 1.0], eps=1e-3)

    assert np.all(values > 0.0)
    assert np.all(values < 1.0)
    assert values[1] == 0.2


def test_logit_probabilities_returns_finite_values() -> None:
    logits = logit_probabilities([0.0, 0.5, 1.0], eps=1e-3)

    assert np.isfinite(logits).all()
    assert logits[1] == 0.0


def test_identity_calibrator_returns_clipped_probabilities() -> None:
    calibrator = IdentityCalibrator().fit([0.2, 0.8], [0, 1])
    result = calibrator.predict_proba(pd.Series([0.2, 0.8], index=[10, 11]))

    assert result.index.tolist() == [10, 11]
    assert result.tolist() == [0.2, 0.8]


def test_platt_calibrator_fits_and_predicts_probabilities() -> None:
    probabilities = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    calibrator = PlattCalibrator().fit(probabilities, y_true)

    result = calibrator.predict_proba(probabilities)

    assert len(result) == len(probabilities)
    assert result.between(0.0, 1.0).all()


def test_isotonic_calibrator_fits_and_predicts_probabilities() -> None:
    probabilities = pd.Series([0.05, 0.15, 0.25, 0.75, 0.85, 0.95])
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    calibrator = IsotonicCalibrator().fit(probabilities, y_true)

    result = calibrator.predict_proba(probabilities)

    assert len(result) == len(probabilities)
    assert result.between(0.0, 1.0).all()


def test_constant_class_case_returns_constant_probability_for_calibrators() -> None:
    probabilities = pd.Series([0.2, 0.4, 0.6])
    y_true = pd.Series([1, 1, 1])

    for calibrator in (PlattCalibrator(), IsotonicCalibrator()):
        calibrator.fit(probabilities, y_true)
        result = calibrator.predict_proba(probabilities)
        assert result.tolist() == [1.0 - 1e-6] * 3


def test_build_probability_calibrator_returns_expected_types() -> None:
    assert isinstance(build_probability_calibrator("identity"), IdentityCalibrator)
    assert isinstance(build_probability_calibrator("platt"), PlattCalibrator)
    assert isinstance(build_probability_calibrator("isotonic"), IsotonicCalibrator)

