"""Standalone probability calibration helpers for classifier-only experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def clip_probabilities(probabilities, *, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities into an open interval for stable calibration math."""

    if eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must lie in (0, 0.5).")
    array = np.asarray(probabilities, dtype=float)
    return np.clip(array, eps, 1.0 - eps)


def logit_probabilities(probabilities, *, eps: float = 1e-6) -> np.ndarray:
    """Convert probabilities to logits after clipping for numerical stability."""

    clipped = clip_probabilities(probabilities, eps=eps)
    return np.log(clipped / (1.0 - clipped))


class ProbabilityCalibrator(Protocol):
    """Minimal protocol for post-hoc probability calibrators."""

    def fit(self, probabilities, y_true) -> "ProbabilityCalibrator": ...

    def predict_proba(self, probabilities) -> pd.Series: ...


@dataclass(slots=True)
class IdentityCalibrator:
    """Leave probabilities unchanged."""

    eps: float = 1e-6

    def fit(self, probabilities, y_true) -> "IdentityCalibrator":
        del probabilities, y_true
        return self

    def predict_proba(self, probabilities) -> pd.Series:
        values = clip_probabilities(probabilities, eps=self.eps)
        index = getattr(probabilities, "index", None)
        return pd.Series(values, index=index, name="calibrated_prediction_probability", dtype=float)


@dataclass(slots=True)
class PlattCalibrator:
    """Fit logistic calibration on raw model probabilities."""

    eps: float = 1e-6
    random_state: int = 42
    model_: LogisticRegression | None = None
    constant_probability_: float | None = None

    def fit(self, probabilities, y_true) -> "PlattCalibrator":
        probs = clip_probabilities(probabilities, eps=self.eps)
        labels = pd.Series(y_true).astype(int)
        if labels.nunique() < 2:
            self.constant_probability_ = float(labels.mean()) if len(labels) else 0.0
            self.model_ = None
            return self

        logits = logit_probabilities(probs, eps=self.eps).reshape(-1, 1)
        self.model_ = LogisticRegression(random_state=self.random_state)
        self.model_.fit(logits, labels.to_numpy())
        self.constant_probability_ = None
        return self

    def predict_proba(self, probabilities) -> pd.Series:
        index = getattr(probabilities, "index", None)
        probs = clip_probabilities(probabilities, eps=self.eps)
        if self.constant_probability_ is not None:
            values = np.full(len(probs), self.constant_probability_, dtype=float)
        else:
            if self.model_ is None:
                raise ValueError("Calibrator must be fit before prediction.")
            logits = logit_probabilities(probs, eps=self.eps).reshape(-1, 1)
            values = self.model_.predict_proba(logits)[:, 1]
        return pd.Series(clip_probabilities(values, eps=self.eps), index=index, name="calibrated_prediction_probability", dtype=float)


@dataclass(slots=True)
class IsotonicCalibrator:
    """Fit isotonic regression on raw model probabilities."""

    eps: float = 1e-6
    model_: IsotonicRegression | None = None
    constant_probability_: float | None = None

    def fit(self, probabilities, y_true) -> "IsotonicCalibrator":
        probs = clip_probabilities(probabilities, eps=self.eps)
        labels = pd.Series(y_true).astype(int)
        if labels.nunique() < 2:
            self.constant_probability_ = float(labels.mean()) if len(labels) else 0.0
            self.model_ = None
            return self

        self.model_ = IsotonicRegression(y_min=self.eps, y_max=1.0 - self.eps, out_of_bounds="clip")
        self.model_.fit(probs, labels.to_numpy())
        self.constant_probability_ = None
        return self

    def predict_proba(self, probabilities) -> pd.Series:
        index = getattr(probabilities, "index", None)
        probs = clip_probabilities(probabilities, eps=self.eps)
        if self.constant_probability_ is not None:
            values = np.full(len(probs), self.constant_probability_, dtype=float)
        else:
            if self.model_ is None:
                raise ValueError("Calibrator must be fit before prediction.")
            values = self.model_.predict(probs)
        return pd.Series(clip_probabilities(values, eps=self.eps), index=index, name="calibrated_prediction_probability", dtype=float)


def build_probability_calibrator(calibration_method: str) -> ProbabilityCalibrator:
    """Return one of the supported standalone calibration methods."""

    normalized = str(calibration_method).lower()
    if normalized in {"identity", "none", "uncalibrated"}:
        return IdentityCalibrator()
    if normalized in {"platt", "platt_scaling", "sigmoid"}:
        return PlattCalibrator()
    if normalized in {"isotonic", "isotonic_regression"}:
        return IsotonicCalibrator()
    raise KeyError(f"Unsupported calibration method {calibration_method!r}.")

