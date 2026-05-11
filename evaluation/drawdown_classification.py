"""Standalone helpers for drawdown-risk classification baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_drawdown_classification_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Return classifier diagnostics for drawdown-event prediction."""

    true = pd.Series(y_true).astype(int)
    score = pd.Series(y_score, index=true.index).astype(float)
    pred = pd.Series(y_pred, index=true.index).astype(int)
    true_unique = int(true.nunique())

    if true_unique < 2:
        auc_roc = 0.5
    else:
        auc_roc = float(roc_auc_score(true, score))

    try:
        brier = float(brier_score_loss(true, score))
    except ValueError:
        brier = float("nan")

    directional_accuracy = float(accuracy_score(true, pred))
    if true_unique < 2:
        balanced_accuracy = directional_accuracy
    else:
        balanced_accuracy = float(balanced_accuracy_score(true, pred))

    return {
        "directional_accuracy": directional_accuracy,
        "auc_roc": auc_roc,
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "brier_score": brier,
        "positive_prediction_rate": float(pred.mean()),
        "base_event_rate": float(true.mean()),
    }


class DrawdownClassifier(Protocol):
    """Minimal protocol for standalone drawdown-risk classifiers."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DrawdownClassifier": ...

    def predict_proba(self, X: pd.DataFrame) -> pd.Series: ...

    def predict(self, X: pd.DataFrame) -> pd.Series: ...


@dataclass(slots=True)
class ConstantProbabilityClassifier:
    """Predict a constant drawdown-event probability estimated from training labels."""

    threshold: float = 0.5
    constant_probability_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConstantProbabilityClassifier":
        del X
        labels = pd.Series(y).astype(float)
        self.constant_probability_ = float(labels.mean()) if len(labels) else 0.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.constant_probability_ is None:
            raise ValueError("Classifier must be fit before prediction.")
        return pd.Series(self.constant_probability_, index=X.index, name="prediction_probability", dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probability = self.predict_proba(X)
        return (probability >= self.threshold).astype(int).rename("prediction")


@dataclass(slots=True)
class RollingEventRateClassifier:
    """Predict a rolling historical event rate using only past training labels."""

    threshold: float = 0.5
    window: int = 63
    rolling_probability_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RollingEventRateClassifier":
        del X
        labels = pd.Series(y).astype(float)
        if len(labels) == 0:
            self.rolling_probability_ = 0.0
        else:
            window = max(1, min(int(self.window), len(labels)))
            self.rolling_probability_ = float(labels.iloc[-window:].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.rolling_probability_ is None:
            raise ValueError("Classifier must be fit before prediction.")
        return pd.Series(self.rolling_probability_, index=X.index, name="prediction_probability", dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probability = self.predict_proba(X)
        return (probability >= self.threshold).astype(int).rename("prediction")


@dataclass(slots=True)
class AlwaysNegativeClassifier:
    """Trivial classifier that always predicts no future drawdown event."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlwaysNegativeClassifier":
        del X, y
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index, name="prediction_probability", dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=X.index, name="prediction", dtype=int)


@dataclass(slots=True)
class LogisticDrawdownClassifier:
    """Small sklearn logistic baseline for drawdown-risk classification."""

    max_iter: int = 1000
    class_weight: str | dict[str, float] | None = "balanced"
    random_state: int = 42
    threshold: float = 0.5
    C: float = 1.0
    model_: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticDrawdownClassifier":
        self.model_ = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            C=self.C,
        )
        self.model_.fit(X, pd.Series(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ValueError("Classifier must be fit before prediction.")
        probability = self.model_.predict_proba(X)[:, 1]
        return pd.Series(probability, index=X.index, name="prediction_probability", dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probability = self.predict_proba(X)
        return (probability >= self.threshold).astype(int).rename("prediction")


@dataclass(slots=True)
class HistGradientBoostingDrawdownClassifier:
    """Small sklearn HistGradientBoosting baseline for drawdown-risk classification."""

    max_depth: int = 3
    max_iter: int = 100
    learning_rate: float = 0.05
    random_state: int = 42
    threshold: float = 0.5
    model_: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistGradientBoostingDrawdownClassifier":
        self.model_ = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.model_.fit(X, pd.Series(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise ValueError("Classifier must be fit before prediction.")
        probability = self.model_.predict_proba(X)[:, 1]
        return pd.Series(probability, index=X.index, name="prediction_probability", dtype=float)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probability = self.predict_proba(X)
        return (probability >= self.threshold).astype(int).rename("prediction")


@dataclass(slots=True)
class RegularizedLinearDrawdownClassifier(LogisticDrawdownClassifier):
    """Thin alias for a more strongly regularized linear classifier baseline."""

    C: float = 0.25


def build_simple_drawdown_classifier(model_name: str) -> DrawdownClassifier:
    """Return one of the supported standalone simple classifier baselines."""

    normalized = str(model_name).lower()
    if normalized in {"always_negative", "always-negative"}:
        return AlwaysNegativeClassifier()
    if normalized in {"event_rate", "historical_event_rate"}:
        return ConstantProbabilityClassifier()
    if normalized in {"rolling_event_rate", "rolling_historical_event_rate"}:
        return RollingEventRateClassifier()
    if normalized in {"logistic", "logistic_regression"}:
        return LogisticDrawdownClassifier()
    if normalized in {"regularized_linear", "regularized_linear_classifier", "linear_regularized"}:
        return RegularizedLinearDrawdownClassifier()
    if normalized in {"hist_gradient_boosting", "histgb", "hist_gradient_boosting_classifier"}:
        return HistGradientBoostingDrawdownClassifier()
    raise KeyError(f"Unsupported drawdown classifier baseline {model_name!r}.")
