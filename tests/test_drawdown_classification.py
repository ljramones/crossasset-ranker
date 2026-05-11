from __future__ import annotations

import pandas as pd

from evaluation.drawdown_classification import (
    AlwaysNegativeClassifier,
    ConstantProbabilityClassifier,
    HistGradientBoostingDrawdownClassifier,
    LogisticDrawdownClassifier,
    RegularizedLinearDrawdownClassifier,
    RollingEventRateClassifier,
    build_simple_drawdown_classifier,
    compute_drawdown_classification_metrics,
)


def test_compute_drawdown_classification_metrics_returns_expected_keys() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    y_pred = pd.Series([0, 1, 0, 1, 1, 0])

    metrics = compute_drawdown_classification_metrics(y_true, y_score, y_pred)

    assert {
        "directional_accuracy",
        "auc_roc",
        "balanced_accuracy",
        "precision",
        "recall",
        "brier_score",
        "positive_prediction_rate",
        "base_event_rate",
    } == set(metrics)


def test_constant_probability_classifier_learns_training_event_rate() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([0, 1, 1, 0])
    model = ConstantProbabilityClassifier()

    model.fit(X, y)
    probability = model.predict_proba(X)
    prediction = model.predict(X)

    assert probability.tolist() == [0.5, 0.5, 0.5, 0.5]
    assert prediction.tolist() == [1, 1, 1, 1]


def test_rolling_event_rate_classifier_uses_recent_training_window() -> None:
    X = pd.DataFrame({"x": list(range(6))})
    y = pd.Series([0, 0, 1, 1, 1, 0])
    model = RollingEventRateClassifier(window=3)

    model.fit(X, y)
    probability = model.predict_proba(pd.DataFrame({"x": [10, 11]}))

    assert probability.tolist() == [2 / 3, 2 / 3]


def test_always_negative_classifier_outputs_zero_probabilities() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    model = AlwaysNegativeClassifier().fit(X, pd.Series([0, 1, 0]))

    assert model.predict_proba(X).tolist() == [0.0, 0.0, 0.0]
    assert model.predict(X).tolist() == [0, 0, 0]


def test_logistic_and_histgb_classifiers_fit_and_predict_on_synthetic_data() -> None:
    X = pd.DataFrame(
        {
            "f1": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
            "f2": [1.0, 0.5, 0.2, 0.2, 0.5, 1.0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    for model in (LogisticDrawdownClassifier(), HistGradientBoostingDrawdownClassifier(max_iter=20)):
        model.fit(X, y)
        probability = model.predict_proba(X)
        prediction = model.predict(X)

        assert len(probability) == len(X)
        assert len(prediction) == len(X)
        assert probability.between(0.0, 1.0).all()


def test_build_simple_drawdown_classifier_returns_expected_types() -> None:
    assert isinstance(build_simple_drawdown_classifier("always_negative"), AlwaysNegativeClassifier)
    assert isinstance(build_simple_drawdown_classifier("event_rate"), ConstantProbabilityClassifier)
    assert isinstance(build_simple_drawdown_classifier("rolling_event_rate"), RollingEventRateClassifier)
    assert isinstance(build_simple_drawdown_classifier("logistic"), LogisticDrawdownClassifier)
    assert isinstance(build_simple_drawdown_classifier("regularized_linear"), RegularizedLinearDrawdownClassifier)
    assert isinstance(build_simple_drawdown_classifier("histgb"), HistGradientBoostingDrawdownClassifier)
