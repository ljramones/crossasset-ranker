"""Walk-forward-safe market regime detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import compute_trading_metrics


@dataclass(slots=True)
class RegimeDetectionConfig:
    """Configuration for a regime detector."""

    model_type: str = "hmm"
    n_regimes: int = 3
    inference_columns: tuple[str, ...] = (
        "return_5d",
        "return_20d",
        "momentum_norm",
        "vol_ratio",
        "realized_vol_20",
        "downside_vol_ratio",
        "vix_zscore",
        "implied_vs_realized_vol",
        "vix_momentum_5d",
    )


@dataclass(slots=True)
class RegimePrediction:
    """Predicted regime states and probabilities for one frame."""

    labels: pd.Series
    probabilities: pd.DataFrame


@dataclass(slots=True)
class RegimeFilterDecision:
    """Train-window decision used for aggressive regime filtering."""

    best_regime_id: int
    regime_scores: dict[int, float]


class MarketRegimeDetector:
    """Detect three market regimes from stationary features without leakage."""

    def __init__(self, config: RegimeDetectionConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model: Any | None = None
        self.backend = "gmm"
        self.feature_columns: list[str] = []
        self.regime_mapping_: dict[int, int] = {}
        self.transition_matrix_: np.ndarray | None = None

    def fit(self, train_frame: pd.DataFrame) -> None:
        """Fit the detector using training-window data only."""

        self.feature_columns = [column for column in self.config.inference_columns if column in train_frame.columns]
        if len(self.feature_columns) < 3:
            raise ValueError("Regime detection requires at least three available inference features.")

        training_features = train_frame[self.feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        if training_features.empty:
            raise ValueError("Regime detector received no finite training rows.")

        scaled = self.scaler.fit_transform(training_features)
        self.model, self.backend = self._build_model()
        self.model.fit(scaled)

        raw_labels, raw_probabilities = self._predict_raw(training_features)
        self.regime_mapping_ = self._build_canonical_mapping(
            train_frame=train_frame.loc[training_features.index],
            raw_labels=raw_labels,
        )
        remapped = self._remap_labels(raw_labels)
        self.transition_matrix_ = self._compute_transition_matrix(remapped, self.config.n_regimes)

    def predict(self, frame: pd.DataFrame) -> RegimePrediction:
        """Infer regimes and regime probabilities for a new time window.

        This method preserves the existing vectorized behavior. For HMM-backed
        detectors it may use full-slice posterior probabilities. Use
        ``predict_live_safe(...)`` when validation/test realism requires
        expanding-prefix inference that does not look ahead within the slice.
        """

        return self._predict_frame(frame, live_safe=False)

    def predict_live_safe(self, frame: pd.DataFrame) -> RegimePrediction:
        """Infer regimes with an expanding-prefix probability path.

        For GMM backends this is equivalent to ``predict(...)`` because the
        probabilities are observation-local. For HMM backends, each row is
        inferred from observations available through that row only.
        """

        return self._predict_frame(frame, live_safe=True)

    def _predict_frame(self, frame: pd.DataFrame, *, live_safe: bool) -> RegimePrediction:
        """Shared prediction path with optional live-safe prefix inference."""

        if self.model is None:
            raise RuntimeError("Regime detector must be fitted before prediction.")

        features = frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        valid_mask = features.notna().all(axis=1)
        scaled = self.scaler.transform(features.loc[valid_mask])
        if live_safe:
            raw_labels, raw_probabilities = self._predict_raw_live_safe(
                features.loc[valid_mask],
                scaled_override=scaled,
            )
        else:
            raw_labels, raw_probabilities = self._predict_raw(features.loc[valid_mask], scaled_override=scaled)
        labels = pd.Series(np.nan, index=frame.index, name="regime_id")
        probabilities = pd.DataFrame(
            np.nan,
            index=frame.index,
            columns=[f"regime_prob_{index}" for index in range(self.config.n_regimes)],
        )

        remapped_labels = self._remap_labels(raw_labels)
        remapped_probabilities = self._remap_probabilities(raw_probabilities)
        labels.loc[valid_mask] = remapped_labels
        probabilities.loc[valid_mask, :] = remapped_probabilities

        if live_safe:
            labels = labels.ffill().fillna(0).astype(int)
            probabilities = probabilities.ffill().fillna(1.0 / self.config.n_regimes)
        else:
            labels = labels.ffill().bfill().fillna(0).astype(int)
            probabilities = probabilities.ffill().bfill().fillna(1.0 / self.config.n_regimes)
        return RegimePrediction(labels=labels, probabilities=probabilities)

    def identify_best_regime(
        self,
        train_frame: pd.DataFrame,
        annualization_factor: int,
    ) -> RegimeFilterDecision:
        """Select the single strongest train-window regime by regime-conditional net Sharpe."""

        if "regime_id" not in train_frame.columns:
            raise ValueError("Training frame must include regime_id before best-regime selection.")

        regime_scores: dict[int, float] = {}
        for regime_id in range(self.config.n_regimes):
            regime_slice = train_frame.loc[train_frame["regime_id"] == regime_id]
            if regime_slice.empty:
                regime_scores[regime_id] = float("-inf")
                continue
            metrics = compute_trading_metrics(
                returns=regime_slice["forward_simple_return_1d"],
                benchmark_returns=regime_slice["benchmark_return_1d"],
                signal=pd.Series(1, index=regime_slice.index, name="regime_signal"),
                annualization_factor=annualization_factor,
                transaction_cost_bps=0.0,
            )
            regime_scores[regime_id] = float(metrics["net_sharpe"])
        best_regime_id = max(regime_scores, key=regime_scores.get)
        return RegimeFilterDecision(best_regime_id=int(best_regime_id), regime_scores=regime_scores)

    def _build_model(self) -> tuple[Any, str]:
        """Create the preferred detector backend."""

        if self.config.model_type == "hmm":
            try:
                from hmmlearn.hmm import GaussianHMM
            except ImportError:
                pass
            else:
                return (
                    GaussianHMM(
                        n_components=self.config.n_regimes,
                        covariance_type="full",
                        n_iter=200,
                        random_state=42,
                    ),
                    "hmm",
                )

        return (
            GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type="full",
                random_state=42,
                max_iter=300,
            ),
            "gmm",
        )

    def _predict_raw(
        self,
        frame: pd.DataFrame,
        scaled_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer raw backend labels and probabilities."""

        scaled = scaled_override if scaled_override is not None else self.scaler.transform(frame[self.feature_columns])
        if self.backend == "hmm":
            labels = self.model.predict(scaled)
            probabilities = self.model.predict_proba(scaled)
            return labels, probabilities

        labels = self.model.predict(scaled)
        probabilities = self.model.predict_proba(scaled)
        return labels, probabilities

    def _predict_raw_live_safe(
        self,
        frame: pd.DataFrame,
        scaled_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer raw labels/probabilities without using future rows in the slice.

        GMM-style detectors are already observation-local, so the standard path
        is safe. HMM-style detectors are evaluated on expanding prefixes and
        only the last-row label/probability from each prefix is retained.
        """

        scaled = scaled_override if scaled_override is not None else self.scaler.transform(frame[self.feature_columns])
        if self.backend != "hmm":
            return self._predict_raw(frame, scaled_override=scaled)

        labels = np.zeros(len(scaled), dtype=int)
        probabilities = np.zeros((len(scaled), self.config.n_regimes), dtype=float)
        for row_index in range(len(scaled)):
            prefix = scaled[: row_index + 1]
            prefix_labels = self.model.predict(prefix)
            prefix_probabilities = self.model.predict_proba(prefix)
            labels[row_index] = int(prefix_labels[-1])
            probabilities[row_index, :] = prefix_probabilities[-1]
        return labels, probabilities

    def _build_canonical_mapping(self, train_frame: pd.DataFrame, raw_labels: np.ndarray) -> dict[int, int]:
        """Map raw states into stable bull/bear/high-volatility ids."""

        summary = (
            pd.DataFrame(
                {
                    "raw_label": raw_labels,
                    "return_1d": train_frame["return_1d"].values,
                    "realized_vol_20": train_frame["realized_vol_20"].values,
                }
            )
            .groupby("raw_label", as_index=True)
            .agg({"return_1d": "mean", "realized_vol_20": "mean"})
        )

        bull_label = int(summary["return_1d"].idxmax())
        bear_label = int(summary["return_1d"].idxmin())
        remaining = [label for label in summary.index.tolist() if label not in {bull_label, bear_label}]
        high_vol_label = int(remaining[0]) if remaining else bear_label
        return {
            bull_label: 0,
            bear_label: 1,
            high_vol_label: 2,
        }

    def _remap_labels(self, raw_labels: np.ndarray) -> np.ndarray:
        """Apply the canonical label mapping."""

        return np.array([self.regime_mapping_.get(int(label), int(label)) for label in raw_labels], dtype=int)

    def _remap_probabilities(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Reorder probability columns to the canonical label order."""

        reordered = np.zeros_like(raw_probabilities)
        for raw_label, canonical_label in self.regime_mapping_.items():
            reordered[:, canonical_label] = raw_probabilities[:, raw_label]
        return reordered

    @staticmethod
    def _compute_transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
        """Estimate transition probabilities from an ordered label stream."""

        counts = np.zeros((n_regimes, n_regimes), dtype=float)
        for left, right in zip(labels[:-1], labels[1:]):
            counts[int(left), int(right)] += 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        probabilities = np.zeros_like(counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(counts, row_sums, out=probabilities, where=row_sums > 0.0)
        probabilities[row_sums.squeeze() == 0.0] = 0.0
        return probabilities


def add_aggressive_trade_filter_columns(
    frame: pd.DataFrame,
    *,
    best_regime_id: int,
    min_regime_prob: float,
) -> None:
    """Annotate a frame with train-derived aggressive regime filter metadata."""

    probability_column = f"regime_prob_{best_regime_id}"
    if probability_column not in frame.columns:
        raise ValueError(f"Missing required regime probability column: {probability_column}")

    frame["best_regime_id"] = int(best_regime_id)
    frame["best_regime_prob"] = frame[probability_column].astype(float)
    frame["trade_allowed_aggressive"] = (
        frame["regime_id"].astype(int).eq(int(best_regime_id))
        & frame["best_regime_prob"].ge(float(min_regime_prob))
    ).astype(int)
