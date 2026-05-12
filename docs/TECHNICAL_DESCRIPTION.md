# Cross-Asset Ranking System: Technical Description

## Overview

The cross-asset ranking system is a quantitative investment framework that predicts which of 18 liquid ETFs (and BTC) will outperform over a forward horizon, using a learning-to-rank algorithm. It is designed for personal-investor deployment with monthly or weekly rebalancing, holding 1-2 positions at a time.

The system was developed through a multi-experiment campaign that empirically falsified the original hypothesis of building a "decision-grade cross-sectional ranker" for this universe at short horizons, while producing usable top-selection artifacts. The empirical finding is that this asset class at 5-20 day horizons supports a small but real edge over equal-weight allocation (~16% improvement in Sharpe ratio), not a strong universe-wide ranking signal.

## Problem Formulation

At each rebalance date *t*, given features for each of N=18 assets in the universe, the model produces a score *s_{i,t}* for asset *i*. Assets are ranked by score and the top-*k* are held until *t + h* (next rebalance), where *h* is the forward horizon (5 or 20 trading days).

The ranking target is the forward risk-adjusted return: *(asset return from t to t+h) / (asset realized volatility over a 20-day window)*. Risk-adjustment ensures the ranker isn't biased toward high-volatility assets with larger raw return magnitudes.

The ranking quality metric is per-date Spearman rank correlation between the model's score-based ranking and the realized return-based ranking. Across many dates, the mean Spearman is the system's Information Coefficient (IC).

This is a classical *learning-to-rank* problem applied to cross-sectional asset selection — the same problem class as web search result ranking, recommender systems, and ad ranking.

## Core Algorithm: LambdaRank on Gradient-Boosted Decision Trees

The model is a LambdaRank implementation running on a Gradient-Boosted Decision Tree (GBDT) backbone, specifically LightGBM.

### Gradient-Boosted Decision Trees

GBDT is an ensemble of shallow decision trees built sequentially. Each tree partitions the feature space via axis-aligned splits (e.g., "is `xs_rank_vol_20d` > 0.65?"); each leaf produces a numerical contribution. The model's prediction is the sum of all tree contributions.

Trees are added one at a time: tree *t+1* is trained to predict the residual error of trees 1...*t*. This "boosting" structure means each tree corrects what previous trees got wrong. After ~100-300 trees (LightGBM defaults), the model represents a complex non-linear function as a sum of simple piecewise-constant pieces.

GBDT dominates small-to-medium tabular data because:

- **Strong inductive bias for tabular features.** Axis-aligned splits naturally fit the structure financial features have.
- **Scale-invariant.** Only feature ordering matters for splits, so feature scaling is unnecessary.
- **Built-in regularization.** Each tree corrects only a fraction of error (via learning rate); sequential structure prevents aggressive memorization.
- **Sample-efficient.** Effective parameter count is small (~few hundred), so the model works well with thousands of samples — unlike neural networks which need 100K+ samples per parameter.

LightGBM specifically is the production-grade implementation: histogram-based tree learning, leaf-wise growth, and efficient grouped-data handling for ranking objectives.

### LambdaRank Objective

LambdaRank is the loss function the GBDT minimizes. Instead of regression loss, LambdaRank optimizes the *ranking* of items within a group.

For each (date, asset) pair, the model produces a score. On each training date, the model is presented with all 18 assets in the universe. The loss considers every pair of assets *(i, j)* on that date and penalizes the model for ranking them in the wrong order relative to actual forward returns.

The penalty per misranked pair is weighted by **NDCG@k** (Normalized Discounted Cumulative Gain at rank k), a metric that emphasizes correctness at the top of the ranking. Errors that move the actual #1 asset to position #5 hurt the loss much more than errors swapping positions #15 and #16. This is exactly the right loss for a top-k allocation strategy: we care immensely about which asset is best, less about which is mediocre.

LambdaRank produces "lambda gradients" — pseudo-gradients computed from pairwise penalties weighted by their NDCG impact. These lambdas are fed to the GBDT as the target for the next tree. The model is trained directly on ranking quality rather than value prediction.

### Per-Date Grouping (Critical Architectural Feature)

Training data is organized as (date, asset) rows, grouped by date. The ranking loss is computed *within* each date's group of 18 assets — never across dates.

This grouping is what makes LambdaRank fundamentally different from regression. An ordinary regression model trained on the same data would have to find features that consistently relate to returns across all assets and all dates. This fails because:

- An asset's features at one date aren't directly comparable to a different asset's features at a different date (different scale, regime, absolute levels)
- Per-asset z-score normalization makes features comparable within each asset's history, not across assets
- A pooled regression model implicitly averages over these incomparabilities, often producing nonsense correlations

LambdaRank with per-date grouping sidesteps this: it learns "given these 18 assets on this date, which should rank higher than which?" The model never commits to absolute relationships; only relative ones within each date.

The Ridge baseline experiment confirmed this empirically: pooled Ridge regression on identical features produced *negative* Spearman (-0.030 at 5d, -0.055 at 20d) — actively anti-predictive — while LambdaRank produced modest positive signal (+0.028). Ridge tried to learn cross-date relationships that don't generalize; LambdaRank only learned within-date relationships and was protected from the failure mode.

### Why This Algorithm for This Problem

Cross-sectional asset ranking is a small-data tabular ranking problem. ~13,600 (date, asset) rows per fold over 5 walk-forward folds, ~33 features per row. The right model class is:

- Tree-based (handles tabular well, scale-invariant, sample-efficient)
- Ranking-objective (optimizes what we care about, not a proxy)
- Group-aware (respects within-date structure)

LambdaRank + LightGBM uniquely satisfies all three. XGBoost with `rank:ndcg` is equivalent; CatBoost with `PairLogitPairwise` is close. Neural rankers (TabNet, transformer-based) underperform at this data scale due to 100-1000× more parameters than useful training patterns. Classical methods (linear regression, factor models) fail when features aren't cross-sectionally comparable, as the Ridge baseline demonstrated.

## Feature Engineering

The feature set comprises ~33 features per (date, asset) row, in two functional categories.

### Per-Asset Features (~28 features)

Computed individually for each asset's time series and z-scored within that asset's training history (`per_asset_train_zscore` normalization).

Categories:
- **Trailing returns** at multiple windows: 5d, 20d, 60d
- **Realized volatility** at multiple windows: 5d, 20d, 60d
- **Drawdown depth and duration** from rolling maximums
- **Volume features** (where available)
- **VIX-derived features**: `vix_relative`, `vix_extreme`, `vix_momentum_5d`
- **Price acceleration and momentum features**: second-derivative-style signals

Z-scoring within each asset's history makes these features comparable to themselves over time (a +2σ value means "in the top decile of this asset's history"). It does *not* make them comparable across assets on the same date (BTC's +2σ vol is in a completely different absolute regime than SHY's +2σ vol).

### Cross-Sectional Features (5 features, the v1 innovation)

Computed per-date across the full 18-asset universe, normalized to [0, 1]:

- `xs_rank_ret_5d`: cross-sectional rank of 5-day trailing return
- `xs_rank_ret_20d`: cross-sectional rank of 20-day trailing return
- `xs_rank_ret_60d`: cross-sectional rank of 60-day trailing return
- `xs_rank_vol_20d`: cross-sectional rank of 20-day realized vol
- `xs_rank_drawdown_60d`: cross-sectional rank of current drawdown from 60-day peak

These features encode *relative position within today's universe* directly. They are inherently cross-sectionally comparable: an `xs_rank_vol_20d` value of 0.95 means "this asset has the highest vol of the 18 assets today," regardless of asset class or absolute level.

### Why Both Feature Types Are Needed

Per-asset features capture within-asset dynamics. Cross-sectional features capture between-asset comparison that per-asset normalization erases.

Empirical importance ranking (averaged across LambdaRank training folds):
- `xs_rank_vol_20d` as the #1 feature (~9% of total gain at 5d horizon, ~19% at 20d)
- Multiple cross-sectional features in the top half of importance
- Per-asset features filling the long tail

The Ridge baseline finding sharpens this: pooled Ridge on the same 33 features produces actively negative Spearman because it tries to learn cross-asset relationships on per-asset z-scored features. LambdaRank's per-date grouping prevents the failure mode entirely.

**Architectural implication**: any model that doesn't respect per-date grouping will fail on this problem, regardless of feature engineering quality.

## Walk-Forward Cross-Validation

The campaign used walk-forward CV with 5 expanding-window splits:

- Training window: 756 trading days (~3 years)
- Validation window: 252 trading days (~1 year, used for early stopping)
- Test window: 252 trading days (~1 year)
- Step size: 252 trading days between folds

5 sequential test periods of ~1 year each, with strict temporal separation. Walk-forward CV is the standard for time-series ML; k-fold CV would let future information leak into training.

In live production, the script uses 1008-day training (train + val combined). The val window is unnecessary for inference (no early-stopping decision) and combining them lets the model see the most recent data.

## Evaluation Metrics

### Information Coefficient (IC)

Per-date Spearman rank correlation between predicted ranks and realized forward-return ranks, aggregated as mean across test dates.

Reasonable benchmarks: IC > 0.05 is the floor for a meaningful cross-sectional signal in published quant research; IC > 0.10 is strong. The empirical ceiling for this 18-asset universe at 5-20d horizons is ~0.032, below the meaningful threshold — the campaign's central finding.

### ICIR

`ICIR = mean(per-date IC across folds) / std(per-date IC across folds)`. Stability of the IC signal across time.

- ICIR > 0.5 strong
- ICIR > 0.3 acceptable
- ICIR < 0.3 indicates the mean is dominated by individual lucky folds

The campaign's best ICIR was v1's 0.94 — a stable signal with a low mean. The architectural pivot's ICIR fell to 0.42, indicating high fold-to-fold variability.

### Information Ratio (IR)

Portfolio excess return over benchmark divided by tracking error. Computed for top-k allocation versus equal-weight benchmark. The campaign's economic high-water mark was v3 5d top-2 at IR = 0.748.

### Drop-Best-Fold Mean IR

Mean IR after excluding the best-performing fold. A stress test for whether the signal is robust or carried by one lucky period. v3 top-2's drop-best of +0.240 confirmed robustness; 20d top-1's +0.120 is the first positive top-1 drop-best in the campaign.

### Random-Null Testing

500 random allocation portfolios simulated per top-k policy per fold. The actual portfolio's IR is compared to this null distribution; p < 0.05 indicates the actual return is unlikely under random allocation. The campaign treated 2 of 5 folds passing as the directional threshold.

## Production Deployment

The live prediction script reuses the campaign's pipeline functions but configures for inference:

- **Training window**: 1008 days (train + val combined; no validation set needed)
- **Causal-only operations**: all rolling windows, z-scores, ranks use strictly past data
- **No look-ahead**: features at date *t* use only data available at or before *t*
- **Per-prediction record**: each prediction produces a JSON with model's pick, scores for all assets, feature values, top features by importance, and cache freshness metadata

The operational wrapper (`run_operational_prediction.py`) refreshes the cache for all 18 assets + VIX before each prediction.

## Universe Characteristics

The 18-asset universe spans major asset classes:
- US equity indices: SPY, QQQ, IWM, DIA
- International equity: EFA, EEM
- Treasury bonds: TLT, IEF, SHY
- Credit bonds: LQD, HYG
- Precious metals: GLD, SLV
- Commodities: USO, DBA
- Currency: UUP
- Real estate: VNQ
- Crypto: BTC-USD

Plus VIX as a regime indicator (used for derived features, not traded).

Empirical characteristics from Phase 1 diagnostics:
- Mean cross-sectional dispersion of 5-day forward returns: 2.49% per-date std
- Mean cross-sectional dispersion of 20-day forward returns: 5.30% per-date std
- Mean pairwise correlation of 5-day returns: 0.19 (not single-factor dominated)

## The Empirical Universe Ceiling

The campaign's most important finding was empirical, not algorithmic: this universe at 5-20 day horizons has a structural Spearman ceiling of approximately 0.03.

**Phase 1B trivial baselines.** A pure momentum predictor — using only the cross-sectional rank of trailing 5d or 20d returns — produces:
- 5d horizon: per-date Spearman of -0.009 (anti-predictive; momentum reverses at this horizon)
- 20d horizon: per-date Spearman of +0.010 (mildly positive; classical Jegadeesh-Titman regime)

These are the floors. Any model has to add something above these baselines to be useful.

**Campaign-wide Spearman across 8+ configurations**: all landed in [0.006, 0.032]. Model family, universe size, target horizon, normalization, feature set, and architectural variants (regime-conditioned interactions, per-regime LambdaRank) all produced Spearman within this band. Variation between configurations was smaller than variation across folds within any single configuration.

**Conclusion**: the ceiling is structural to the universe at these horizons, not a near-miss waiting for one more clever feature.

**Implications**:
- The original campaign goal of "decision-grade cross-sectional ranker" is not achievable on this universe at these horizons
- A "top-selection strategy" (pick 1-2 best assets per rebalance) is achievable and produces a modest but real edge over equal-weight (Sharpe 1.18 vs 1.02 in backtest)
- Future work targeting stronger signal requires a different universe (more assets, more dispersion, less single-factor exposure) or longer horizon (60+ days, where cross-sectional momentum effects are more reliable)

## Implementation Stack

- **Python 3.x**
- **LightGBM** (LambdaRank objective with NDCG@k)
- **scikit-learn** (Ridge baseline diagnostic only)
- **pandas, numpy** for data manipulation
- **yfinance** for price data acquisition
- **Internal pipeline modules**: `experiments/cross_asset_ranking_experiment.py`, `evaluation/cross_asset_ranking.py`

Walk-forward CV, feature engineering, scoring, and allocation logic are pure Python with vectorized pandas/numpy. Full campaign runs in minutes per experiment on a modern laptop.

## References to Other Documentation

- `OPERATIONAL_RUNBOOK.md` — day-to-day procedures
- `TROUBLESHOOTING.md` — common issues and fixes
- `GETTING_STARTED.md` — orientation for new users
- `PATCH_CROSS_ASSET_RANKING_*.md` — campaign audit trail
- `PATCH_LIVE_PREDICTION_SCRIPT.md`, `PATCH_OPERATIONAL_WRAPPER.md` — deployment infrastructure documentation
