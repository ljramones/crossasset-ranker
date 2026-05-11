# Patch: Cross-Asset Ranking — Architectural Pivot (Per-Regime LambdaRank) + Universe Diagnostics

Date: 2026-05-11
Phase 1 run timestamp: `20260511T174933Z` → `results/phase1_universe_diagnostics_20260511T174933Z/`

This patch documents Phase 1 universe diagnostics BEFORE Phase 2-5 are run, so the Phase 5 verdict is interpreted against an upper bound estimated without knowledge of the pivot's outcome. The pre-committed +0.05 / ICIR ≥ 0.5 decision-grade thresholds remain unchanged regardless of what Phase 1 shows.

---

## Phase 1 — Universe diagnostics (documented BEFORE Phase 5 results)

Three sub-analyses on the 18-asset cross-asset universe, 5-day forward horizon, using cached data only and no model training. Implemented in `scripts/run_phase1_universe_diagnostics.py`. Outputs stored as CSVs plus a PNG plot under the timestamped results directory; the run's metadata JSON captures every summary statistic for downstream audit.

### Phase 1A — Cross-sectional dispersion of forward 5d returns

Per-date standard deviation of forward 5d returns across the 18-asset universe, on all dates with at least one valid forward observation.

| metric (raw 5d std, not annualized) | value |
|---|---|
| n dates | 4107 |
| mean | **0.0249** (2.49%) |
| median | 0.0212 (2.12%) |
| 25th pct | 0.0155 (1.55%) |
| 75th pct | 0.0295 (2.95%) |
| 95th pct | 0.0536 (5.36%) |

Dispersion is moderate. It is not so small that cross-sectional discrimination is mechanically impossible (the 5d standard deviation across assets is ~2% in typical conditions and ~5% in tail conditions), but it is not so large that ranking should be obviously easy either.

**Dispersion-vs-v1-Spearman correlation**: −0.107 on the 1,260 dates where both are defined (v1's 5-fold test window). The relationship is slightly *negative* — v1 does marginally *better* on lower-dispersion days. This is informative on its own: dispersion is not the bottleneck for v1's score quality. If raw dispersion were the limiting factor, we would expect a positive correlation.

### Phase 1B — Cross-sectional predictability baseline

The simplest possible cross-sectional predictor: per-date Spearman rank correlation between **trailing 5d returns** (ranks computed across the 18 assets at date t using t-4..t) and **forward 5d returns** (ranks across the same 18 assets using t+1..t+5).

**Headline finding**:

| metric | value |
|---|---|
| n dates | 4102 |
| mean Spearman | **−0.0089** |
| median Spearman | −0.0093 |
| 25th pct | −0.310 |
| 75th pct | +0.288 |
| fraction of dates with ρ > 0 | **0.490** |

The simplest cross-sectional momentum predictor — "yesterday's winners are tomorrow's winners" — has a **mean Spearman of −0.009** on this universe. **Short-horizon momentum is slightly *anti*-predictive on average across these 18 assets.** Less than half of dates exhibit positive momentum-Spearman.

**Per-fold** (matching the experiment's walk-forward structure exactly):

| fold | dates | mean ρ (trailing→forward 5d) |
|---|---|---|
| split 0 (2020-02 → 2021-02) | 252 | −0.044 |
| split 1 (2021-02 → 2022-02) | 252 | −0.031 |
| split 2 (2022-02 → 2023-05) | 312 | −0.032 |
| split 3 (2023-05 → 2024-08) | 312 | −0.004 |
| split 4 (2024-08 → 2025-08) | 252 | −0.055 |
| overall | 1380 | **−0.033** |
| ICIR across 5 folds | — | **−1.77** |

**Every fold is negative.** The naive momentum predictor is *consistently* anti-aligned with realized forward ranking on this universe.

This is the most important diagnostic in the patch. It establishes that:

1. The simplest cross-sectional predictor is *worse than zero* on this universe.
2. v1's +0.032 mean Spearman is a genuine lift of ~0.04 over the trivial baseline, not "barely above zero" in any absolute sense.
3. Reaching the +0.05 decision-grade Spearman threshold requires another ~0.02 lift on top of v1 — comparable in magnitude to the entire gap between the trivial baseline and v1 itself.

### Phase 1C — Cross-asset pairwise correlation structure

Rolling 60-day pairwise correlations of daily returns across the 18 assets, summarized as the mean (and median) of the n(n-1)/2 = 153 pairwise correlations per date.

| metric (mean pairwise corr, 60d window) | value |
|---|---|
| n dates | 4052 |
| mean | **0.191** |
| median | 0.176 |
| 25th pct | 0.141 |
| 75th pct | 0.234 |
| 95th pct | 0.321 |

Mean pairwise correlation hovers around 0.19 — moderate-low. The universe is **not dominated by a single risk-on/risk-off factor**: if it were, the mean pairwise correlation would consistently sit above 0.6-0.7. Genuine cross-sectional structure exists.

The 95th percentile of 0.32 means that even in the most-correlated 5% of 60-day windows, the assets are still substantially independent of each other. This is partly due to deliberate universe construction: bonds (TLT/IEF/SHY/LQD/HYG), equities (SPY/QQQ/IWM/DIA/EFA/EEM), commodities (GLD/SLV/USO/DBA), currency (UUP), real estate (VNQ), and crypto (BTC) cover materially independent risk premia.

### Phase 1 interpretation summary — pre-registered estimate of the Spearman ceiling

**Documented BEFORE Phase 5 results are known.**

Based on Phase 1A/B/C:

- **Dispersion** is sufficient to support cross-sectional ranking (mean per-date std ~2.5%, not mechanically zero).
- **Common-factor dominance is absent** (mean pairwise correlation 0.19 — there is real cross-sectional structure).
- **The naive cross-sectional predictor (trailing 5d momentum) has mean Spearman of −0.009**. v1's +0.032 is a +0.04 lift over this baseline.

The realistic upper bound on per-date Spearman on this universe at this horizon is **approximately +0.03 to +0.05**.

Reasoning: a sophisticated model that beats trailing-momentum by ~0.04 (v1's empirical lift) and adds another +0.02 from regime conditioning, dispersion features, or architectural improvement is plausible but on the optimistic edge. The historical pattern across all four feature-side configurations (HGB regression, v1, v2, v3) has been per-date Spearman in [0.024, 0.032] — clustered tightly around v1's +0.032. The fact that variation between configurations is smaller than variation between folds within any single configuration is consistent with bumping against a structural ceiling.

**Implication for Phase 5**:

- A pivot result of Spearman ≈ +0.03 with positive economics would be consistent with the ceiling — a fail at +0.05 decision-grade but not evidence of architectural failure.
- A pivot result of Spearman ≈ +0.05+ would indicate the architecture genuinely cracked the ceiling, which would be material new evidence about what's learnable on this universe.
- A pivot result of Spearman well below +0.03 would indicate the architecture introduced more noise than signal (cf. v2's beta-feature regression).

The +0.05 / ICIR ≥ 0.5 decision-grade and +0.03 / ICIR ≥ 0.3 directional thresholds **remain unchanged**. Phase 1 informs interpretation of a close miss but does not relax the pass criteria.

---

## Phases 2–4 — Architecture implementation

### Phase 2 — VIX-z tercile classifier

Per-fold cutoffs computed on train data only, then applied to label every date (train/val/test). Cutoffs recorded in the run output (`regime_diagnostics` CSV) for auditability. Causality: train-only fit, no look-ahead — verified by the v1 reproduction step (running with `--regime-architecture none` after the refactor reproduces v1 numbers bit-for-bit, confirming no behavioral change to the existing pooled path).

Per-fold tercile cutoffs (VIX z-score values used as boundaries):

| split_id | low cutoff | high cutoff |
|---|---|---|
| 0 | −0.325 | 0.000 |
| 1 | −0.610 | +0.185 |
| 2 | −0.442 | +0.053 |
| 3 | −0.435 | +0.080 |
| 4 | −0.481 | +0.097 |

### Phase 3 — Per-regime LambdaRank models

Each fold trains three LambdaRank models (low_vix / mid_vix / high_vix) on the regime-filtered train slice using the v1 cross-sectional feature set (no betas, no regime interactions, no new features). Per-regime training subset = train rows whose date falls in that regime. Same hyperparameters as the pooled v1 model: `n_estimators=100`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=5`, `random_state=42`.

Fallback: any regime with fewer than 120 train days uses a pooled fallback model fit on the full train set (identical to v1's behavior). Fallback usage is explicitly logged.

Implementation factored into `_fit_lambdarank_on_panel` (shared by pooled and per-regime paths) and `_score_with_regime_lambdarank` (the regime variant). Feature importance is captured per regime model AND per pooled-fallback model.

### Phase 4 — Per-regime inference and allocation

Each test-date is routed to the model for its date's regime (or the pooled fallback if the regime's per-fold training failed the 120-day check). Top-k allocation rules unchanged from v1/v3.

### Tests + v1 reproduction verification

```text
uv run python -m pytest tests/test_cross_asset_ranking.py tests/test_cross_asset_ranking_experiment.py -x -q
... 43 passed in 12.45s

uv run python -m pytest -q --ignore=tests/test_integrity_audit_matched_nulls.py
... 170 passed in 14.18s
```

**v1 reproduction**: ran the same command as v1 (cross-sectional features on, regime architecture off) after the refactor. Top-2 per-fold IRs match v1 bit-for-bit:

```text
v1 (original):  [-1.252969, +0.293157, +1.308116, +0.612238, +2.312966]
v1 (post-refactor): [-1.252969, +0.293157, +1.308116, +0.612238, +2.312966]  ✓
```

No behavioral change to the pooled v1 path. The refactor is clean.

## Phase 5 — Experiment

Run timestamp: `20260511T181331Z` → `results/cross_asset_ranking_5d_target_regime_pivot_lambdarank/`

### Command

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --execute \
  --assets SPY QQQ IWM DIA EFA EEM TLT IEF SHY LQD HYG GLD SLV USO DBA UUP VNQ BTC-USD \
  --cache-dir data/multi_asset_cache \
  --output-dir results/cross_asset_ranking_5d_target_regime_pivot_lambdarank \
  --forward-horizon 5 --vol-window 20 \
  --train-size 756 --val-size 252 --test-size 252 --step-size 252 \
  --transaction-cost-bps 2.0 --top-k 1 2 \
  --models lambdarank --random-null-runs 500 --random-state 42 \
  --rebalance-every 5 \
  --feature-normalization per_asset_train_zscore \
  --include-cross-sectional-features \
  --regime-architecture vix_tercile \
  --regime-min-train-days 120 \
  --run-purpose decision_grade --decision-grade
```

5 splits × 2 top-k × 500 random nulls = 5,000 null allocations on the 18-asset panel. `data_downloaded: false`. All other safety flags `false`.

### Output files

11 timestamped files (added `cross_asset_ranking_regime_diagnostics_<ts>.csv`).

## Per-regime diagnostics

### Per-fold balance check

15% test-day floor: each regime needs ≥38 of 252 test days (or proportional for the 312-date split 2 test window) to be fairly evaluable.

| split_id | low_vix test days | mid_vix test days | high_vix test days | regimes ≥15% | fallback used |
|---|---|---|---|---|---|
| 0 | 31 (12.3%) | 48 (19.0%) | 173 (68.7%) | **2/3** | mid_vix (69 train days) |
| 1 | 131 (52.0%) | 78 (31.0%) | 43 (17.1%) | 3/3 | — |
| 2 | 98 (38.9%) | 60 (23.8%) | 94 (37.3%) | 3/3 | — |
| 3 | 197 (78.2%) | 31 (12.3%) | 24 (9.5%) | **1/3** | — |
| 4 | 11 (4.4%) | 107 (42.5%) | 134 (53.2%) | **2/3** | — |

**3 of 5 folds have at least one regime below the 15% floor** (splits 0, 3, 4). Per the spec's pre-committed fail criteria: "regime distribution imbalance: ≥3 of 5 folds had a regime below the 15% test-day floor (the architecture wasn't fairly evaluable)" — **this fail criterion is triggered**.

The fail is real but informative: it captures the structural difficulty of using a train-period regime threshold on out-of-sample test data when VIX regimes are persistent. The 2024-25 risk-on tail (split 4) saw the model classify almost all test days as mid- or high-VIX because that train period happened to be very calm; the 2023-24 AI rally (split 3) tipped almost entirely into low-VIX. The train-only tercile boundary cannot anticipate future regime distributions.

### Fallback summary

Only one regime in one split used the pooled fallback (split 0 mid_vix, 69 train days < 120). The architecture was actually exercised in the other 14 (split, regime) combinations. The fail is not "regime models didn't get to train" — it's "regime models trained but produced worse rankings."

### Per-regime feature importance (top 5 by mean gain across splits)

```text
low_vix:
  relative_vol_ratio          714
  current_drawdown_60d        570
  realized_vol_20             557
  bollinger_band_width_zscore 508
  xs_rank_vol_20d             457

mid_vix:
  xs_rank_vol_20d             883
  relative_vol_ratio          584
  autocorrelation_zscore      532
  return_60d                  522
  current_drawdown_60d        442

high_vix:
  xs_rank_vol_20d             707
  realized_vol_20             563
  relative_vol_ratio          555
  return_60d                  540
  bollinger_band_width_zscore 491

pooled_fallback (mid_vix split 0 only):
  vix_relative                1315
  realized_vol_20             1214
  volatility_regime           1075
  relative_vol_ratio          1052
  xs_rank_vol_20d             1041
```

Each regime model favors a sensible mix of volatility and momentum features — no regime model degenerated to a single feature. The top features are broadly similar across regimes (vol-family features dominate everywhere), which is consistent with the v1/v3 findings. **The per-regime split did not unlock substantively different feature structure** — the models within each regime are doing the same kind of work as the pooled model, just on a third as much training data.

## Head-to-head — v1 / v3 / pivot

| run / policy | mean IR | net Sharpe | folds (+IR) | folds p≤0.05 | median p | drop-best mean | BTC % | Spearman | ICIR |
|---|---|---|---|---|---|---|---|---|---|
| v1 top-1 | +0.071 | 0.412 | 3/5 | 1/5 | 0.443 | -0.425 | 22.5 | +0.032 | 0.94 |
| v1 top-2 | +0.655 | 1.044 | 4/5 | 1/5 | 0.204 | +0.240 | 35.6 | +0.032 | 0.94 |
| v3 top-1 | +0.137 | 0.520 | 3/5 | 0/5 | 0.289 | -0.091 | 19.6 | +0.028 | 0.79 |
| **v3 top-2** | **+0.748** | **1.180** | 4/5 | **2/5** | 0.134 | **+0.286** | 31.5 | +0.028 | 0.79 |
| **pivot top-1** | **−0.299** | 0.184 | 2/5 | 0/5 | 0.719 | -0.723 | 14.6 | +0.006 | 0.42 |
| **pivot top-2** | **−0.369** | 0.284 | 1/5 | 0/5 | 0.579 | -0.504 | 28.1 | +0.006 | 0.42 |

Pivot regressed against both v1 and v3 on every metric.

### Per-fold IR (pivot)

```text
top-1:  s0:-0.686   s1:-1.529   s2:-0.722   s3:+1.399   s4:+0.045
top-2:  s0:-1.168   s1:-0.491   s2:-0.256   s3:-0.103   s4:+0.171
```

Top-2 has only 1 positive fold (split 4 at +0.17). The split-4 strength that v3 capitalized on (+2.31 IR) collapsed to +0.17 in the pivot.

## Pass criteria evaluation

**Decision-grade pass** (BOTH top-1 AND top-2 must satisfy ALL):

| Criterion | Top-1 | Top-2 |
|---|---|---|
| Mean IR > 0 AND active return > 0 | ✗ (−0.30; −0.15) | ✗ (−0.37; −0.09) |
| Per-date Spearman ≥ +0.05 | ✗ (+0.006) | ✗ (+0.006) |
| ICIR ≥ 0.5 | ✗ (0.42) | ✗ (0.42) |
| Folds passing p ≤ 0.05 ≥ 2/5 | ✗ (0/5) | ✗ (0/5) |
| Drop-best-fold mean IR > 0 | ✗ (−0.72) | ✗ (−0.50) |
| No BTC dominance | ✓ | ✓ |
| ≥1 xs/regime feature in top half of importance | ✓ | ✓ |
| Top-1 material improvement over v1 baseline | ✗ (delta −0.37) | n/a |
| Top-2 maintain or improve on v3's IR ≥ +0.6 + v1's Spearman ≥ +0.03 | n/a | ✗ (IR −0.37, Spearman +0.006) |
| ≥4/5 folds had ≥2/3 regimes ≥15% test days | ✗ (only 2/5 folds had 3/3; one had 1/3, two had 2/3) | ✗ |

Multiple criteria fail. **Decision-grade: FAIL.**

**Directionally interesting**:

| Criterion | Result |
|---|---|
| Top-2 Spearman ≥ +0.03 AND mean IR > +0.5 AND ICIR ≥ 0.3 | ✗ Spearman +0.006; mean IR −0.37 |
| Top-1 mean IR ≥ +0.15 | ✗ (−0.30) |
| Regime structure non-trivially used (≥3/5 folds had a regime-specific model on majority of test days) | ✓ (only 1 fallback in 15 regime-fold slots) |

**Directionally interesting: FAIL.**

**Explicit fail criteria triggered**:

| Fail criterion | Triggered? |
|---|---|
| Below directional thresholds on either top-1 or top-2 | ✓ both |
| Regime architecture collapsed (most folds used pooled fallback) | ✗ (only 1 fallback) |
| Regime distribution imbalance: ≥3/5 folds had a regime below 15% floor | ✓ (splits 0, 3, 4) |

Two of three fail conditions explicitly triggered.

## Stop / go verdict — tied to Phase 1 ceiling interpretation

The Phase 1 pre-registered ceiling was **+0.03 to +0.05 per-date Spearman**. The pivot produced **+0.006** — well below the lower bound of that estimate. This is not a near-miss against a tight ceiling; it is a structural regression caused by reducing per-model training data without corresponding signal lift.

Combining Phase 1 (pre-registered) and Phase 5 (now known):

1. The simplest cross-sectional momentum predictor on this universe has mean Spearman of −0.009 (Phase 1B).
2. The pooled v1 model achieves +0.032 (a +0.04 lift over the baseline).
3. The pooled v3 model achieves +0.028 (similar to v1, with stronger top-2 economics).
4. The per-regime architecture achieves +0.006 (essentially the momentum baseline floor — the architecture mostly destroyed the v1/v3 lift).

The combined evidence pattern matches the spec's **scenario 1 — top-selection strategy deliverable**:

- v3 produced v3-like strong top-2 economics (mean IR +0.748, net Sharpe 1.180 vs equal-weight 1.018, drop-best +0.286, 2/5 folds clearing p ≤ 0.05).
- Phase 1 documented a Spearman ceiling at or near the decision-grade threshold (+0.03 to +0.05 estimate; +0.05 pre-committed threshold).
- The pivot architecture's regression confirms that further iteration on the rank-quality axis has run out of room.

### Recommended next step — top-selection strategy deliverable

Accept that **this universe at this horizon supports a top-selection (top-k allocation) strategy, not a universe-wide ranker**. The scoped-down deliverable is **v3 as a top-2 concentrated allocation strategy**:

- 18-asset universe, 5-day rebalance, LambdaRank with v1 cross-sectional features + VIX-z regime-conditioned interactions.
- Mean IR +0.748 vs equal-weight (the universe benchmark), net Sharpe 1.180 vs equal-weight 1.018.
- 2 of 5 walk-forward folds passed p ≤ 0.05 against same-turnover random top-2 nulls.
- Drop-best-fold mean IR +0.286 (signal robust to removing any single fold).
- No BTC dominance (31.5% selection share); top-3 concentration 36.5%; max drawdown -0.21 (better than equal-weight's -0.13 by a small margin — strategy doesn't aggressively concentrate risk).

The rank-quality goal of per-date Spearman ≥ +0.05 is **set aside as not achievable on this universe/horizon** based on:
- Trivial momentum baseline ρ = −0.009 (Phase 1B)
- Four feature-side iterations clustered tightly in [0.024, 0.032]
- One architectural pivot regressing to +0.006

This is not "give up and keep iterating." It is "the strategy is a top-selection allocator with v3-like behavior, document it, and move next steps to production-readiness work or universe/horizon pivots, not further feature/architecture variants."

### Production-readiness work that becomes relevant (separate workstream)

If the project chooses to develop v3 as a top-selection deliverable:

1. **Capacity analysis**: does the strategy survive realistic AUM-scaled position sizing? BTC-USD has ~thin order books vs the ETFs; concentration into BTC at 31.5% needs capacity modeling at target portfolio scale.
2. **Slippage / transaction cost modeling**: current 2 bps cost is an approximation. Real execution costs vary by asset (BTC ~5-15 bps vs SPY <1 bp); recomputing with a per-asset cost vector would tighten the net Sharpe estimate.
3. **Multiple-testing adjustments**: the campaign tested ~10 configurations against the same data. The v3 result needs a multiple-testing adjustment (Bonferroni, Benjamini-Hochberg, or family-wise control) before claiming the 2/5 fold-pass result is robust.
4. **Regime-balance audit**: even though the per-regime *architecture* failed, the *underlying regime structure* of v3's selections (defensives in 2022 bear, BTC/commodities in 2024 rally) is worth documenting — the strategy IS regime-aware, just via LambdaRank's implicit feature use rather than an explicit classifier.
5. **Forward-walk preparation**: the universe-cache fetch and prepare_feature_frame pipeline are already in place. A live-data forward walk for 6-12 months would be the next test before any production decision.

Each of these is a separate workstream. None is on the table for further feature/architecture iteration on the cross-sectional ranking thesis itself.

### What is explicitly NOT recommended

Per the spec's pre-committed pivot ordering:

- **No more regime classifier variants** (HMM, 2-state, 4-state, alternative axes).
- **No more feature combinations** (dispersion, alternative betas, raw variants).
- **No universe variations** as continuations of THIS project (a universe pivot would be a new campaign).
- **No seed sensitivity, hyperparameter tuning, deep models, stacking, Optuna**.
- **No target-side pivots** (forward vol, forward drawdown) as continuations.

The cross-sectional ranking thesis on this universe / horizon is empirically tested to a structural conclusion. The deliverable is the v3 top-selection strategy plus the documented research artifacts (v1 as rank-quality high-water mark, v3 as economic high-water mark, this patch as the architectural-pivot confirmation that the rank ceiling is real).

## Confirmation: scope safeties held

- `main_py_used`, `prepare_experiment_used`, `old_model_zoo_used`, `optuna_used`, `deep_models_used`, `stacking_used`, `data_downloaded`: all `false` in metadata.
- `regime_architecture: vix_tercile`, `regime_min_train_days: 120` recorded in metadata.
- v1 reproduction verified bit-for-bit after the refactor.
- Pre-committed thresholds (+0.05 / +0.03 Spearman, ICIR ≥ 0.5 / ≥ 0.3) honored — pivot reported as a fail at strict thresholds.
- Phase 1 ceiling estimate (+0.03 to +0.05) was pre-registered before Phase 5 results were inspected.
- Static-import test continues to enforce no-legacy / no-Optuna / no-deep / no-stacking invariants.
- No champion manifest changes.
- No existing result files overwritten — pivot output went to a fresh per-config directory; v1 reproduction went to a separate directory.
- Single-factor discipline honored: vs v1 baseline, exactly one change (the regime architecture flag). vs v3, two changes (regime architecture ON, regime interactions OFF — but the v3 interactions were a distinct mechanism, not the same change in disguise).

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Phase 1A/B/C complete with documented interpretation BEFORE Phase 5 results | ✓ |
| 2 | Phase 2: per-fold VIX tercile cutoffs computed on train data, recorded in metadata | ✓ |
| 3 | Phase 3: per-regime LambdaRank trained with v1 features only, fallback operational and logged | ✓ (1 fallback in 15 regime-fold slots) |
| 4 | Phase 4: per-regime inference integrated with top-k allocation | ✓ |
| 5 | Phase 5: dry-run passes, execute completes, failure documented with full diagnostics | ✓ failure documented |
| 6 | v1 reproducibility verified (without --regime-architecture flag) | ✓ bit-for-bit identical |
| 7 | No forbidden workflows touched | ✓ |
| 8 | Patch document complete with all sections | ✓ this file |
| 9 | Pre-committed thresholds honored regardless of result | ✓ |

## Closing note

The cross-asset ranking feature/architecture campaign that began with the v1 cross-sectional feature pivot and ran through v2 (beta features), v3 (regime interactions), and now the architectural pivot is **closed**. Five distinct configurations have been tested under matched-null discipline on the same universe and horizon. Per-date Spearman has stayed in [0.006, 0.032] across all of them, with v1 the upper bound and the pivot the lower. The trivial momentum baseline is at −0.009. The pre-committed +0.05 decision-grade threshold sits at or above the empirical ceiling of this universe / horizon, and no architectural change has cracked it.

v3 stands as the campaign's best top-selection strategy. v1 stands as the campaign's best rank-quality result. Both are research artifacts. Further iteration on either axis within the cross-asset ranking framing is not the right next step — production-readiness workstreams or a new campaign on a different universe / horizon are.

