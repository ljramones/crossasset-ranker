# Patch: Cross-Asset Ranking Split-2 Forensics

Date: 2026-05-10
Inputs analyzed:

- `results/cross_asset_ranking_decision_run/cross_asset_ranking_*_20260510T231114Z.csv` (daily, all 3 models)
- `results/cross_asset_ranking_rebalance_5d/cross_asset_ranking_*_20260511T000316Z.csv` (5d, HGB only)
- `results/cross_asset_ranking_rebalance_20d/cross_asset_ranking_*_20260511T000803Z.csv` (20d, HGB only)
- `data/multi_asset_cache/*.csv` for asset-level realized returns

This patch is **pure post-hoc analysis**. No model training, no Optuna, no deep models, no stacking, no data download, no feature-engineering changes.

## TL;DR diagnosis

**Hypothesis A — HGB chose wrong assets despite real ranking opportunity** is the primary cause. **Hypothesis C — feature scale mismatch** is a contributing factor. Hypotheses B (low dispersion), D (calendar), F (strong null) are all rejected by the data.

The deeper weakness exposed by split 2 is that HGB has **minimal per-date ranking skill** in any fold (mean Spearman rank correlation between scores and the realized target is +0.17 in the best fold, near zero in three folds, and slightly negative in one). The model's successes come from sustained directional bets that happen to align with the prevailing market regime, not from per-date cross-sectional edge. Split 2 inverted the regime, so the model's bull-market directional preferences became actively wrong.

**Recommended next step: per-asset feature normalization sensitivity**, with a follow-on consideration of switching from regression loss to a pairwise/listwise rank loss to address the deeper per-date ranking weakness.

## 1. Split 2 date range

| split_id | test start | test end | n_test_dates |
|---|---|---|---|
| 0 | 2020-02-18 | 2021-02-16 | 252 |
| 1 | 2021-02-17 | 2022-02-14 | 252 |
| **2** | **2022-02-15** | **2023-05-12** | **252** |
| 3 | 2023-05-15 | 2024-08-09 | 252 |
| 4 | 2024-08-12 | 2025-08-13 | 252 |

Split 2 covers the **2022 bond crash + equity bear market + early 2023 banking stress** (Russia/Ukraine, Fed hiking cycle, October-2022 equity bottom, November-2022 BTC bottom, March-2023 SVB collapse).

Train window for split 2 ≈ 2017-09 → 2021-02 — almost entirely a strong bull market for risk assets, with TLT/GLD as low-vol diversifiers. The model's training distribution and test distribution differ structurally.

## 2. Asset realized performance during split 2 test window

Sorted by annualized return (best to worst):

| asset | cum return | ann return | ann vol | Sharpe | max drawdown |
|---|---|---|---|---|---|
| **GLD** | **+6.91%** | **+5.54%** | 15.6% | **+0.36** | -21.0% |
| SPY | -4.36% | -3.54% | 22.5% | -0.16 | -22.1% |
| QQQ | -5.63% | -4.57% | 29.6% | -0.15 | -29.6% |
| IWM | -12.37% | -10.11% | 26.3% | -0.38 | -22.4% |
| TLT | -21.15% | -17.46% | 20.4% | -0.86 | -33.7% |
| BTC-USD | -37.06% | -31.20% | 62.6% | -0.50 | -66.7% |

**GLD was the only asset with positive return and positive Sharpe.** The other five lost between 4% and 37%. The "right" allocation for split 2 was GLD-heavy.

## 3. What HGB top-2 actually selected in split 2

Selection counts (out of 252 test dates):

| asset | dates held | % of dates | rank by selection |
|---|---|---|---|
| IWM | 117 | **46.4%** | 1 |
| QQQ | 109 | **43.3%** | 2 |
| SPY | 85 | 33.7% | 3 |
| BTC-USD | 73 | 29.0% | 4 |
| TLT | 73 | 29.0% | 4 |
| **GLD** | **47** | **18.7%** | **6 (least)** |

Average HGB raw score per asset over all 252 test dates:

```text
QQQ        +0.140
IWM        +0.126
SPY        +0.120
BTC-USD    +0.102
TLT        +0.095
GLD        +0.083    <-- LOWEST
```

HGB consistently scored equities highest and **gold lowest** throughout the test period. It selected the only winning asset less often than any other — almost exactly inverting the realized winner ranking.

Selection patterns across all five splits show HGB always favors something — but *what* it favors is fold-dependent and seemingly tracks the most recent bull regime in each train window:

| split | top selections (% of dates) |
|---|---|
| 0 | QQQ 54%, IWM 44%, GLD 42% |
| 1 | TLT 54%, IWM 46%, BTC 34% |
| **2** | **IWM 46%, QQQ 43%, SPY 34%** (equities → bear market) |
| 3 | SPY 42%, IWM 38%, BTC 37% |
| 4 | SPY 44%, TLT 42%, QQQ 34% |

## 4. Hit rates and realized rank vs the actual top assets

For each test date, the actual top-1 and top-2 assets are determined by realized forward 20-day return.

| metric | split 2 result | random baseline (k=2 of 6) |
|---|---|---|
| HGB picks include actual top-1 | **0.377** | 2/6 = **0.333** |
| HGB picks include any of actual top-2 | **0.571** | 1 − C(4,2)/C(6,2) = **0.600** |
| Mean realized rank of HGB picks (1 = best) | **3.44** | (6+1)/2 = **3.50** |
| Median realized rank | **4.00** | 3.50 |

**HGB is statistically indistinguishable from random asset selection on a per-date basis in split 2.** Top-1 hit rate is barely above random (0.377 vs 0.333). Top-2 hit rate is *worse* than random (0.571 vs 0.600). Mean realized rank of selections matches random.

The model isn't choosing anti-skillfully on a per-date basis — it's choosing roughly randomly, with a sustained directional tilt toward equities that proved wrong over the full window.

## 5. Score-vs-target rank correlation (Spearman) per date, per split

Computed per date: rank assets by HGB score, rank assets by realized `forward_20d_risk_adjusted_return`, take Pearson correlation of the two rank vectors.

| split | n_dates | mean ρ | median ρ | frac dates ρ > 0 | p25 ρ | p75 ρ |
|---|---|---|---|---|---|---|
| 0 | 252 | **+0.168** | +0.200 | **0.671** | -0.143 | +0.486 |
| 1 | 252 | -0.069 | -0.086 | 0.437 | -0.371 | +0.257 |
| **2** | 252 | **+0.036** | +0.057 | 0.540 | -0.371 | +0.486 |
| 3 | 252 | +0.068 | +0.086 | 0.567 | -0.257 | +0.429 |
| 4 | 252 | +0.050 | +0.086 | 0.567 | -0.314 | +0.429 |

**This is the deepest finding in the patch.** Across all five folds, mean per-date Spearman ρ is between -0.07 and +0.17. There is *some* genuine ranking edge in split 0 (mean ρ = +0.17, 67% of dates positive) and that fold returned IR +1.72. Every other fold has mean |ρ| < 0.10 — the model has effectively no per-date cross-sectional skill in 4 of 5 folds. The high IR on those folds was driven by sustained correct directional bets, not by per-date discrimination.

In split 2 specifically, mean ρ = +0.036 is essentially zero. The model isn't *anti*-correlated; it's just not correlated. With near-zero per-date skill, a sustained wrong directional bet (overweight equities in a bear market) determines the result.

## 6. Null distribution context for split 2 (HGB top-2)

| quantity | value |
|---|---|
| Canonical HGB IR (split 2) | **−2.319** |
| Random null distribution n (3 model calls × 500 nulls) | 1500 |
| Null mean IR | -0.413 |
| Null median IR | -0.377 |
| Null std IR | 0.999 |
| Null min IR | -3.183 |
| Null p5 IR | -2.061 |
| Null p95 IR | +1.273 |
| Nulls worse than canonical | 39 / 1500 = **2.6%** |

Two important observations:

1. **The null distribution itself is shifted negative** (mean IR -0.41). This means even random asset selection underperformed equal-weight in split 2 — the universe-wide bear market hurt anything except a GLD-heavy allocation.
2. **HGB sits at percentile 2.6 of the null distribution.** It performed worse than ~97% of random top-2 allocations. This is *not* a strong-null problem — the null is weak — it's a wrong-pick problem. The model actively chose worse than random.

**Hypothesis F (null was unusually strong) is rejected.**

## 7. Cross-sectional dispersion across folds

Mean cross-sectional std of daily asset returns:

| split | period | mean dispersion | equal-weight cum_ret |
|---|---|---|---|
| 0 | 2020-02 → 2021-02 | 0.01875 | +62.9% (COVID rebound) |
| 1 | 2021-02 → 2022-02 | 0.01627 | +2.4% (sideways) |
| **2** | **2022-02 → 2023-05** | **0.01415** | **−9.9%** (bear) |
| 3 | 2023-05 → 2024-08 | 0.01221 | +38.3% (AI rally) |
| 4 | 2024-08 → 2025-08 | 0.01165 | +32.4% |

**Dispersion in split 2 is higher than splits 3 and 4.** There was *more* ranking opportunity than in the bull rallies that followed. **Hypothesis B (low dispersion) is rejected.**

## 8. Feature scale across assets in split 2

Means (μ) and standard deviations (σ) of key features over the split-2 test window:

| feature | SPY | QQQ | IWM | TLT | GLD | BTC-USD |
|---|---|---|---|---|---|---|
| `return_1d` σ | 0.014 | 0.018 | 0.016 | 0.013 | 0.010 | **0.042** |
| `return_5d` σ | 0.029 | 0.039 | 0.034 | 0.025 | 0.021 | **0.096** |
| `return_20d` σ | 0.052 | 0.072 | 0.061 | 0.050 | 0.046 | **0.187** |
| `realized_vol_20` μ | 0.014 | 0.018 | 0.016 | 0.013 | 0.010 | **0.040** |
| `vol_ratio` μ (already normalized) | 0.96 | 0.95 | 0.96 | 1.00 | 0.97 | 0.90 |
| `momentum_norm` μ (already normalized) | +0.05 | +0.17 | -0.09 | -0.13 | **+0.35** | -0.01 |

Two observations:

1. **Raw return/vol features have severe cross-asset scale mismatch** — BTC's `return_20d std` is 3-4× the equities and ~10× TLT. HGB trained on pooled cross-asset data with raw-scale features will learn implicit asset identification: "this row has return_20d magnitude in the BTC range" → "treat it as BTC-like." That's a brittle proxy for cross-sectional ranking.
2. **`momentum_norm` for GLD in split 2 was +0.35**, the highest of any asset. So a model that uses normalized momentum should have favored GLD. The fact that HGB scored GLD lowest tells us the model wasn't relying on `momentum_norm` — it was leaning on raw-scale features that flagged equities as "high-magnitude movers" in the train window.

**Hypothesis C (feature scale mismatch) is supported as a contributing factor.** The fix is per-asset z-scoring of return/vol features (or per-date cross-sectional z-scoring) so the model sees comparable inputs.

## 9. Calendar / BTC alignment

| asset | dates in split-2 window | dates outside the inner-join |
|---|---|---|
| SPY | 312 | 0 |
| QQQ | 312 | 0 |
| IWM | 312 | 0 |
| TLT | 312 | 0 |
| GLD | 312 | 0 |
| BTC-USD | 312 | 0 |

After `prepare_single_asset_feature_frame` restricts each asset to dates with non-NaN BenchmarkClose (i.e. SPY's calendar), all six assets share exactly the same 312 dates in the split-2 window. The inner-join in `build_cross_asset_panel` is a no-op here. There are no missing-asset rows, no extra BTC weekend rows, no NaN-dominated rows.

**Hypothesis D (calendar issue) is rejected.**

## 10. Summary diagnosis

Mapping the data to the seven candidate hypotheses:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| A. HGB chose wrong assets despite strong dispersion | **PRIMARY CAUSE** | GLD was the only winner; HGB selected GLD least; sat at 2.6th percentile of null distribution |
| B. Asset dispersion was too low / weak ranking opportunity | **REJECTED** | Split 2 dispersion 0.01415, higher than splits 3 and 4 |
| C. Feature scale mismatch distorted scores | **CONTRIBUTING FACTOR** | BTC raw-feature scale 3-4× equities; `momentum_norm` (already normalized) showed GLD as most attractive but HGB ignored it |
| D. Calendar alignment issue | **REJECTED** | All 6 assets present on every panel date in split 2 |
| E. Split 2 market regime is structurally different | **TRUE BUT NOT THE BUG** | It is a regime change (bull → bear), but a model with genuine cross-sectional skill should still find the winner; HGB's per-date rank correlation was +0.04 — basically chance |
| F. Null distribution was simply too strong | **REJECTED** | Null mean IR was -0.41; random underperformed too, HGB just underperformed worse |
| G. Inconclusive | partially — see below | Per-date Spearman ρ across all folds is in [-0.07, +0.17]; the model has very weak per-date ranking skill in any fold, including the ones it "passed" |

**The deeper finding (hypothesis G in spirit, hypothesis A in letter):** HGB's good folds aren't really wins of *ranking* — they're wins of *direction*. When the train regime matches the test regime, HGB's persistent directional tilt happens to be right. When the regime inverts (split 2), the directional tilt becomes wrong and there's no per-date discrimination to fall back on.

This explains the rebalance robustness result too. At 5d rebalance, the model still has the same weak per-date ranking edge, but the lower-decision-frequency null comparison no longer rewards "many cheap chances to be slightly right" — it rewards "fewer chances to be substantially right." HGB doesn't have substantially-right per-date discrimination, so it doesn't beat the null at 5d.

## 11. Recommended next step

Per the spec's hypothesis-conditional recommendations:

> If the report points to feature scale mismatch, the next task should be: per-asset feature normalization sensitivity

That's the right next step. Concretely:

1. **Per-asset feature normalization sensitivity test.** Add per-asset rolling z-scoring (e.g. 252-bar rolling mean / std per (asset, feature)) for `return_1d/5d/20d/realized_vol_20` and re-run the daily and 5d decision-grade evaluations. If linear regression also revives, the scale mismatch was a real impediment for all models. If HGB top-2's split-2 IR moves substantially (not necessarily passing, just moving), the model is now seeing comparable inputs.

2. **Cross-sectional (per-date) z-score variant.** Same idea but normalize per date across assets instead of per asset across time. Test as a separate pass — these two normalizations capture different invariants.

3. **Only after (1) and (2)**, reconsider the deeper finding: even with normalized features, the model's per-date Spearman rank correlation needs to climb above ~0.15 for the strategy to be more than a regime-direction bet. If normalization doesn't move ρ, the next experiment is a **rank-based loss function** (LambdaRank or pairwise ranking objective) — the current regression target may not be teaching the model to discriminate within-date.

4. **Do not run seed sensitivity yet.** Per the spec: seed instability without diagnosing the cause is not informative; the cause has now been diagnosed.

5. **Defer the rebalance-frequency rerun on the normalized panel** until (1) and (2) show that normalization actually changes the per-date ranking quality. If it doesn't, the rebalance question is moot.

## Confirmation: scope safeties held

- No model training, no Optuna, no deep models, no stacking, no data download.
- No feature engineering or model logic was changed.
- No existing result files were overwritten.
- No champion manifest was modified.
- All inputs were existing artifact CSVs and the read-only data cache.
- No legacy modules touched.

## Definition of Done

| # | Requirement | Status |
|---|---|---|
| 1 | Report exists | ✓ this file |
| 2 | No training run | ✓ |
| 3 | No data download | ✓ |
| 4 | No old workflow used | ✓ |
| 5 | Split 2 diagnosis recorded | ✓ Hypothesis A primary, C contributing |
| 6 | Next step recommended | ✓ per-asset normalization sensitivity |
