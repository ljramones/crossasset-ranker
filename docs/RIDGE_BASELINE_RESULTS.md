# Ridge Baseline Results — Extraction and 2×2 Comparison

Generated 2026-05-11 from on-disk decision-grade outputs. No new experiments; metrics are read as-is from each run's `summary`, `fold_details`, `null_pvalues`, `feature_importance`, and `allocations` artifacts.

## Source artifacts

| Configuration | Output directory | Stamp |
| --- | --- | --- |
| 5d Ridge baseline | `results/cross_asset_ranking_5d_ridge_baseline/` | `20260511T200015Z` |
| 20d Ridge baseline | `results/cross_asset_ranking_20d_ridge_baseline/` | `20260511T201039Z` |
| 5d LambdaRank v3 (regime interactions) | `results/cross_asset_ranking_5d_target_xs_v3_regime_lambdarank/` | `20260511T165440Z` |
| 20d LambdaRank | `results/cross_asset_ranking_20d_target_lambdarank/` | `20260511T193814Z` |

All four runs: 5-fold walk-forward (train=756, val=252, test=252, step=252), 18-asset universe, equal-weight (k=18) benchmark Sharpe = +1.018 across the same calendar. Ridge runs used per-asset train z-score normalization, 39 features, `include_regime_interactions=false`. v3 LambdaRank used 45 features (added `vix_zscore_252d` and five `xs_rank_*_x_vix_z` interactions); 20d LambdaRank used 39 features (no interactions). Ridge alpha was selected via RidgeCV on pool: 5d folds picked α∈{1.0, 1.0, 10.0, 1.0, 10.0}; 20d folds picked α=1.0 in all 5.

## Section 1 — The 2×2 Grid

### Top-1 policy

| Metric | 5d LambdaRank (v3) | 5d Ridge | 20d LambdaRank | 20d Ridge |
| --- | ---: | ---: | ---: | ---: |
| Mean IR vs equal-weight | **+0.137** | −0.906 | **+0.352** | −0.608 |
| Net Sharpe (eq-wt = +1.018) | +0.520 | −0.405 | +0.755 | −0.234 |
| Annualized active return | +0.047 | −0.197 | +0.130 | −0.094 |
| Per-date Spearman (mean) | **+0.0276** | −0.0300 | **+0.0276** | −0.0549 |
| ICIR (mean / sample std n−1) | **+0.788** | −2.235 | +0.397 | −1.579 |
| Drop-best-fold mean IR | −0.091 | −1.255 | +0.120 | −0.880 |
| Folds with +IR | 3/5 | 1/5 | 3/5 | 2/5 |
| Folds passing p ≤ 0.05 | 0/5 | 0/5 | 0/5 | 0/5 |
| Median p-value | 0.289 | 0.655 | 0.220 | 0.796 |
| Max drawdown | −0.283 | −0.262 | −0.213 | −0.126 |
| Turnover / day | 0.306 | 0.258 | 0.077 | 0.045 |
| Cost drag | 0.0180 | 0.0121 | 0.0047 | 0.0024 |
| BTC % of held days | 19.6% | 4.4% | 20.0% | 3.2% |
| Top-3 asset concentration | 41.2% (BTC/UUP/USO) | 49.1% (SHY/UUP/USO) | 48.9% (BTC/SHY/USO) | 77.8% (SHY/UUP/USO) |

### Top-2 policy

| Metric | 5d LambdaRank (v3) | 5d Ridge | 20d LambdaRank | 20d Ridge |
| --- | ---: | ---: | ---: | ---: |
| Mean IR vs equal-weight | **+0.748** | −1.113 | +0.257 | −1.038 |
| Net Sharpe (eq-wt = +1.018) | **+1.180** | −0.446 | +0.827 | −0.162 |
| Annualized active return | +0.102 | −0.168 | +0.048 | −0.173 |
| Per-date Spearman (mean) | +0.0276 | −0.0300 | +0.0276 | −0.0549 |
| ICIR | +0.788 | −2.235 | +0.397 | −1.579 |
| Drop-best-fold mean IR | +0.286 | −1.265 | +0.065 | −1.124 |
| Folds with +IR | 4/5 | 0/5 | 3/5 | 0/5 |
| Folds passing p ≤ 0.05 | 2/5 (p=0.046, 0.002) | 0/5 | 0/5 | 0/5 |
| Median p-value | 0.134 | 0.776 | 0.415 | 0.806 |
| Max drawdown | −0.209 | −0.206 | −0.170 | −0.179 |
| Turnover / day | 0.279 | 0.200 | 0.072 | 0.047 |
| Cost drag | 0.0182 | 0.0095 | 0.0042 | 0.0022 |
| BTC % of held days | 15.8% | 5.0% | 19.2% | 4.4% |
| Top-3 asset concentration | 36.5% (BTC/UUP/DBA) | 42.4% (SHY/UUP/GLD) | 41.6% (BTC/SHY/USO) | 56.3% (SHY/UUP/DBA) |

### Per-fold Spearman (the five values that feed Spearman / ICIR)

| Configuration | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std (n−1) | ICIR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5d LambdaRank v3 | −0.0345 | +0.0470 | +0.0361 | +0.0414 | +0.0482 | +0.0276 | 0.0351 | +0.788 |
| 5d Ridge | −0.0306 | −0.0213 | −0.0521 | −0.0283 | −0.0176 | **−0.0300** | 0.0134 | **−2.235** |
| 20d LambdaRank | −0.0803 | +0.0258 | +0.0435 | +0.1135 | +0.0354 | +0.0276 | 0.0695 | +0.397 |
| 20d Ridge | −0.0884 | −0.0450 | −0.0903 | −0.0427 | −0.0079 | **−0.0549** | 0.0348 | **−1.579** |

### Per-fold IR (top-1)

| Configuration | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5d LambdaRank v3 | −1.601 | +1.049 | +0.543 | −0.092 | +0.785 |
| 5d Ridge | +0.492 | −0.343 | −2.215 | −2.031 | −0.431 |
| 20d LambdaRank | +0.801 | −0.272 | +1.279 | −0.741 | +0.691 |
| 20d Ridge | −1.104 | +0.476 | −0.816 | +0.066 | −1.665 |

### Per-fold p-values (top-1)

| Configuration | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5d LambdaRank v3 | 0.952 | 0.118 | 0.289 | 0.495 | 0.190 |
| 5d Ridge | 0.263 | 0.577 | 0.990 | 0.956 | 0.655 |
| 20d LambdaRank | 0.214 | 0.611 | 0.084 | 0.778 | 0.220 |
| 20d Ridge | 0.816 | 0.317 | 0.796 | 0.471 | 0.968 |

### Per-fold p-values (top-2)

| Configuration | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5d LambdaRank v3 | 0.986 | 0.251 | 0.134 | **0.046** | **0.002** |
| 5d Ridge | 0.665 | 0.776 | 0.966 | 0.936 | 0.709 |
| 20d LambdaRank | 0.206 | 0.641 | 0.531 | 0.415 | 0.174 |
| 20d Ridge | 0.986 | 0.553 | 0.806 | 0.739 | 0.906 |

### Rebalance decisions per fold

Driven by `rebalance_every`: 5d horizon ≈ 252/5 ≈ **50 rebalance opportunities** per fold; 20d horizon ≈ 252/20 ≈ **13 rebalance opportunities** per fold. Actual position-flip counts per fold (from `annualized_turnover`, top-1):

| Configuration | F0 | F1 | F2 | F3 | F4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5d LambdaRank v3 | 65 | 85 | 73 | 81 | 81 |
| 5d Ridge | 45 | 85 | 63 | 55 | 77 |
| 20d LambdaRank | 19 | 23 | 15 | 21 | 19 |
| 20d Ridge | 11 | 15 | 5 | 17 | 9 |

(Flip counts exceed the bar count when both top-1 picks at adjacent rebalances differ; with non-rebalance days holding prior weights, a flip costs once per change.)

Ridge selected α per fold:
- 5d Ridge: {0: 1.0, 1: 1.0, 2: 10.0, 3: 1.0, 4: 10.0}
- 20d Ridge: {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

## Section 2 — Spearman comparison at each horizon

The mean per-date Spearman is the rank-quality summary statistic; it is invariant to allocation policy choice (top-k, rebalance cadence).

### 5d horizon

| Model | Spearman | Lift vs trivial baseline (−0.009) |
| --- | ---: | ---: |
| 5d Ridge | **−0.0300** | **−0.021** (worse than trivial) |
| Trivial baseline (per Phase 1B) | −0.009 | 0 |
| 5d LambdaRank v3 | **+0.0276** | **+0.037** |

**5d gap LambdaRank − Ridge = +0.058** (5.8× the +0.01 "material gap" threshold; 11.6× the +0.005 "close" threshold). Ridge sits below the no-model baseline; LambdaRank sits above it.

### 20d horizon

| Model | Spearman | Lift vs trivial baseline (+0.010) |
| --- | ---: | ---: |
| 20d Ridge | **−0.0549** | **−0.065** (worse than trivial) |
| Trivial baseline (per Phase 1B) | +0.010 | 0 |
| 20d LambdaRank | **+0.0276** | **+0.018** |

**20d gap LambdaRank − Ridge = +0.083** (8.3× the +0.01 "material gap" threshold). 20d Ridge underperforms the trivial baseline by an even larger margin than 5d Ridge does.

Both horizons land Ridge with **strongly negative ICIR** (−2.235 at 5d, −1.579 at 20d) — meaning the negative Spearman is consistent across folds, not noise-driven. Ridge is anti-informative, not just uninformative.

## Section 3 — Feature importance comparison

Ridge feature importance is read as **mean |coef| across the 5 folds** (per-asset z-scored features → magnitudes are comparable across features). LambdaRank importance is read as **total gain across the 5 folds**. The two scales are not directly comparable, so we compare *rank order* and the qualitative *families* of features each model prioritizes.

### 5d horizon top-10

| Rank | 5d Ridge (mean \|coef\|) | 5d LambdaRank v3 (total gain) |
| --- | --- | --- |
| 1 | return_5d (0.103, +++++) | xs_rank_vol_20d (6 652) |
| 2 | price_acceleration (0.092, −−−−−) | realized_vol_20 (4 762) |
| 3 | return_20d (0.052, −−−−−) | relative_vol_ratio (4 682) |
| 4 | asset_return_vs_spy (0.029, mixed) | return_60d (4 560) |
| 5 | relative_strength_vs_benchmark (0.027, mixed) | xs_rank_ret_60d (3 825) |
| 6 | vix_zscore (0.024) | bollinger_band_width_zscore (3 824) |
| 7 | vix_momentum_5d (0.021) | xs_rank_ret_20d (3 599) |
| 8 | vix_return_5d (0.019) | autocorrelation_zscore (3 598) |
| 9 | vix_vol_interaction (0.013) | current_drawdown_60d (3 542) |
| 10 | vix_extreme (0.013) | volatility_regime (3 151) |

**Overlap at 5d**: essentially none in the top 10. Ridge prioritizes **short-horizon momentum / mean-reversion** (return_5d positive, price_acceleration negative, return_20d negative — i.e. it leans long after 5d-up + decelerating + 20d-down) and **VIX level features** (vix_zscore, vix_momentum_5d, vix_extreme). LambdaRank prioritizes **cross-sectional ranks** (xs_rank_vol_20d, xs_rank_ret_60d, xs_rank_ret_20d), **volatility shape** (realized_vol_20, relative_vol_ratio, bollinger_band_width_zscore, volatility_regime), and **longer-horizon return + drawdown** (return_60d, current_drawdown_60d). Common-name features that appear in both top-10s: **none**.

No 5d Ridge feature has near-zero |coef| (<0.001 mean) — the linear model spreads weight across the full feature set, with only the top three carrying material magnitude.

### 20d horizon top-10

| Rank | 20d Ridge (mean \|coef\|) | 20d LambdaRank (total gain) |
| --- | --- | --- |
| 1 | return_5d (0.263, +++++) | xs_rank_vol_20d (21 044) |
| 2 | price_acceleration (0.233, −−−−−) | realized_vol_20 (8 613) |
| 3 | relative_strength_vs_benchmark (0.197, +++++) | xs_rank_ret_60d (7 578) |
| 4 | asset_return_vs_spy (0.193, −−−−−) | relative_vol_ratio (6 936) |
| 5 | return_20d (0.130, −−−−−) | return_60d (6 602) |
| 6 | vix_return_5d (0.084, −−−−−) | current_drawdown_60d (5 386) |
| 7 | vix_momentum_5d (0.075, +++++) | vix_relative (5 011) |
| 8 | vix_extreme (0.047, −−−−−) | close_to_open_gap (4 671) |
| 9 | current_drawdown_60d (0.035, +++++) | bollinger_band_width_zscore (4 538) |
| 10 | realized_vol_20 (0.034, +++++) | xs_rank_ret_20d (4 395) |

**Overlap at 20d**: `realized_vol_20` and `current_drawdown_60d` appear in both top-10s. Otherwise the model families diverge in the same direction as at 5d: Ridge keeps weight on short/medium-horizon **per-asset returns and momentum** (return_5d, price_acceleration, relative_strength_vs_benchmark, asset_return_vs_spy, return_20d) plus **VIX level / shape** (vix_return_5d, vix_momentum_5d, vix_extreme); LambdaRank again leans on **cross-sectional ranks** (xs_rank_vol_20d, xs_rank_ret_60d, xs_rank_ret_20d) and **vol shape** features. None of the `xs_rank_*` features land in either Ridge top-10, despite being present in both Ridge feature sets.

Diagnostic answer: **Ridge and LambdaRank do not prioritize the same features**. Ridge cannot extract usable signal from cross-sectional rank features (their relationship to forward risk-adjusted return appears to be non-monotone or rank-conditional in a way the linear model cannot capture) and instead falls back on naive momentum + VIX level signals — which are systematically wrong-signed enough at both horizons to produce negative Spearman. LambdaRank's gain concentrates on `xs_rank_*` and volatility-shape features, indicating it is finding **non-linear or rank-based structure** that linear coefficients miss.

## Section 4 — Interpretation against pre-committed criteria

Applying the four pre-committed cells from the Ridge baseline prompt as written:

| Cell | Trigger | Triggered? |
| --- | --- | --- |
| Ridge ≈ LambdaRank (Spearman within ~0.005 at same horizon) | feature engineering is the whole story; GBDT is decoration | **No** — gap is +0.058 (5d) and +0.083 (20d), both ≫ 0.005 |
| Ridge materially below LambdaRank (gap > 0.01) | GBDT non-linearity is doing real work; deployment should use LambdaRank | **YES** — gap exceeds 0.01 by ~6× at 5d and ~8× at 20d |
| Ridge materially above LambdaRank (unlikely) | LambdaRank is overfitting; Ridge preferred | **No** — Ridge underperforms at both horizons |
| Both in same modest zone with Ridge slightly below | confirms data-vs-model ceiling | **No** — Ridge is not in the same zone; it is below the trivial baseline at both horizons |

**Matching cell: "Ridge materially below LambdaRank (gap > 0.01) — GBDT non-linearity is doing real work; deployment should use LambdaRank, not Ridge."**

Reasoning, restricted to the pre-committed criteria:
1. Spearman gap is +0.058 (5d) and +0.083 (20d), both above the +0.01 "material" threshold with no ambiguity.
2. ICIR is negative for Ridge at both horizons (−2.235, −1.579), meaning the underperformance is structural, not fold-driven noise.
3. Feature-importance comparison (Section 3) corroborates the mechanism: the non-linear/rank-based signal that LambdaRank extracts from `xs_rank_*` and volatility-shape features is the alpha; Ridge cannot represent it and instead loads onto naively-signed momentum.

The interpretation answers the modeling question the Ridge baseline was designed to answer: **the model family choice matters; GBDT non-linearity is load-bearing for whatever signal is present.** This is independent of whether the LambdaRank signal itself is strong enough to deploy.

## Section 5 — Final recommendation

The pre-committed conclusion of the immediately prior decision document — `docs/PATCH_CROSS_ASSET_RANKING_20D_HORIZON.md` — already issued a **wind-down verdict** for the cross-asset ranking campaign, on the basis that:

- Pre-committed decision-grade Spearman threshold: **+0.05** (with ICIR ≥ 0.5). Best observed across seven configurations: +0.032 (v1) down to +0.006 (architectural pivot).
- Pre-committed directional Spearman threshold: **+0.03** (with ICIR ≥ 0.3). 20d LambdaRank landed at +0.0276 — fails by 0.002.
- Per-fold p-value test does not pass for any decision-grade configuration's top-1 policy.

The Ridge baseline was the eighth configuration. Both Ridge horizons land **below the trivial baseline**, with strongly negative ICIR. Ridge therefore **does not reverse** the wind-down verdict; it confirms it by ruling out the "linear model + good features is enough" branch.

### Candidate-profile evaluation (against the 5 pre-committed profiles)

| Profile | Mean IR | ICIR | Spearman | Folds +IR | Folds p ≤ 0.05 | Drop-best IR | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `v3_5d_top2` | **+0.748** | +0.788 | +0.0276 | 4/5 | 2/5 (p=0.046, 0.002) | +0.286 | Best LambdaRank cell; still misses pre-committed +0.03 Spearman / ICIR ≥ 0.5; IR concentrated in fold 4 (IR +2.598). |
| `20d_top1` | +0.352 | +0.397 | +0.0276 | 3/5 | 0/5 | +0.120 | Misses Spearman by 0.002; lowest turnover (7.7%/day); 0/5 folds significant at p ≤ 0.05 (but per the 20d patch's pre-committed statistical-power note, per-fold p was de-emphasized at the 20d horizon). |
| `20d_top2` | +0.257 | +0.397 | +0.0276 | 3/5 | 0/5 | +0.065 | Same Spearman/ICIR as 20d_top1; lower IR; drop-best collapses to near zero. |
| `ridge_5d_top2` | **−1.113** | **−2.235** | **−0.0300** | 0/5 | 0/5 | −1.265 | Worse than equal-weight. Eliminated. |
| `ridge_20d_top1` | **−0.608** | **−1.579** | **−0.0549** | 2/5 | 0/5 | −0.880 | Worse than equal-weight on Sharpe and IR; Spearman below trivial baseline. Eliminated. |

### Recommendation

1. **Deployment candidate**: None of the five pre-committed profiles clears the pre-committed Spearman / ICIR thresholds. Both Ridge profiles fail outright. Of the LambdaRank profiles, `v3_5d_top2` has the strongest economic signature (net Sharpe +1.180 vs eq-wt +1.018, drop-best IR still positive at +0.286, 2/5 folds p ≤ 0.05) but **fails** the +0.03 directional Spearman threshold by 0.002 and the +0.05 decision-grade threshold by 0.022.

2. **Proceed to forward-walk + live prediction, or wind down?** **Wind down.** Per the pre-committed thresholds documented in `PATCH_CROSS_ASSET_RANKING_20D_HORIZON.md`, no configuration crosses the directional bar. Ridge's failure is informative (it confirms the gap is non-linearity, not features) but does not produce a new candidate above the bar. Restating that patch's verdict: eight configurations have now produced Spearman in the narrow band [−0.055, +0.032] with the upper end set by v1 LambdaRank and the lower end now set by 20d Ridge. The pre-committed +0.05 decision-grade threshold sits above the empirical ceiling of this universe under any configuration tested.

3. **If proceeding anyway** (against the pre-committed verdict — flagged as a goalpost move): the only defensible choice is **`v3_5d_top2`**, on the basis of best economic metrics (Sharpe +1.180, IR +0.748, ICIR +0.788, drop-best IR +0.286, 4/5 folds positive IR, 2/5 folds at p ≤ 0.05). `20d_top1` is the conservative fallback if low turnover and risk-budget capacity are the dominant constraints — but its IR sits at +0.352 with ICIR +0.397 and 0/5 folds at p ≤ 0.05, so it is the lower-conviction choice.

**Honest combined verdict**, following the "no goalpost moves" rule from the prior patch chain: **wind down the cross-asset ranking project.** The Ridge baseline did the job it was scoped to do (rule out one explanation for the ceiling) and confirmed the ceiling.

## Addendum — mechanism refinement and deferred follow-up

Section 4 originally framed Ridge's underperformance as evidence that "GBDT non-linearity is doing real work." That framing is partially right but mis-locates the dominant mechanism. The more likely explanation is **architectural, not non-linearity**:

- LambdaRank's loss is **grouped per date**: pairwise rank comparisons are constrained within the 18 assets on each test date and never cross dates. The model is solving an *intra-date* ranking problem.
- Pooled Ridge sees **all (date, asset) rows as one regression**. Combined with **per-asset train z-score normalization** (each feature standardized against that asset's own history), a value of +2σ on `return_5d` for BTC and +2σ on `return_5d` for SHY are pooled as if equivalent — but they carry completely different cross-sectional information at any given date.
- The fitted Ridge coefficients confirm this is the failure mode: across both horizons, signs on the top per-asset-normalized features are economically incoherent (e.g. 5d Ridge fits `return_5d` positively and `return_20d` negatively across all 5 folds; 20d Ridge fits `relative_strength_vs_benchmark` positively and `asset_return_vs_spy` negatively across all 5 folds — these are near-duplicates of each other in the cross-section). The result is sign-inverted ranking on the test set, producing negative Spearman with high consistency (ICIR −2.2 / −1.6).
- This re-frames the modeling lesson generically: **respect per-date ranking structure**. If pooling across dates, features must be cross-sectionally comparable at each date (rank-based or cross-sectional z-score), not per-asset history-normalized.

**Deferred follow-up (out of scope for this campaign):** Ridge fit on **only the five `xs_rank_*` features** (which are cross-sectionally comparable by construction), dropping the 34 per-asset z-scored features. Outcomes:
- If that Ridge variant matches LambdaRank's Spearman → the story is "features are the signal, model family is decoration **when features are cross-sectionally comparable**." GBDT non-linearity is not load-bearing.
- If it still underperforms → LambdaRank's grouping + tree structure is doing additional work beyond feature engineering.

Either result is useful for **future cross-sectional ranking projects on different universes / horizons** — but is not a reason to extend this campaign, whose pre-committed verdict is already wind-down. Noted here so it isn't re-derived from scratch next time.
