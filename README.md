# ml-trading-signal-exploration

**Cross-asset ranking system for personal-investor deployment.** Picks 1–2 ETFs to hold over a 5- or 20-trading-day horizon from an 18-asset universe (major equity indices, bonds, commodities, currency, real estate, BTC), using LambdaRank on a Gradient-Boosted Decision Tree backbone (LightGBM).

## Status

The cross-asset ranking research campaign closed in May 2026 after testing 8+ configurations against pre-committed thresholds. Empirical Spearman ceiling for this universe at 5–20d horizons is ~0.032 — below the +0.05 decision-grade threshold. Three LambdaRank profiles emerged as deployable; Ridge baselines were tested and rejected ([RIDGE_BASELINE_RESULTS.md](docs/RIDGE_BASELINE_RESULTS.md)).

Deployment infrastructure is built and validated. Current state: ready for real-money paper-walk / small-capital deployment.

## Documentation

Start here if you're new (or returning after a break):

| Doc | Purpose |
| --- | --- |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | First-prediction orientation. Under 10 minutes from clone to a real prediction. |
| [docs/OPERATIONAL_RUNBOOK.md](docs/OPERATIONAL_RUNBOOK.md) | Day-to-day procedure: rebalance cadence, the 5-step rebalance flow, calendar planning, monthly / quarterly / annual reviews, anti-patterns. |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Symptom → Diagnosis → Fix for the 8 most common operational failures. |
| [docs/TECHNICAL_DESCRIPTION.md](docs/TECHNICAL_DESCRIPTION.md) | Algorithm, feature engineering, walk-forward CV, evaluation metrics, empirical universe ceiling. |

Deployment-infrastructure design docs:

- [docs/PATCH_LIVE_PREDICTION_SCRIPT.md](docs/PATCH_LIVE_PREDICTION_SCRIPT.md) — architecture, profile definitions, causal-data guards, validation evidence (100% per-date top-1 match against the campaign runner)
- [docs/PATCH_OPERATIONAL_WRAPPER.md](docs/PATCH_OPERATIONAL_WRAPPER.md) — cache-refresh wrapper, error-handling matrix, smoke-test evidence

Research audit trail:

- [docs/RIDGE_BASELINE_RESULTS.md](docs/RIDGE_BASELINE_RESULTS.md) — 2×2 LambdaRank-vs-Ridge comparison, mechanism diagnosis, wind-down verdict
- `docs/PATCH_CROSS_ASSET_RANKING_*.md` — chronological campaign decision log (feasibility → cross-sectional pivot v1/v2/v3 → regime pivot → 20d horizon → close-out)
- `docs/FIRST_PRINCIPLES_RESET_PLAN.md` and earlier `docs/PATCH_*.md` — pre-pivot drawdown / regime / vol-overlay experiments (frozen)

## Quick start

```bash
# 1. Install / sync dependencies (Python 3.12, managed by uv)
uv sync

# 2. Refresh caches and run a prediction for the cost-efficient profile
uv run python -m scripts.run_operational_prediction \
    --profile 20d_top1 \
    --output-dir predictions/

# 3. Read the operational runbook before placing your first real trade
open docs/OPERATIONAL_RUNBOOK.md
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for the full first-prediction walkthrough.

## The four deployable profiles

LambdaRank only. Per-date grouping is load-bearing; pooled regression models on per-asset-normalized features fail empirically.

| Profile | Horizon | Top-k | Notes |
| --- | ---: | ---: | --- |
| `v3_5d_top2` | 5 trading days | 2 | Economic high-water mark (backtest Sharpe ~1.18, IR +0.748). Highest turnover. |
| `20d_top1` | 20 trading days | 1 | Cost-efficient alternative. Lowest turnover, positive drop-best-fold IR. |
| `20d_top2` | 20 trading days | 2 | Concentration hedge of `20d_top1`. |
| `v1_5d_top2` | 5 trading days | 2 | Rank-quality reference. Retained for validation, **not for deployment.** |

## Structure

Active deployment path:

- `scripts/run_operational_prediction.py` — wrapper: refresh 19 caches via yfinance, then invoke the live prediction
- `scripts/run_live_prediction.py` — single-date or replay prediction; pure orchestration over existing pipeline functions
- `scripts/run_cross_asset_ranking_experiment.py` — campaign experiment runner (used for retraining / new diagnostic runs)
- `scripts/prepare_feature_frame.py` — per-asset feature panel builder (yfinance + feature engineering + cache)

Active library code:

- `experiments/cross_asset_ranking_experiment.py` — pure-logic ranking experiment (no I/O, no argparse)
- `evaluation/cross_asset_ranking.py` — feature engineering, normalization, ranking, allocation, null baselines
- `evaluation/walk_forward.py` — strictly time-ordered split construction (single source of truth)
- `evaluation/metrics.py` — Sharpe / Sortino / Calmar / IR / turnover / cost accounting
- `features/`, `regime/` — feature builders and regime-detection primitives

Data and outputs:

- `data/multi_asset_cache/` — raw OHLCV cache for the 18-asset universe + VIX (refreshed by the wrapper)
- `predictions/` — per-prediction JSON records + append-only `predictions_log.jsonl` + `operational_log.jsonl`
- `results/` — campaign experiment outputs (gitignored)
- `champions/current_champion_manifest.yaml` — frozen Champion v1.0 manifest (legacy reference only; not a production candidate per the reset plan)

**Frozen legacy paths** (do not invoke; kept for reference):

- `main.py`, `utils/experiment.py`, `audit/integrity_audit.py` — pre-reset model-zoo CLI, references modules that no longer exist on disk
- `models/` (referenced by legacy CLI; absent from working tree)

## Commands

```bash
uv sync                                                 # install / refresh deps
uv run python -m pytest                                 # full test suite
uv run python -m pytest tests/test_walk_forward.py      # single file
uv run python -m pytest -k <pattern>                    # name pattern
```

Research entry points (standalone CLIs, all support `--dry-run` / `--execute`):

```bash
uv run python -m scripts.run_cross_asset_ranking_experiment --dry-run [...]
uv run python -m scripts.run_drawdown_label_viability_report --dry-run [...]
uv run python -m scripts.run_drawdown_risk_classifier_experiment --dry-run [...]
uv run python -m scripts.run_regime_overlay_experiment --dry-run [...]
uv run python -m scripts.run_vol_quantile_overlay_experiment --dry-run [...]
uv run python -m scripts.generate_matched_null_report_from_oof_artifacts --dry-run [...]
```

Validate with `--dry-run` before `--execute`. See each script's `--help` for flags.

## Financial ML safeguards

- No raw prices are used as features or targets (raw OHLCV excluded by `select_cross_asset_feature_columns`).
- Features are lagged or contemporaneous only; targets are forward-looking and used only for training.
- Validation is strictly walk-forward, with no shuffled rows or pooled future data.
- Per-asset z-score normalization fits mean/std on the training window only; test rows never touch the fitting statistics. Calibrators fit on validation, applied on test.
- Live-mode prediction enforces no-look-ahead via frame truncation to `<= as_of_date` before panel construction, so forward targets at the live date are NaN by construction.
- Seed is `42` everywhere (numpy, python-random, torch, HMM, GMM) via `utils.reproducibility.seed_everything`.

For the leakage discipline as enforced in code, see [docs/TECHNICAL_DESCRIPTION.md](docs/TECHNICAL_DESCRIPTION.md) and [docs/PATCH_LIVE_PREDICTION_SCRIPT.md](docs/PATCH_LIVE_PREDICTION_SCRIPT.md) (risk surface table).

## Project conventions

- Python 3.12 pinned via `.python-version`, dependencies managed by `uv` (`pyproject.toml` / `uv.lock`).
- Scripts use `--dry-run` / `--execute` discipline; always validate with `--dry-run` before committing to writes.
- Run purposes: `plumbing`, `diagnostic`, `decision_grade`. Only decision-grade runs are cited as gate evidence.
- Style: 4-space indent, `snake_case`, `PascalCase` for classes, type hints on public functions, comments reserved for quant-specific subtleties.
- Tests: `pytest`, deterministic fixtures, shape/leakage-contract assertions over performance thresholds.
- Don't commit: `.venv/`, cached market data, generated plots, model artifacts, anything under `results/` (gitignored).

See `CLAUDE.md` for the full set of conventions and the repository-state context that shapes how to work in here.
