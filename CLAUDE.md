# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & commands

- Python is pinned to 3.12 (`.python-version`); the project is managed with `uv`.
- `uv sync` — install/refresh dependencies from `pyproject.toml` / `uv.lock`.
- `uv run python -m pytest` — run the full test suite.
- `uv run python -m pytest tests/test_walk_forward.py::test_<name>` — run a single test.
- `uv run python -m pytest -k <pattern>` — run by name pattern across files.
- Most current research entry points are the standalone runners under `scripts/` (see "Two parallel pipelines" below). They follow a shared `--dry-run` / `--execute` discipline; always validate with `--dry-run` before `--execute`.

Examples of the standalone CLIs (each takes `--input-csv` of a prepared feature frame and writes to `--output-dir`):

- `scripts/run_drawdown_label_viability_report.py`
- `scripts/run_drawdown_risk_classifier_experiment.py`
- `scripts/run_drawdown_risk_calibration_experiment.py`
- `scripts/run_drawdown_classifier_calibration_diagnostics.py`
- `scripts/run_regime_overlay_experiment.py`
- `scripts/run_vol_quantile_overlay_experiment.py`
- `scripts/generate_matched_null_report_from_oof_artifacts.py` — runs purely on saved CSVs, no data downloads.

## Repository state — important context

This repo is in the middle of a **first-principles reset** (see `docs/FIRST_PRINCIPLES_RESET_PLAN.md` and the `docs/PATCH_*` series). The reset reframed the project away from "daily SPY directional alpha" toward drawdown-risk classification and regime/volatility overlays.

Practical implications for any work done here:

- **`main.py` and `audit/integrity_audit.py` reference modules that do not currently exist on disk** — `data.market_data`, `models.*`, `models.ensemble`, `models.registry`, etc. The legacy "model zoo" CLI in `main.py` therefore cannot be run as-is. Treat `main.py`, `utils/experiment.py`, and `audit/integrity_audit.py` as the frozen *legacy* path, kept for reference and for the `champion_v1.0` manifest. Do not assume importing from them works.
- The **active** workflow lives in `experiments/` (library code) and `scripts/` (thin CLIs) and operates on **prepared feature CSVs**, not on yfinance downloads inside the process. This was a deliberate split so the new research track can evolve without touching the old model-zoo CLI.
- `champions/current_champion_manifest.yaml` documents the frozen `regime_stacking_ensemble_regime` (Champion v1.0). Per the reset plan it is **not** a validated alpha model — it harvested SPY beta via near-always-long exposure. Don't treat it as a production candidate or a benchmark to beat in isolation; the gates require beating *matched* nulls and a vol-targeted baseline.

## Architecture (active research track)

Daily flow for the current drawdown / overlay research:

1. **Prepared feature frame** (CSV) is the input. Required columns include `date`, `Adj Close`, `forward_simple_return_1d`, `benchmark_return_1d`, regime features when applicable, and either an existing target or raw prices to derive one.
2. **Drawdown labels**: `evaluation/drawdown_labels.py` derives `target_drawdown_event_<H>d_<T>pct` columns via `append_drawdown_label_grid` from forward max-drawdown over horizon `H` and threshold `T`. Naming is stable and load-bearing — downstream code matches column prefixes.
3. **Walk-forward splits**: `evaluation/walk_forward.generate_walk_forward_splits` produces strictly time-ordered, non-overlapping `(train, validation, test)` `WalkForwardSplit`s. Defaults: `train=756, val=252, test=252, step=252`. **All** new evaluation code must split this way — no shuffled folds, no pooled future data.
4. **Per-split model fit**: classifiers in `evaluation/drawdown_classification.py` (logistic, regularized linear) are fit on `train`, tuned/checked on `validation`, scored on `test`. Calibration (`platt`, `isotonic`, `identity`) lives in `evaluation/probability_calibration.py` and is fold-safe (fit on validation, applied on test).
5. **OOF artifact assembly**: every test split produces a row-aligned artifact frame via `evaluation/audit_artifacts.build_standard_audit_artifact_frame`. This frame is the canonical hand-off between experiments and downstream audit/null tooling — see `STANDARD_AUDIT_COLUMNS`. It intentionally keeps both new column names (`raw_signal`, `executed_position`, `strategy_net_return`) **and** legacy aliases (`prediction`, `forward_simple_return_1d`, etc.) so artifacts remain compatible with `audit/integrity_audit.py` and `scripts/generate_matched_null_report_from_oof_artifacts.py`.
6. **Matched-null tests**: `evaluation/null_baselines.py` and `evaluation/regime_overlay.py` / `evaluation/volatility_overlay.py` provide the matched-exposure null distributions. Strategies are tested by comparing to nulls that preserve key exposure properties — beating "no signal" is not enough.
7. **Calibration diagnostics**: `evaluation/calibration_diagnostics.py` computes Brier, ECE, reliability vs. an `event_rate` constant baseline. Per Patch 38, calibration must beat the constant baseline on Brier; "improved versus raw classifier" alone is not a passing gate.

Key library modules:

- `evaluation/metrics.py` — Sharpe/Sortino/Calmar, IR, active-return diagnostics, turnover, transaction-cost accounting. Uses signal-shifted-by-one-bar execution semantics; transaction cost is applied on position flips.
- `evaluation/walk_forward.py` — single source of truth for split construction.
- `evaluation/audit_artifacts.py` — single source of truth for OOF artifact schema.
- `experiments/` — pure functions that take splits + frames and return result dataclasses (`*_ExperimentResult`). No file IO, no argparse.
- `scripts/` — argparse + IO wrappers around `experiments/`. They handle `--dry-run`, manifest writing, markdown report generation.
- `regime/regime_detection.py`, `features/regime_features.py` — HMM/GMM regime fitting and the regime-derived feature columns (`regime_id`, `regime_prob_*`, interactions). HMM seeds and inference column lists are pinned in the champion manifest.

## Conventions

- **Leakage discipline is non-negotiable.** Features must be lagged/contemporaneous only; targets are forward-looking. Walk-forward splits are strictly time-ordered. Scalers and calibrators are fit on train (or validation for calibration) only — never on test, never pooled.
- **Raw prices are not features.** `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `BenchmarkClose`, `VIXClose` are excluded from the feature matrix; see `RAW_PRICE_COLUMNS` in `audit/integrity_audit.py` and `infer_feature_columns` in `scripts/run_drawdown_risk_classifier_experiment.py`.
- **Reproducibility**: seed is `42` everywhere (numpy, python-random, torch, HMM, GMM). `utils/reproducibility.seed_everything` is the entry point. New randomized code should accept and thread a `seed`.
- **Run purposes**: scripts distinguish `plumbing`, `diagnostic`, and `decision_grade` runs (`--run-purpose` / `--decision-grade`). Decision-grade runs are the only ones that should be cited as gate evidence; the others exist for wiring/sanity checks.
- **Style**: 4-space indent, `snake_case` functions/vars, `PascalCase` classes, type hints on public functions. Comments are reserved for quant-specific subtleties (leakage, signal construction); avoid restating what the code does.
- **Tests**: `pytest` with `test_<module>.py` / `test_<behavior>` names. Prefer deterministic fixtures and shape/leakage-contract assertions over performance thresholds.
- **Don't commit** `.venv/`, cached market data, generated plots, model artifacts, large datasets, or anything under `results/` (gitignored).

## Docs to read before non-trivial changes

The `docs/PATCH_*` markdown files are a chronological decision log — they record what was tried, what failed, and why. Read the most recent few before proposing a new direction; previous patches frequently invalidate seemingly-reasonable approaches (e.g., HMM hard-veto overlay, simple vol-quantile overlay, raw daily-direction targets are all explicitly retired).
