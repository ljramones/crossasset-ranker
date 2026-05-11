"""Generate a matched-null markdown report from saved OOF artifacts only.

This helper intentionally avoids any experiment preparation, model evaluation,
or data downloads. It reads existing CSV artifacts, resolves the required
columns defensively, reconstructs split-local executed positions when needed,
and runs matched-null diagnostics on the saved outputs only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.null_baselines import run_matched_null_suite


DATE_CANDIDATES = ["date", "Date", "index"]
SPLIT_CANDIDATES = ["split_id", "split", "fold", "fold_id"]
ASSET_RETURN_CANDIDATES = [
    "asset_return",
    "forward_simple_return_1d",
    "return_1d",
    "asset_return",
    "asset_returns",
    "returns",
    "simple_return_1d",
]
BENCHMARK_RETURN_CANDIDATES = ["benchmark_return_1d", "benchmark_return", "benchmark_returns"]
EXECUTED_POSITION_CANDIDATES = ["executed_position", "position", "final_position"]
PREDICTION_CANDIDATES = ["raw_signal", "signal", "prediction", "pred", "y_pred"]
PROBABILITY_CANDIDATES = ["prediction_probability", "probability", "probability__ensemble", "meta_probability"]
REGIME_CANDIDATES = ["regime_id", "regime", "hmm_regime"]
MODEL_CANDIDATES = ["model_name", "model", "artifact_name"]


@dataclass(slots=True)
class ArtifactColumnResolution:
    date_col: str | None
    split_col: str | None
    asset_return_col: str | None
    benchmark_return_col: str | None
    executed_position_col: str | None
    prediction_col: str | None
    probability_col: str | None
    regime_col: str | None
    model_col: str | None

    @property
    def can_evaluate(self) -> bool:
        return (
            self.asset_return_col is not None
            and self.benchmark_return_col is not None
            and (self.executed_position_col is not None or self.prediction_col is not None)
        )


@dataclass(slots=True)
class ArtifactAuditResult:
    artifact_path: Path
    artifact_label: str
    columns: ArtifactColumnResolution
    executed_position_source: str | None
    canonical_metrics: dict[str, float] | None
    null_summaries: dict[str, Any] | None
    notes: list[str]


def _resolve_first_present(columns: list[str], candidates: list[str]) -> str | None:
    available = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = available.get(candidate.lower())
        if match is not None:
            return match
    return None


def inspect_oof_artifact_columns(frame: pd.DataFrame) -> ArtifactColumnResolution:
    """Resolve relevant columns from a saved OOF artifact."""

    columns = list(frame.columns)
    return ArtifactColumnResolution(
        date_col=_resolve_first_present(columns, DATE_CANDIDATES),
        split_col=_resolve_first_present(columns, SPLIT_CANDIDATES),
        asset_return_col=_resolve_first_present(columns, ASSET_RETURN_CANDIDATES),
        benchmark_return_col=_resolve_first_present(columns, BENCHMARK_RETURN_CANDIDATES),
        executed_position_col=_resolve_first_present(columns, EXECUTED_POSITION_CANDIDATES),
        prediction_col=_resolve_first_present(columns, PREDICTION_CANDIDATES),
        probability_col=_resolve_first_present(columns, PROBABILITY_CANDIDATES),
        regime_col=_resolve_first_present(columns, REGIME_CANDIDATES),
        model_col=_resolve_first_present(columns, MODEL_CANDIDATES),
    )


def _artifact_label(path: Path, frame: pd.DataFrame, resolution: ArtifactColumnResolution) -> str:
    if resolution.model_col is not None and resolution.model_col in frame.columns:
        non_null = frame[resolution.model_col].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])
    return path.stem


def reconstruct_executed_positions(frame: pd.DataFrame, resolution: ArtifactColumnResolution) -> tuple[pd.Series, str]:
    """Return executed positions, reconstructing split-locally from predictions if needed."""

    if resolution.executed_position_col is not None:
        return frame[resolution.executed_position_col].astype(float).copy(), resolution.executed_position_col

    if resolution.prediction_col is None:
        raise ValueError("Artifact is missing both executed position and prediction columns.")

    prediction = frame[resolution.prediction_col].astype(float)
    if resolution.split_col is not None:
        executed = prediction.groupby(frame[resolution.split_col], sort=False).shift(1).fillna(0.0)
        return executed.astype(float), f"{resolution.prediction_col} (split-local shift)"
    return prediction.shift(1).fillna(0.0).astype(float), f"{resolution.prediction_col} (global shift)"


def evaluate_oof_artifact(
    artifact_path: Path,
    *,
    n_runs: int = 100,
    seed: int = 42,
    transaction_cost_bps: float = 2.0,
    decision_metric: str = "information_ratio",
) -> ArtifactAuditResult:
    """Evaluate one saved OOF artifact using matched-null diagnostics only."""

    frame = pd.read_csv(artifact_path)
    resolution = inspect_oof_artifact_columns(frame)
    notes: list[str] = []
    label = _artifact_label(artifact_path, frame, resolution)
    if not resolution.can_evaluate:
        notes.append("Artifact skipped: missing required return and/or prediction/position columns.")
        return ArtifactAuditResult(
            artifact_path=artifact_path,
            artifact_label=label,
            columns=resolution,
            executed_position_source=None,
            canonical_metrics=None,
            null_summaries=None,
            notes=notes,
        )

    executed_position, position_source = reconstruct_executed_positions(frame, resolution)
    asset_returns = frame[resolution.asset_return_col].astype(float)
    benchmark_returns = frame[resolution.benchmark_return_col].astype(float)
    regime_labels = frame[resolution.regime_col] if resolution.regime_col is not None else None
    suite = run_matched_null_suite(
        positions=executed_position,
        returns=asset_returns,
        benchmark_returns=benchmark_returns,
        regime_labels=regime_labels,
        n_runs=n_runs,
        seed=seed,
        transaction_cost_bps=transaction_cost_bps,
        decision_metric=decision_metric,
        include_block_bootstrap=True,
    )
    if regime_labels is None:
        notes.append("No regime labels found; same-regime-exposure null was omitted.")
    if resolution.executed_position_col is None:
        notes.append("Executed positions were reconstructed from predictions with a split-local shift.")
    else:
        notes.append("Artifact supplied executed positions directly; no additional shift was applied.")
    return ArtifactAuditResult(
        artifact_path=artifact_path,
        artifact_label=label,
        columns=resolution,
        executed_position_source=position_source,
        canonical_metrics=suite["canonical_metrics"],
        null_summaries=suite["null_summaries"],
        notes=notes,
    )


def build_markdown_report(
    results: list[ArtifactAuditResult],
    *,
    generated_at: str,
    n_runs: int,
    seed: int,
    transaction_cost_bps: float,
    decision_metric: str,
) -> str:
    """Render the partial artifact-only matched-null report."""

    lines = [
        "# Partial Artifact-Only Matched-Null Audit Report",
        "",
        f"- Generated at: `{generated_at}`",
        "- Scope: saved OOF artifacts only",
        "- This report does not retrain models, rebuild experiments, rerun walk-forward evaluation, or download data.",
        "- Coverage: partial only; it does not replace the full comparative audit workflow.",
        f"- Null runs per artifact: `{n_runs}`",
        f"- Random seed: `{seed}`",
        f"- Transaction cost bps: `{transaction_cost_bps}`",
        f"- Decision metric: `{decision_metric}`",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result.artifact_label}",
                "",
                f"- Artifact: `{result.artifact_path}`",
                f"- Executed position source: `{result.executed_position_source or 'unavailable'}`",
                "",
                "### Column Inspection",
                "",
                "| Field | Resolved Column |",
                "| --- | --- |",
                f"| date | {result.columns.date_col or 'N/A'} |",
                f"| split | {result.columns.split_col or 'N/A'} |",
                f"| asset returns | {result.columns.asset_return_col or 'N/A'} |",
                f"| benchmark returns | {result.columns.benchmark_return_col or 'N/A'} |",
                f"| executed position | {result.columns.executed_position_col or 'N/A'} |",
                f"| prediction | {result.columns.prediction_col or 'N/A'} |",
                f"| probability | {result.columns.probability_col or 'N/A'} |",
                f"| regime | {result.columns.regime_col or 'N/A'} |",
                "",
            ]
        )
        if result.canonical_metrics is None or result.null_summaries is None:
            lines.extend(
                [
                    "### Status",
                    "",
                    "- Artifact could not be evaluated from saved fields alone.",
                    *[f"- {note}" for note in result.notes],
                    "",
                ]
            )
            continue
        canonical = result.canonical_metrics
        lines.extend(
            [
                "### Canonical Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| information_ratio | {canonical.get('information_ratio', 0.0):.6f} |",
                f"| annualized_active_return | {canonical.get('annualized_active_return', 0.0):.6f} |",
                f"| active_calmar | {canonical.get('active_calmar', 0.0):.6f} |",
                f"| fraction_in_market | {canonical.get('fraction_in_market', 0.0):.6f} |",
                f"| daily_turnover | {canonical.get('daily_turnover', 0.0):.6f} |",
                f"| position_flip_count | {canonical.get('position_flip_count', 0.0):.0f} |",
                "",
                "### Matched Null Diagnostics",
                "",
                "| Null Baseline | Mean Null IR | 95th Percentile Null IR | p-value | Decision |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for name, payload in result.null_summaries.items():
            summary = payload["summary"]
            decision = "PASS" if summary.canonical_value > summary.percentile_95_null_value and summary.p_value < 0.05 else "FAIL"
            lines.append(
                "| "
                f"{name} | {summary.mean_null_value:.6f} | {summary.percentile_95_null_value:.6f} | "
                f"{summary.p_value:.6f} | {decision} |"
            )
        lines.extend(
            [
                "",
                "### Notes",
                "",
                *[f"- {note}" for note in result.notes],
                "",
            ]
        )
    lines.extend(
        [
            "## Verdict",
            "",
            "- This is a partial artifact-only audit.",
            "- It covers only artifacts that expose enough saved information to reconstruct executed positions and benchmark-relative active returns safely.",
            "- It does not cover models whose saved artifacts include only per-split strategy returns and equity curves.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(
    artifact_paths: list[Path],
    *,
    output_dir: Path,
    n_runs: int,
    seed: int,
    transaction_cost_bps: float,
    decision_metric: str,
) -> Path:
    """Generate a new timestamped artifact-only matched-null audit report."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"matched_null_artifact_audit_report_{timestamp}.md"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")
    results = [
        evaluate_oof_artifact(
            path,
            n_runs=n_runs,
            seed=seed,
            transaction_cost_bps=transaction_cost_bps,
            decision_metric=decision_metric,
        )
        for path in artifact_paths
    ]
    markdown = build_markdown_report(
        results,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        n_runs=n_runs,
        seed=seed,
        transaction_cost_bps=transaction_cost_bps,
        decision_metric=decision_metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a matched-null report from saved OOF artifacts only.")
    parser.add_argument(
        "--artifacts",
        nargs="+",
        required=True,
        help="One or more saved OOF CSV artifact paths.",
    )
    parser.add_argument("--output-dir", default="results", help="Directory for the new timestamped markdown report.")
    parser.add_argument("--n-runs", type=int, default=25, help="Monte Carlo null runs per artifact.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for null generators.")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Transaction cost applied to executed positions.")
    parser.add_argument("--decision-metric", default="information_ratio", help="Decision metric for null summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_paths = [Path(value) for value in args.artifacts]
    output_path = generate_report(
        artifact_paths,
        output_dir=Path(args.output_dir),
        n_runs=int(args.n_runs),
        seed=int(args.seed),
        transaction_cost_bps=float(args.transaction_cost_bps),
        decision_metric=str(args.decision_metric),
    )
    print(f"Saved partial artifact-only matched-null audit report to: {output_path}")


if __name__ == "__main__":
    main()
