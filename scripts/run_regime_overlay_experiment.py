"""CLI wrapper for the standalone regime-overlay experiment runner.

This script is intentionally separate from ``main.py`` so the overlay workflow
can evolve without touching the existing model-zoo CLI.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evaluation.walk_forward import generate_walk_forward_splits
from experiments.regime_overlay_experiment import (
    build_fold_details_frame,
    build_matched_nulls_frame,
    run_fold_local_regime_overlay_experiment,
)
from regime.regime_detection import MarketRegimeDetector, RegimeDetectionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone regime-overlay experiment.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate the command shape without loading data or writing outputs.")
    mode.add_argument("--execute", action="store_true", help="Run the real standalone overlay experiment.")
    parser.add_argument("--input-csv", help="Feature frame CSV containing returns and regime features.")
    parser.add_argument("--output-dir", default="results", help="Directory for summary and audit outputs.")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--realized-vol-window", type=int, default=20)
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--null-runs", type=int, default=100)
    parser.add_argument("--asset-name", default="SPY")
    parser.add_argument("--model-name", default="hmm_regime_overlay_hard_veto")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--model-type", default="hmm", choices=("hmm", "gmm"))
    return parser


def _validate_execute_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    required = {
        "--input-csv": args.input_csv,
        "--train-size": args.train_size,
        "--val-size": args.val_size,
        "--test-size": args.test_size,
        "--step-size": args.step_size,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error(f"--execute requires: {', '.join(missing)}")


def _build_output_paths(output_dir: Path, *, timestamp: str) -> dict[str, Path]:
    return {
        "summary": output_dir / f"regime_overlay_experiment_summary_{timestamp}.csv",
        "fold_details": output_dir / f"regime_overlay_experiment_fold_details_{timestamp}.csv",
        "audit_artifacts": output_dir / f"regime_overlay_experiment_audit_artifacts_{timestamp}.csv",
        "matched_nulls": output_dir / f"regime_overlay_experiment_matched_nulls_{timestamp}.csv",
        "report_markdown": output_dir / f"regime_overlay_experiment_report_{timestamp}.md",
        "metadata": output_dir / f"regime_overlay_experiment_metadata_{timestamp}.json",
    }


def _build_run_metadata(args: argparse.Namespace, *, timestamp: str, n_splits: int) -> dict[str, object]:
    return {
        "generated_at_utc": timestamp,
        "mode": "execute",
        "input_csv": args.input_csv,
        "output_dir": args.output_dir,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "target_vol": args.target_vol,
        "realized_vol_window": args.realized_vol_window,
        "transaction_cost_bps": args.transaction_cost_bps,
        "null_runs": args.null_runs,
        "asset_name": args.asset_name,
        "model_name": args.model_name,
        "date_column": args.date_column,
        "model_type": args.model_type,
        "n_splits": n_splits,
        "decision_grade": False,
        "notes": [
            "Standalone regime-overlay workflow only.",
            "Do not treat results as validated unless matched-null gates pass at higher null-run budgets.",
        ],
    }


def _dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.astype(object).itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = _format_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [_format_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def _build_markdown_report(
    *,
    summary_frame: pd.DataFrame,
    fold_details_frame: pd.DataFrame,
    matched_nulls_frame: pd.DataFrame,
    metadata: dict[str, object],
) -> str:
    lines = [
        "# Regime Overlay Experiment Report",
        "",
        f"- Generated at (UTC): {metadata['generated_at_utc']}",
        f"- Input CSV: `{metadata['input_csv']}`",
        f"- Model type: `{metadata['model_type']}`",
        f"- Asset: `{metadata['asset_name']}`",
        f"- Splits: `{metadata['n_splits']}`",
        f"- Null runs: `{metadata['null_runs']}`",
        f"- Decision grade: `{metadata['decision_grade']}`",
        "",
        "## Summary",
        "",
        _dataframe_to_markdown_table(summary_frame),
        "",
        "## Fold Details",
        "",
        _dataframe_to_markdown_table(fold_details_frame),
        "",
        "## Matched Null Diagnostics",
        "",
        _dataframe_to_markdown_table(matched_nulls_frame),
        "",
        "## Notes",
        "",
    ]
    for note in metadata["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)

    if args.dry_run:
        print("Regime overlay runner dry run only.")
        print("No data will be loaded. No detector will be fitted. No outputs will be written.")
        print(f"Output directory (not created in dry run): {output_dir}")
        print(f"Model type: {args.model_type}")
        return output_dir / "DRY_RUN_ONLY"

    _validate_execute_args(args, parser)

    frame = pd.read_csv(args.input_csv)
    if args.date_column in frame.columns:
        frame[args.date_column] = pd.to_datetime(frame[args.date_column])
        frame = frame.set_index(args.date_column)

    splits = generate_walk_forward_splits(
        frame,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    result = run_fold_local_regime_overlay_experiment(
        frame=frame,
        splits=splits,
        detector_factory=lambda: MarketRegimeDetector(RegimeDetectionConfig(model_type=args.model_type)),
        asset_name=args.asset_name,
        model_name=args.model_name,
        target_vol=args.target_vol,
        realized_vol_window=args.realized_vol_window,
        transaction_cost_bps=args.transaction_cost_bps,
        null_n_runs=args.null_runs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_paths = _build_output_paths(output_dir, timestamp=timestamp)
    fold_details = build_fold_details_frame(result)
    matched_nulls = build_matched_nulls_frame(result)
    metadata = _build_run_metadata(args, timestamp=timestamp, n_splits=len(result.split_results))
    markdown_report = _build_markdown_report(
        summary_frame=result.summary,
        fold_details_frame=fold_details,
        matched_nulls_frame=matched_nulls,
        metadata=metadata,
    )

    result.summary.to_csv(output_paths["summary"], index=False)
    fold_details.to_csv(output_paths["fold_details"], index=False)
    result.audit_artifact_frame.to_csv(output_paths["audit_artifacts"], index=False)
    matched_nulls.to_csv(output_paths["matched_nulls"], index=False)
    output_paths["report_markdown"].write_text(markdown_report, encoding="utf-8")
    output_paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_paths["summary"]


if __name__ == "__main__":
    main()
