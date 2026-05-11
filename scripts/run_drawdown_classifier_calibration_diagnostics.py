"""Run calibration diagnostics on saved drawdown-classifier OOF artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import gmtime, strftime

import pandas as pd

from evaluation.calibration_diagnostics import (
    build_calibration_table,
    summarize_fold_probability_diagnostics,
    summarize_probability_diagnostics,
)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows_"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--bin-strategy", choices=["quantile", "uniform"], default="quantile")
    return parser


def _latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} under {directory}")
    return matches[-1]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = strftime("%Y%m%d_%H%M%S", gmtime())

    pooled_rows = []
    fold_rows = []
    calibration_tables = []
    processed_models = []

    for model_dir in sorted([path for path in input_root.iterdir() if path.is_dir()]):
        oof_path = _latest_file(model_dir, "drawdown_risk_classifier_oof_artifacts_*.csv")
        oof = pd.read_csv(oof_path)
        model_name = str(oof["model_name"].iloc[0]) if "model_name" in oof.columns else model_dir.name

        pooled = summarize_probability_diagnostics(
            oof["target"],
            oof["prediction_probability"],
            n_bins=args.n_bins,
            strategy=args.bin_strategy,
        )
        pooled_rows.append({"model_name": model_name, **pooled})

        fold = summarize_fold_probability_diagnostics(
            oof,
            model_name=model_name,
            n_bins=args.n_bins,
            strategy=args.bin_strategy,
        )
        fold_rows.append(fold)

        table = build_calibration_table(
            oof["target"],
            oof["prediction_probability"],
            n_bins=args.n_bins,
            strategy=args.bin_strategy,
        )
        table.insert(0, "model_name", model_name)
        calibration_tables.append(table)
        processed_models.append(model_name)

    pooled_df = pd.DataFrame(pooled_rows).sort_values(
        ["auc_roc", "brier_score"],
        ascending=[False, True],
    ).reset_index(drop=True)
    fold_df = pd.concat(fold_rows, ignore_index=True)
    calibration_df = pd.concat(calibration_tables, ignore_index=True)

    pooled_path = output_dir / f"drawdown_classifier_calibration_summary_{timestamp}.csv"
    fold_path = output_dir / f"drawdown_classifier_calibration_by_fold_{timestamp}.csv"
    table_path = output_dir / f"drawdown_classifier_calibration_bins_{timestamp}.csv"
    report_path = output_dir / f"drawdown_classifier_calibration_report_{timestamp}.md"
    metadata_path = output_dir / f"drawdown_classifier_calibration_metadata_{timestamp}.json"

    pooled_df.to_csv(pooled_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    calibration_df.to_csv(table_path, index=False)

    metadata = {
        "generated_at_utc": timestamp,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "processed_models": processed_models,
        "n_bins": args.n_bins,
        "bin_strategy": args.bin_strategy,
        "classification_only": True,
        "retrained_models": False,
        "used_existing_oof_predictions_only": True,
        "notes": [
            "Calibration diagnostics only.",
            "No model retraining was performed.",
            "No economic overlay evaluation was run.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Drawdown Classifier Calibration Diagnostics",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Input root: `{input_root}`",
        f"- Processed models: `{processed_models}`",
        f"- Bin strategy: `{args.bin_strategy}`",
        f"- Bin count: `{args.n_bins}`",
        "- Used existing OOF predictions only: `True`",
        "- Retrained models: `False`",
        "- Economic overlay evaluation: `False`",
        "",
        "## Pooled Summary",
        "",
        _markdown_table(pooled_df),
        "",
        "## Per-Fold Summary",
        "",
        _markdown_table(fold_df),
        "",
        "## Calibration Bins",
        "",
        _markdown_table(calibration_df),
        "",
        "## Notes",
        "",
        "- Calibration diagnostics only.",
        "- No model retraining was performed.",
        "- No economic overlay evaluation was run.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved pooled summary to: {pooled_path}")
    print(f"Saved per-fold summary to: {fold_path}")
    print(f"Saved calibration bins to: {table_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
