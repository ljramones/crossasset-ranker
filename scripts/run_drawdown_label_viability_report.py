"""Generate an offline drawdown-label viability report from a prepared feature CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import strftime, gmtime

import pandas as pd

from evaluation.drawdown_labels import (
    append_drawdown_label_grid,
    evaluate_candidate_drawdown_labels,
    get_drawdown_label_columns,
)
from evaluation.walk_forward import generate_walk_forward_splits


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _select_primary_target(candidate_summary: pd.DataFrame) -> str | None:
    if candidate_summary.empty:
        return None
    ranked = candidate_summary.sort_values(
        [
            "all_splits_viable",
            "fraction_viable_splits",
            "min_positive_count",
            "mean_positive_rate",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return str(ranked.iloc[0]["label"])


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
    parser.add_argument("--input-csv", required=True, help="Prepared offline feature CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for timestamped outputs.")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--price-column", default="Adj Close")
    parser.add_argument("--horizons", default="10,20")
    parser.add_argument("--thresholds", default="-0.02,-0.03,-0.05")
    parser.add_argument("--train-size", type=int, default=756)
    parser.add_argument("--val-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--step-size", type=int, default=252)
    parser.add_argument("--min-positive-count", type=int, default=5)
    parser.add_argument("--min-negative-count", type=int, default=5)
    parser.add_argument("--min-positive-rate", type=float, default=0.01)
    parser.add_argument("--max-positive-rate", type=float, default=0.99)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = strftime("%Y%m%d_%H%M%S", gmtime())

    frame = pd.read_csv(input_path)
    if args.date_column in frame.columns:
        frame[args.date_column] = pd.to_datetime(frame[args.date_column])
        frame = frame.sort_values(args.date_column).reset_index(drop=True)

    horizons = _parse_int_list(args.horizons)
    thresholds = _parse_float_list(args.thresholds)
    enriched = append_drawdown_label_grid(
        frame,
        price_column=args.price_column,
        horizons=horizons,
        thresholds=thresholds,
    )
    splits = generate_walk_forward_splits(
        enriched,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )

    label_columns = get_drawdown_label_columns(enriched)
    diagnostics = evaluate_candidate_drawdown_labels(
        enriched,
        splits,
        label_columns=label_columns,
        min_positive_count=args.min_positive_count,
        min_negative_count=args.min_negative_count,
        min_positive_rate=args.min_positive_rate,
        max_positive_rate=args.max_positive_rate,
    )

    prevalence = diagnostics["prevalence"]
    viability = diagnostics["viability"]
    candidate_summary = diagnostics["candidate_summary"]
    primary_target = _select_primary_target(candidate_summary)

    prevalence_path = output_dir / f"drawdown_label_prevalence_{timestamp}.csv"
    viability_path = output_dir / f"drawdown_label_viability_{timestamp}.csv"
    summary_path = output_dir / f"drawdown_label_candidate_summary_{timestamp}.csv"
    report_path = output_dir / f"drawdown_label_viability_report_{timestamp}.md"
    metadata_path = output_dir / f"drawdown_label_viability_metadata_{timestamp}.json"

    prevalence.to_csv(prevalence_path, index=False)
    viability.to_csv(viability_path, index=False)
    candidate_summary.to_csv(summary_path, index=False)

    metadata = {
        "generated_at_utc": timestamp,
        "input_csv": str(input_path),
        "output_dir": str(output_dir),
        "price_column": args.price_column,
        "date_column": args.date_column,
        "horizons": list(horizons),
        "thresholds": list(thresholds),
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "step_size": args.step_size,
        "num_splits": len(splits),
        "label_columns": label_columns,
        "primary_target_recommendation": primary_target,
        "notes": [
            "Label diagnostics only.",
            "No models were trained.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Drawdown Label Viability Report",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Input CSV: `{input_path}`",
        f"- Splits: `{len(splits)}`",
        f"- Horizons: `{list(horizons)}`",
        f"- Thresholds: `{list(thresholds)}`",
        f"- Recommended primary target: `{primary_target}`" if primary_target else "- Recommended primary target: `None`",
        "",
        "## Candidate Summary",
        "",
        _markdown_table(candidate_summary),
        "",
        "## Overall Prevalence",
        "",
        _markdown_table(prevalence),
        "",
        "## Split Viability",
        "",
        _markdown_table(viability),
        "",
        "## Notes",
        "",
        "- Label diagnostics only.",
        "- No classifier training or economic overlay evaluation was run.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved prevalence to: {prevalence_path}")
    print(f"Saved viability to: {viability_path}")
    print(f"Saved candidate summary to: {summary_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved metadata to: {metadata_path}")
    if primary_target:
        print(f"Recommended primary target: {primary_target}")


if __name__ == "__main__":
    main()
