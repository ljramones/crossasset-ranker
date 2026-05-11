"""Standalone CLI for fold-safe drawdown-risk probability calibration experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import gmtime, strftime

import pandas as pd

from evaluation.drawdown_labels import append_drawdown_label_grid
from evaluation.walk_forward import generate_walk_forward_splits
from experiments.drawdown_risk_calibration_experiment import run_drawdown_risk_calibration_experiment
from scripts.run_drawdown_risk_classifier_experiment import infer_feature_columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--input-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--price-column", default="Adj Close")
    parser.add_argument("--target-column", default="target_drawdown_event_20d_3pct")
    parser.add_argument("--base-model-name", default="regularized_linear")
    parser.add_argument("--calibration-method", choices=["identity", "platt", "isotonic"], default="platt")
    parser.add_argument("--asset-name", default="SPY")
    parser.add_argument("--run-purpose", choices=["plumbing", "diagnostic", "decision_grade"], default="diagnostic")
    parser.add_argument("--decision-grade", action="store_true")
    parser.add_argument("--train-size", type=int, default=756)
    parser.add_argument("--val-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--step-size", type=int, default=252)
    parser.add_argument("--n-bins", type=int, default=10)
    return parser


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


def determine_decision_grade(*, run_purpose: str, decision_grade_flag: bool) -> bool:
    return bool(decision_grade_flag or run_purpose == "decision_grade")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print("Drawdown-risk calibration runner dry run only.")
        print("No data will be loaded. No outputs will be written.")
        return

    required = ["input_csv", "output_dir", "target_column", "base_model_name", "calibration_method"]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"Missing required execute arguments: {missing}")

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
    decision_grade = determine_decision_grade(
        run_purpose=args.run_purpose,
        decision_grade_flag=args.decision_grade,
    )

    frame = pd.read_csv(input_path)
    if args.date_column in frame.columns:
        frame[args.date_column] = pd.to_datetime(frame[args.date_column])
        frame = frame.sort_values(args.date_column).reset_index(drop=True)
    frame = append_drawdown_label_grid(
        frame,
        price_column=args.price_column,
        horizons=(20,),
        thresholds=(-0.03,),
    )
    feature_columns = infer_feature_columns(frame)
    splits = generate_walk_forward_splits(
        frame,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        step_size=args.step_size,
    )
    result = run_drawdown_risk_calibration_experiment(
        frame=frame,
        splits=splits,
        feature_columns=feature_columns,
        target_column=args.target_column,
        base_model_name=args.base_model_name,
        calibration_method=args.calibration_method,
        asset_name=args.asset_name,
        n_bins=args.n_bins,
    )

    summary_path = output_dir / f"drawdown_risk_calibration_summary_{timestamp}.csv"
    fold_details_path = output_dir / f"drawdown_risk_calibration_fold_details_{timestamp}.csv"
    oof_path = output_dir / f"drawdown_risk_calibration_oof_artifacts_{timestamp}.csv"
    bins_path = output_dir / f"drawdown_risk_calibration_bins_{timestamp}.csv"
    report_path = output_dir / f"drawdown_risk_calibration_report_{timestamp}.md"
    metadata_path = output_dir / f"drawdown_risk_calibration_metadata_{timestamp}.json"

    result.summary.to_csv(summary_path, index=False)
    result.fold_details.to_csv(fold_details_path, index=False)
    result.oof_artifacts.to_csv(oof_path, index=False)
    result.calibration_bins.to_csv(bins_path, index=False)

    report_lines = [
        "# Drawdown-Risk Calibration Experiment Report",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Input CSV: `{input_path}`",
        f"- Base model: `{args.base_model_name}`",
        f"- Calibration method: `{args.calibration_method}`",
        f"- Target: `{args.target_column}`",
        f"- Splits: `{len(splits)}`",
        f"- Run purpose: `{args.run_purpose}`",
        f"- Decision grade: `{decision_grade}`",
        "- Classification only: `True`",
        "- Economic overlay used: `False`",
        "- Trading strategy validated: `False`",
        "",
        "## Summary",
        "",
        _markdown_table(result.summary),
        "",
        "## Fold Details",
        "",
        _markdown_table(result.fold_details),
        "",
        "## Calibration Bins",
        "",
        _markdown_table(result.calibration_bins.head(20)),
        "",
        "## Notes",
        "",
        "- Standalone calibration workflow only.",
        "- Base classifier fit on train only.",
        "- Calibrator fit on validation only.",
        "- Final calibrated probabilities evaluated on test only.",
        "- No economic overlay evaluation was run.",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    metadata = {
        "generated_at_utc": timestamp,
        "input_csv": str(input_path),
        "output_dir": str(output_dir),
        "base_model_name": args.base_model_name,
        "calibration_method": args.calibration_method,
        "target_column": args.target_column,
        "asset_name": args.asset_name,
        "num_splits": len(splits),
        "feature_count": len(feature_columns),
        "run_purpose": args.run_purpose,
        "decision_grade": decision_grade,
        "classification_only": True,
        "economic_overlay_used": False,
        "trading_strategy_validated": False,
        "notes": [
            "Standalone calibration workflow only.",
            "Base classifier fit on train only.",
            "Calibrator fit on validation only.",
            "Final calibrated probabilities evaluated on test only.",
            "No economic overlay evaluation was run.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved fold details to: {fold_details_path}")
    print(f"Saved OOF artifacts to: {oof_path}")
    print(f"Saved calibration bins to: {bins_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()

