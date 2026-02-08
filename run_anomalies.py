#!/usr/bin/env python3
"""
Build anomaly-focused dataset from all_runs_clean.csv.

Reads data_ready/all_runs_clean.csv, filters to anomalies only (excludes weld, valve, tee, bend),
adds anomaly_id and quality_flags, and writes data_ready/anomalies_clean.csv.
Does not modify all_runs_clean.csv.
"""

import sys
from pathlib import Path

import pandas as pd

# Rows with these feature_type_norm are NOT anomalies (excluded)
NON_ANOMALY_NORMS = {"weld", "valve", "tee", "bend"}

# Appurtenance-like: exclude if feature_type_raw contains any (case-insensitive)
EXCLUDE_APPURTENANCE_SUBSTRINGS = [
    "tap", "marker", "test lead", "anode", "cp", "casing", "sleeve",
    "fixture", "fitting", "station",
]

OUT_COLUMNS = [
    "anomaly_id",
    "run_year",
    "row_index",
    "feature_type_norm",
    "feature_type_raw",
    "distance_raw_m",
    "clock_position_deg",
    "depth_percent",
    "length_mm",
    "width_mm",
    "joint_number",
    "quality_flags",
]


def main() -> int:
    data_ready = Path("data_ready")
    input_path = data_ready / "all_runs_clean.csv"
    output_path = data_ready / "anomalies_clean.csv"

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path, low_memory=False)

    required = {"run_year", "row_index", "feature_type_norm", "feature_type_raw", "distance_raw_m",
                "clock_position_deg", "depth_percent", "length_mm", "width_mm", "joint_number"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in {input_path}: {missing}", file=sys.stderr)
        return 1

    # 1) Anomalies only: feature_type_norm NOT IN weld, valve, tee, bend
    anomalies = df[~df["feature_type_norm"].str.lower().isin(NON_ANOMALY_NORMS)].copy()
    anomalies_before_exclusion = len(anomalies)

    # 2) Exclude appurtenance-like rows (feature_type_raw substring match, case-insensitive)
    raw_lower = anomalies["feature_type_raw"].astype(str).str.lower()
    appurtenance_mask = pd.Series(False, index=anomalies.index)
    for sub in EXCLUDE_APPURTENANCE_SUBSTRINGS:
        appurtenance_mask = appurtenance_mask | raw_lower.str.contains(sub, na=False, regex=False)
    excluded_appurtenances_count = int(appurtenance_mask.sum())
    anomalies = anomalies[~appurtenance_mask].copy()
    anomalies_after_exclusion = len(anomalies)

    # 3) Stable anomaly_id
    anomalies["anomaly_id"] = (
        anomalies["run_year"].astype(str) + "-" + anomalies["row_index"].astype(str)
    )

    # 4) quality_flags
    def flag_row(row: pd.Series) -> str:
        flags = []
        if pd.isna(row.get("depth_percent")):
            flags.append("missing_depth")
        if pd.isna(row.get("clock_position_deg")):
            flags.append("missing_clock")
        if pd.isna(row.get("joint_number")):
            flags.append("missing_joint")
        return ";".join(flags) if flags else "complete"

    anomalies["quality_flags"] = anomalies.apply(flag_row, axis=1)

    # 5) Select columns in order
    out = anomalies[OUT_COLUMNS].copy()

    # 6) Do not drop rows (already not dropping)
    # 7) Sort by run_year ASC, distance_raw_m ASC
    out = out.sort_values(["run_year", "distance_raw_m"], ascending=[True, True]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    # Console summary
    print("Anomaly summary")
    print("--------------")
    print(f"  anomalies_before_exclusion: {anomalies_before_exclusion}")
    print(f"  excluded_appurtenances_count: {excluded_appurtenances_count}")
    print(f"  anomalies_after_exclusion: {anomalies_after_exclusion}")
    print("  Per-run anomaly counts (after exclusion):")
    for year in sorted(out["run_year"].unique()):
        n = len(out[out["run_year"] == year])
        print(f"    Run {int(year)}: {n}")
    n_total = len(out)
    print(f"  Total: {n_total} anomalies")
    for year in sorted(out["run_year"].unique()):
        sub = out[out["run_year"] == year]
        n = len(sub)
        if n:
            pct_depth = (sub["depth_percent"].isna().sum() / n * 100)
            pct_clock = (sub["clock_position_deg"].isna().sum() / n * 100)
            print(f"  Run {int(year)}: {pct_depth:.1f}% missing depth, {pct_clock:.1f}% missing clock")
    print(f"Written: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
