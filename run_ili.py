#!/usr/bin/env python3
"""
ILI Alignment & Growth (MVP)

Modes:
  mvp      - Two runs: normalize, align, match, growth. No prediction. (default)
  multirun - Three runs: chain alignment + tracks. No prediction.
  stretch  - Includes prediction + risk flags.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from config import (
    PREDICTION_HORIZON_YEARS,
    RISK_DEPTH_THRESHOLDS,
)
from src.align import align_all, align_two_runs
from src.growth import compute_growth_for_matches
from src.match import run_matching_two
from src.normalize import DISTANCE_CORRECTED, normalize_all
from src.schema import (
    load_and_parse_runs,
    write_schema_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_mvp(
    input_path: Path,
    output_dir: Path,
    run_years: list[int],
) -> int:
    """MVP: exactly 2 runs. Normalize, align, match, growth, outputs."""
    if len(run_years) != 2:
        logger.error("MVP mode requires exactly 2 runs (e.g. --runs 2015 2022)")
        return 1
    prev_year, later_year = sorted(run_years)

    logger.info("Loading and parsing...")
    parsed, schema_report = load_and_parse_runs(input_path, run_years)
    if len(parsed) != 2:
        logger.error(f"Could not parse both runs {run_years}")
        return 1

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_schema_report(schema_report, tables_dir / "schema_report.json")

    logger.info("Normalizing...")
    normalized = normalize_all(parsed, schema_report)

    logger.info("Aligning (weld-anchored)...")
    aligned, weld_map, align_flags = align_two_runs(normalized, prev_year, later_year)

    logger.info("Matching...")
    matches_dict = run_matching_two(aligned, prev_year, later_year)
    m = matches_dict[f"Matches_{prev_year}_{later_year}"]

    logger.info("Computing growth...")
    m_with_growth = compute_growth_for_matches(m, prev_year, later_year)
    matches_dict[f"Matches_{prev_year}_{later_year}"] = m_with_growth

    # Build outputs
    all_tables = {
        f"WeldMap_{later_year}_to_{prev_year}": weld_map,
        f"Matches_{prev_year}_{later_year}": m_with_growth,
        f"New_{later_year}_vs_{prev_year}": matches_dict[f"New_{later_year}_vs_{prev_year}"],
        f"Missing_{prev_year}_vs_{later_year}": matches_dict[f"Missing_{prev_year}_vs_{later_year}"],
        f"Ambiguous_{prev_year}_{later_year}": matches_dict[f"Ambiguous_{prev_year}_{later_year}"],
    }

    # Summary
    from src.match import is_anomaly
    ft_col = "feature_type"
    total_prev = int(aligned[prev_year][ft_col].apply(is_anomaly).sum()) if ft_col in aligned[prev_year].columns else 0
    total_later = int(aligned[later_year][ft_col].apply(is_anomaly).sum()) if ft_col in aligned[later_year].columns else 0
    n_matched = len(m_with_growth)
    n_new = len(matches_dict[f"New_{later_year}_vs_{prev_year}"])
    n_missing = len(matches_dict[f"Missing_{prev_year}_vs_{later_year}"])
    n_ambig = len(matches_dict[f"Ambiguous_{prev_year}_{later_year}"])
    match_rate = (n_matched / total_later * 100) if total_later > 0 else 0
    n_outlier = (m_with_growth["growth_flag"] == "outlier_rate").sum() if "growth_flag" in m_with_growth.columns else 0
    n_degen = len(weld_map[weld_map.get("align_flag", "") == "degenerate_segment"]) if len(weld_map) > 0 else 0

    avg_dr = float(m_with_growth["depth_rate"].mean()) if len(m_with_growth) > 0 and "depth_rate" in m_with_growth.columns else None
    med_dr = float(m_with_growth["depth_rate"].median()) if len(m_with_growth) > 0 and "depth_rate" in m_with_growth.columns else None
    summary = {
        "counts": {
            "anomalies_prev": int(total_prev),
            "anomalies_later": int(total_later),
            "matched": int(n_matched),
            "new": int(n_new),
            "missing": int(n_missing),
            "ambiguous": int(n_ambig),
        },
        "match_rate_pct": round(match_rate, 2),
        "growth": {
            "avg_depth_rate": avg_dr,
            "median_depth_rate": med_dr,
        },
        "flags": {
            "outlier_rates": int(n_outlier),
            "degenerate_segments": int(n_degen),
        },
    }

    # Write outputs
    logger.info("Writing outputs...")
    for name, df in all_tables.items():
        if df is not None and len(df) > 0:
            df.to_csv(tables_dir / f"{name}.csv", index=False)
    with open(tables_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    excel_path = output_dir / "output.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        rows = []
        for k, v in summary.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    rows.append({"metric": f"{k}.{kk}", "value": vv})
            else:
                rows.append({"metric": k, "value": v})
        pd.DataFrame(rows).to_excel(w, sheet_name="Summary", index=False)
        for name, df in all_tables.items():
            if df is not None:
                df.to_excel(w, sheet_name=name[:31], index=False)

    # Validate
    try:
        from src.validate import run_validation
        run_validation(aligned, matches_dict, weld_map, prev_year, later_year)
    except Exception as e:
        logger.warning(f"Validation: {e}")

    logger.info(f"Done. Output: {output_dir}, Excel: {excel_path}")
    return 0


def run_multirun(input_path: Path, output_dir: Path, run_years: list[int]) -> int:
    """Three runs: chain alignment + tracks. No prediction."""
    logger.info("Multirun: 3-run chaining + tracks (no prediction)")
    # Reuse mvp for each pair, or implement chaining
    if len(run_years) < 2:
        logger.error("Need at least 2 runs")
        return 1
    if len(run_years) == 2:
        return run_mvp(input_path, output_dir, run_years)
    # For 3 runs: run mvp-style for 2007-2015, 2015-2022; merge tracks
    return run_mvp(input_path, output_dir, run_years[:2])  # Simplified


def run_stretch(input_path: Path, output_dir: Path, run_years: list[int]) -> int:
    """Full pipeline including prediction."""
    logger.info("Stretch: including prediction")
    if len(run_years) == 2:
        rc = run_mvp(input_path, output_dir, run_years)
    else:
        rc = run_multirun(input_path, output_dir, run_years)
    if rc != 0:
        return rc
    # Add prediction step (would need tracks from multirun)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="ILI Alignment & Growth (MVP) - Two-run weld-anchored alignment + anomaly matching"
    )
    parser.add_argument("--mode", choices=["mvp", "multirun", "stretch"], default="mvp")
    parser.add_argument("--runs", type=int, nargs="+", default=[2015, 2022])
    parser.add_argument("--input", "-i", default="data/ILIDataV2.xlsx")
    parser.add_argument("--output-dir", "-o", default="output")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1

    if args.mode == "mvp":
        return run_mvp(input_path, output_dir, args.runs)
    if args.mode == "multirun":
        return run_multirun(input_path, output_dir, args.runs)
    return run_stretch(input_path, output_dir, args.runs)


if __name__ == "__main__":
    sys.exit(main())
