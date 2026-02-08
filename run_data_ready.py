#!/usr/bin/env python3
"""
Data readiness CLI for ILI hackathon.

Ingest Excel (sheets 2007, 2015, 2022), standardize schema + units,
emit clean datasets + schema report + data quality report.
"""

import sys
from pathlib import Path

from src.data_ready import (
    build_dq_metrics,
    run_ingest,
    sanity_check,
    print_console_summary,
    print_output_summary,
    write_dq_md,
    write_readme,
)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="ILI data readiness: ingest Excel, standardize schema + units, emit clean CSVs + reports"
    )
    parser.add_argument("--input", "-i", required=True, help="Input Excel path (e.g. /mnt/data/ILIDataV2.xlsx)")
    parser.add_argument("--out", "-o", default="./data_ready", help="Output directory (default: ./data_ready)")
    parser.add_argument("--runs", type=int, nargs="+", default=[2007, 2015, 2022], help="Run years (sheet names)")
    parser.add_argument("--debug", action="store_true", help="Write debug artifacts (per-run CSVs, frequency tables, data_quality.md) under out/debug/")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        return 1

    cleaned, schema_report, clock_failures_by_year = run_ingest(
        input_path, args.runs, out_dir, debug=args.debug
    )

    errors = schema_report.get("errors", [])
    missing_required = [e for e in errors if "missing required" in e.lower() or "required canonical" in e.lower()]
    if not cleaned or missing_required:
        if missing_required:
            print("ERROR: One or more runs missing required canonical fields.", file=sys.stderr)
        else:
            print("ERROR: No runs could be parsed. Sheets not found or other errors.", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    # PRIMARY: data_quality.csv
    dq_df = build_dq_metrics(cleaned, clock_failures_by_year)
    dq_df.to_csv(out_dir / "data_quality.csv", index=False)

    # PRIMARY: README.md
    write_readme(out_dir)

    # DEBUG only: data_quality.md
    if args.debug:
        write_dq_md(dq_df, out_dir / "debug" / "data_quality.md")

    # Sanity check: required columns + console summary
    ok, errors = sanity_check(cleaned)
    if not ok:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print_output_summary(out_dir, args.debug)
    print()
    print_console_summary(cleaned, dq_df)
    print(f"Done. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
