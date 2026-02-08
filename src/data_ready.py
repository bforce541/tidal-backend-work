"""
Data readiness pipeline for ILI hackathon.

Ingest Excel (sheets 2007, 2015, 2022), standardize schema + units,
emit clean CSVs/parquet + schema report + data quality report.
No alignment, matching, growth, or prediction.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Unit conversions (SI)
FT_TO_M = 0.3048
IN_TO_MM = 25.4

# Canonical schema (spec)
RUN_YEAR = "run_year"
ROW_INDEX = "row_index"
JOINT_NUMBER = "joint_number"
JOINT_LENGTH_FT = "joint_length_ft"
WALL_THICKNESS_IN = "wall_thickness_in"
DISTANCE_ALONG_LINE_FT = "distance_along_line_ft"
DIST_TO_US_WELD_FT = "dist_to_us_weld_ft"
FEATURE_TYPE_RAW = "feature_type_raw"
DEPTH_PERCENT = "depth_percent"
LENGTH_IN = "length_in"
WIDTH_IN = "width_in"
CLOCK_RAW = "clock_raw"
NOTES = "notes"

REQUIRED_CANONICAL = [JOINT_NUMBER, DISTANCE_ALONG_LINE_FT, FEATURE_TYPE_RAW]

# Normalized SI output columns (added during clean)
DISTANCE_RAW_M = "distance_raw_m"
DIST_TO_US_WELD_M = "dist_to_us_weld_m"
LENGTH_MM = "length_mm"
WIDTH_MM = "width_mm"
CLOCK_POSITION_DEG = "clock_position_deg"
FEATURE_TYPE_NORM = "feature_type_norm"

# Header patterns: (regex on normalized header, canonical field)
# Order: more specific first. Applied to normalized header.
HEADER_PATTERNS = [
    (r"joint\s*number|j\.?\s*no\.?", JOINT_NUMBER),
    (r"joint\s*length|j\.?\s*length", JOINT_LENGTH_FT),
    (r"wall\s*thickness|wt\s*\[?\s*in", WALL_THICKNESS_IN),
    (r"log\s*dist|wheel\s*count|ili\s*wheel|odometer|distance\s*along\s*line|distance\s*along", DISTANCE_ALONG_LINE_FT),
    (r"to\s*u/?s\s*w\.?|dist.*u/?s\s*gw|distance\s*to\s*u/?s|dist\s*to\s*us\s*weld", DIST_TO_US_WELD_FT),
    (r"\bevent\b|event\s*desc|feature\s*type|event\s*description", FEATURE_TYPE_RAW),
    (r"metal\s*loss\s*depth|depth\s*\[?\s*%|depth\s*percent", DEPTH_PERCENT),
    (r"length\s*\[?\s*in|length\s*\[", LENGTH_IN),
    (r"width\s*\[?\s*in|width\s*\[", WIDTH_IN),
    (r"o['\u2019']?\s*clock|clock\s*\[|hh\s*:\s*mm|clock\s*pos", CLOCK_RAW),
    (r"\bnotes\b|comment|remarks", NOTES),
]


def normalize_header(h: str) -> str:
    """Robust header normalization: strip, replace newline/cr, collapse spaces, lowercase, punctuation variants."""
    if pd.isna(h):
        return ""
    s = str(h)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    # Normalize punctuation variants: o'clock, o'clock, o'clock -> oclock for matching
    s = re.sub(r"o['\u2018\u2019\u2032]\s*clock", "oclock", s, flags=re.I)
    s = re.sub(r"[^\w\s%\[\]\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_unit(header: str) -> str:
    """Infer unit from header: ft, in, percent."""
    h = normalize_header(header)
    if "[ft]" in h or "ft]" in h or "ft." in h:
        return "ft"
    if "[in]" in h or "in]" in h:
        return "in"
    if "[%]" in h or "%]" in h:
        return "percent"
    return "unknown"


def map_header_to_canonical(header: str) -> Optional[tuple[str, str]]:
    """Map original header to (canonical_field, unit). Returns None if no match."""
    h = normalize_header(header)
    if not h:
        return None
    for pat, canon in HEADER_PATTERNS:
        if re.search(pat, h, re.I):
            unit = infer_unit(header)
            return (canon, unit)
    return None


def discover_sheet(df: pd.DataFrame, sheet_name: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Discover mapping from raw columns to canonical.
    Returns: (original_col -> canonical_field, canonical -> unit, canonical -> original header for report).
    """
    rename: dict[str, str] = {}
    units: dict[str, str] = {}
    canonical_to_original: dict[str, str] = {}
    for col in df.columns:
        orig = str(col)
        mapped = map_header_to_canonical(orig)
        if mapped:
            canon, unit = mapped
            if canon not in rename.values():
                rename[orig] = canon
                units[canon] = unit
                canonical_to_original[canon] = orig
    return rename, units, canonical_to_original


def _clean_numeric_series(ser: pd.Series) -> pd.Series:
    """Remove commas, stray text, whitespace; coerce to numeric."""
    if ser.dtype.kind in "iuif":
        return pd.to_numeric(ser, errors="coerce")
    s = ser.astype(str).str.replace(",", "", regex=False)
    s = s.str.strip().str.replace(r"\s+", "", regex=True)
    # Remove leading/trailing non-numeric
    s = s.replace("", np.nan).replace(r"^[^\d\.\-]+|[^\d\.\-]+$", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _clock_to_degrees_robust(
    v: Any,
    failures: Optional[list[tuple[Any, int]]] = None,
) -> float:
    """
    Convert clock to degrees 0-360.
    Accepts: "4:30", "04:30", "4:30:00"; "4.5" or "4" (hours); numeric 0-360.
    12:00 => 0, 3:00 => 90, 6:00 => 180, 9:00 => 270.
    Returns float (NaN on failure). Appends (raw_value, 1) to failures if provided.
    """
    if pd.isna(v):
        return float("nan")
    # Already numeric degrees
    if isinstance(v, (int, float)):
        f = float(v)
        if 0 <= f <= 360:
            return f
        return f % 360
    s = str(v).strip().replace("\n", " ").replace("\r", " ")
    if not s:
        if failures is not None:
            failures.append((v, 1))
        return float("nan")
    # Try numeric (could be string "90")
    try:
        f = float(s)
        if 0 <= f <= 360:
            return f
        if f > 360:  # might be hours
            return (f % 12) * 30
        return f % 360
    except ValueError:
        pass
    # Try clock format: 4:30, 04:30, 4:30:00
    parts = re.split(r"[\:\.]", s)
    parts = [p.strip() for p in parts if p.strip()]
    try:
        h = int(float(parts[0])) if parts else 0
        m = int(float(parts[1])) if len(parts) > 1 else 0
        sec = int(float(parts[2])) if len(parts) > 2 else 0
        # Interpret as clock: 12 = 0°, 1 = 30°, etc.
        deg = (h % 12) * 30 + (m / 60.0) * 30 + (sec / 3600.0) * 30
        return deg % 360
    except (ValueError, IndexError):
        if failures is not None:
            failures.append((v, 1))
        return float("nan")


def _feature_type_norm(ft: str) -> str:
    """Coarse bucket: weld, valve, metal_loss, dent, other."""
    if pd.isna(ft) or not str(ft).strip():
        return "other"
    s = str(ft).lower()
    if any(x in s for x in ("weld", "girth", "gw")):
        return "weld"
    if "valve" in s:
        return "valve"
    if any(x in s for x in ("metal loss", "metal_loss", "corrosion", "ml")):
        return "metal_loss"
    if "dent" in s:
        return "dent"
    return "other"


def parse_and_clean_run(
    df: pd.DataFrame,
    year: int,
    sheet_name: str,
    clock_failures: Optional[list[tuple[Any, int]]] = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """
    Parse sheet into canonical columns, coerce numerics, convert units, add norm columns.
    Returns: (cleaned_df, schema_info for report, list of missing required field names).
    """
    rename, units, orig_for_report = discover_sheet(df, sheet_name)
    missing_required = [c for c in REQUIRED_CANONICAL if c not in rename.values()]
    if missing_required:
        return pd.DataFrame(), {"sheet": sheet_name, "year": year, "available_columns": list(df.columns)}, missing_required

    out = df.rename(columns=rename).copy()
    out[RUN_YEAR] = year
    out[ROW_INDEX] = np.arange(len(out), dtype=np.int64)

    # Ensure we have all canonical columns (as NaN if missing)
    for c in [
        JOINT_NUMBER,
        JOINT_LENGTH_FT,
        WALL_THICKNESS_IN,
        DISTANCE_ALONG_LINE_FT,
        DIST_TO_US_WELD_FT,
        FEATURE_TYPE_RAW,
        DEPTH_PERCENT,
        LENGTH_IN,
        WIDTH_IN,
        CLOCK_RAW,
        NOTES,
    ]:
        if c not in out.columns:
            out[c] = np.nan

    # Numeric coercion
    if JOINT_NUMBER in out.columns:
        ser = _clean_numeric_series(out[JOINT_NUMBER])
        # Nullable Int64: int where finite, else NA
        int_vals = ser.apply(
            lambda x: int(round(x)) if pd.notna(x) and np.isfinite(x) else pd.NA
        )
        out[JOINT_NUMBER] = pd.array(int_vals, dtype="Int64")
    for col, is_int in [
        (JOINT_LENGTH_FT, False),
        (WALL_THICKNESS_IN, False),
        (DISTANCE_ALONG_LINE_FT, False),
        (DIST_TO_US_WELD_FT, False),
        (DEPTH_PERCENT, False),
        (LENGTH_IN, False),
        (WIDTH_IN, False),
    ]:
        if col in out.columns:
            out[col] = _clean_numeric_series(out[col])

    # Keep feature_type_raw as string; add feature_type_norm
    if FEATURE_TYPE_RAW in out.columns:
        out[FEATURE_TYPE_RAW] = out[FEATURE_TYPE_RAW].astype(str).replace("nan", "")
        out[FEATURE_TYPE_NORM] = out[FEATURE_TYPE_RAW].apply(_feature_type_norm)
    else:
        out[FEATURE_TYPE_NORM] = "other"

    # Clock: parse and log failures
    fail_list: list[tuple[Any, int]] = [] if clock_failures is None else clock_failures
    if CLOCK_RAW in out.columns:
        out[CLOCK_RAW] = out[CLOCK_RAW].astype(str).replace("nan", "")
        out[CLOCK_POSITION_DEG] = out[CLOCK_RAW].apply(
            lambda v: _clock_to_degrees_robust(v, failures=fail_list)
        )
    else:
        out[CLOCK_POSITION_DEG] = float("nan")

    # Unit conversion: add SI columns (keep raw canonical)
    u_dist = units.get(DISTANCE_ALONG_LINE_FT, "ft")
    if u_dist == "in":
        out[DISTANCE_RAW_M] = out[DISTANCE_ALONG_LINE_FT].apply(
            lambda x: float(x) * 0.0254 if pd.notna(x) else float("nan")
        )  # in -> m
    else:
        out[DISTANCE_RAW_M] = out[DISTANCE_ALONG_LINE_FT].apply(
            lambda x: float(x) * FT_TO_M if pd.notna(x) else float("nan")
        )  # ft or unknown -> m

    if DIST_TO_US_WELD_FT in out.columns:
        out[DIST_TO_US_WELD_M] = out[DIST_TO_US_WELD_FT].apply(
            lambda x: float(x) * FT_TO_M if pd.notna(x) else float("nan")
        )
    else:
        out[DIST_TO_US_WELD_M] = float("nan")

    if LENGTH_IN in out.columns:
        out[LENGTH_MM] = out[LENGTH_IN].apply(
            lambda x: float(x) * IN_TO_MM if pd.notna(x) else float("nan")
        )
    else:
        out[LENGTH_MM] = float("nan")
    if WIDTH_IN in out.columns:
        out[WIDTH_MM] = out[WIDTH_IN].apply(
            lambda x: float(x) * IN_TO_MM if pd.notna(x) else float("nan")
        )
    else:
        out[WIDTH_MM] = float("nan")

    schema_info = {
        "sheet": sheet_name,
        "year": year,
        "column_map": {orig: canon for orig, canon in rename.items()},
        "original_headers": orig_for_report,
        "units": units,
    }
    return out, schema_info, []


def run_ingest(
    input_path: Path,
    run_years: list[int],
    out_dir: Path,
    *,
    debug: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict[str, Any], dict[int, list[tuple[Any, int]]]]:
    """
    Load Excel, parse each sheet, clean, write primary outputs and optionally debug outputs.
    Returns: (cleaned_dfs by year, full schema_report dict, clock_failures by year).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(input_path)
    schema_report: dict[str, Any] = {
        "source": str(input_path),
        "sheets_found": xl.sheet_names,
        "run_years_requested": run_years,
        "runs": {},
        "errors": [],
    }
    cleaned: dict[int, pd.DataFrame] = {}
    clock_failures_by_year: dict[int, list[tuple[Any, int]]] = {}

    for year in run_years:
        sheet = str(year)
        if sheet not in xl.sheet_names:
            schema_report["errors"].append(f"Sheet '{sheet}' not found")
            continue
        df = pd.read_excel(input_path, sheet_name=sheet)
        fail_list: list[tuple[Any, int]] = []
        parsed, schema_info, missing_req = parse_and_clean_run(df, year, sheet, clock_failures=fail_list)
        if missing_req:
            schema_report["errors"].append(
                f"Sheet '{sheet}': missing required canonical fields: {missing_req}. Available: {list(df.columns)}"
            )
            schema_report["runs"][sheet] = schema_info
            continue
        cleaned[year] = parsed
        schema_report["runs"][sheet] = schema_info
        clock_failures_by_year[year] = fail_list

        if debug:
            parsed.to_csv(out_dir / "debug" / f"run_{year}_clean.csv", index=False)

    # PRIMARY: all_runs_clean.csv (and optional parquet)
    if cleaned:
        all_dfs = [cleaned[y] for y in sorted(cleaned.keys())]
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_dir / "all_runs_clean.csv", index=False)
        if debug:
            try:
                combined.to_parquet(out_dir / "all_runs_clean.parquet", index=False)
            except Exception:
                pass

    # PRIMARY: schema_report.json
    with open(out_dir / "schema_report.json", "w") as f:
        json.dump(schema_report, f, indent=2)

    # DEBUG only: value_frequencies (feature_type_top20, clock_parse_failures)
    if debug:
        for year, df in cleaned.items():
            if FEATURE_TYPE_RAW in df.columns:
                top = df[FEATURE_TYPE_RAW].value_counts().head(20).reset_index()
                top.columns = ["feature_type_raw", "count"]
                top.to_csv(out_dir / "debug" / f"feature_type_top20_{year}.csv", index=False)
            fail_list = clock_failures_by_year.get(year, [])
            if fail_list:
                agg = Counter()
                for raw, cnt in fail_list:
                    agg[raw] += cnt
                fail_df = pd.DataFrame([{"raw_value": k, "count": v} for k, v in agg.items()])
                fail_df.to_csv(out_dir / "debug" / f"clock_parse_failures_{year}.csv", index=False)
            else:
                pd.DataFrame(columns=["raw_value", "count"]).to_csv(
                    out_dir / "debug" / f"clock_parse_failures_{year}.csv", index=False
                )

    return cleaned, schema_report, clock_failures_by_year


def build_dq_metrics(
    cleaned: dict[int, pd.DataFrame],
    clock_failures_by_year: dict[int, list[tuple[Any, int]]],
) -> pd.DataFrame:
    """Build data_quality.csv (metrics per run)."""
    rows = []
    for year in sorted(cleaned.keys()):
        df = cleaned[year]
        n = len(df)
        fail_count = sum(c for _, c in clock_failures_by_year.get(year, []))
        pct_joint = (df[JOINT_NUMBER].isna().sum() / n * 100) if n else 0
        pct_dist = (df[DISTANCE_RAW_M].isna().sum() / n * 100) if n else 0
        pct_depth = (df[DEPTH_PERCENT].isna().sum() / n * 100) if n and DEPTH_PERCENT in df.columns else 0
        pct_clock = (df[CLOCK_POSITION_DEG].isna().sum() / n * 100) if n else 0
        n_unique_ft = df[FEATURE_TYPE_RAW].nunique() if FEATURE_TYPE_RAW in df.columns else 0
        top5_norm = (
            df[FEATURE_TYPE_NORM].value_counts().head(5)
            if FEATURE_TYPE_NORM in df.columns else pd.Series()
        )
        dist_min = df[DISTANCE_RAW_M].min() if n and DISTANCE_RAW_M in df.columns else None
        dist_max = df[DISTANCE_RAW_M].max() if n and DISTANCE_RAW_M in df.columns else None
        # Monotonic check
        d = df[DISTANCE_RAW_M].dropna()
        if len(d) > 1:
            non_mono = (d.diff() < 0).sum()
            pct_non_mono = non_mono / len(d) * 100
        else:
            pct_non_mono = 0.0
        row = {
            "run_year": year,
            "row_count": n,
            "pct_missing_joint_number": round(pct_joint, 2),
            "pct_missing_distance_raw_m": round(pct_dist, 2),
            "pct_missing_depth_percent": round(pct_depth, 2),
            "pct_missing_clock_position_deg": round(pct_clock, 2),
            "clock_parse_fail_count": fail_count,
            "unique_feature_type_raw_count": int(n_unique_ft),
            "distance_raw_m_min": dist_min,
            "distance_raw_m_max": dist_max,
            "pct_rows_non_monotonic_distance": round(pct_non_mono, 2),
        }
        for i, (cat, cnt) in enumerate(top5_norm.items()):
            row[f"top5_norm_{i+1}_category"] = cat
            row[f"top5_norm_{i+1}_count"] = int(cnt) if pd.notna(cnt) else None
        rows.append(row)
    return pd.DataFrame(rows)


def write_dq_md(dq_df: pd.DataFrame, out_path: Path) -> None:
    """Write human-readable data_quality.md summary."""
    table_str = dq_df.to_string(index=False)
    lines = [
        "# Data Quality Report",
        "",
        "## Per-run metrics",
        "",
        table_str,
        "",
        "## Warnings",
        "",
    ]
    warning_count = 0
    for _, r in dq_df.iterrows():
        y = r.get("run_year", "")
        if r.get("pct_rows_non_monotonic_distance", 0) > 1.0:
            lines.append(f"- Run {y}: distance_raw_m is non-monotonic in >1% of rows (no fix applied).")
            warning_count += 1
    if warning_count == 0:
        lines.append("- None.")
    lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def sanity_check(cleaned: dict[int, pd.DataFrame]) -> tuple[bool, list[str]]:
    """
    Assert required columns exist in each cleaned df.
    Returns: (ok, list of error messages).
    """
    errors = []
    for year, df in cleaned.items():
        for col in REQUIRED_CANONICAL:
            if col not in df.columns:
                errors.append(f"Run {year}: missing required column '{col}'")
    return len(errors) == 0, errors


def write_readme(out_dir: Path) -> None:
    """Write PRIMARY README.md describing the data_ready stage and outputs."""
    content = """# Data ready

## Purpose

This directory is produced by the **data readiness** stage of the ILI pipeline. The stage ingests the source Excel (sheets per run year), standardizes schema and units, and emits clean datasets plus a schema and quality report. No alignment, matching, growth, or prediction is performed here.

**No rows are dropped; missing values are preserved as NaN.**

## Primary outputs

| File | Description |
|------|-------------|
| `all_runs_clean.csv` | Unified dataset: all runs concatenated with canonical columns and SI-normalized fields. |
| `data_quality.csv` | Machine-readable quality metrics per run (row counts, % missing, parse failures, etc.). |
| `schema_report.json` | Column mapping (raw → canonical) and units per run; source path and sheet list. |
| `README.md` | This file. |

## Debug outputs (optional)

To get per-run cleaned CSVs, clock parse failure examples, feature-type frequency tables, and a human-readable `data_quality.md`, re-run with the `--debug` flag. All debug artifacts are written under `debug/`:

- `debug/run_2007_clean.csv`, `run_2015_clean.csv`, `run_2022_clean.csv`
- `debug/clock_parse_failures_2007.csv`, …
- `debug/feature_type_top20_2007.csv`, …
- `debug/data_quality.md`

Example:

```bash
python run_data_ready.py --input /path/to/ILIDataV2.xlsx --out ./data_ready --runs 2007 2015 2022 --debug
```
"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "README.md").write_text(content.strip() + "\n", encoding="utf-8")


def print_output_summary(out_dir: Path, debug: bool) -> None:
    """Print which outputs were written (primary vs debug)."""
    out_dir = Path(out_dir)
    print("Primary outputs written:")
    for name in ["all_runs_clean.csv", "data_quality.csv", "schema_report.json", "README.md"]:
        print(f"  - {out_dir / name}")
    if debug:
        debug_dir = out_dir / "debug"
        print("Debug outputs written:")
        for f in sorted(debug_dir.iterdir()) if debug_dir.exists() else []:
            print(f"  - debug/{f.name}")
    else:
        print("Debug outputs: skipped (use --debug for per-run CSVs, frequency tables, data_quality.md)")


def print_console_summary(cleaned: dict[int, pd.DataFrame], dq_df: pd.DataFrame) -> None:
    """Print one block per run with key metrics."""
    for year in sorted(cleaned.keys()):
        df = cleaned[year]
        row = dq_df[dq_df["run_year"] == year].iloc[0] if len(dq_df) else {}
        print(f"--- Run {year} ---")
        print(f"  row_count: {len(df)}")
        print(f"  required columns: {all(c in df.columns for c in REQUIRED_CANONICAL)}")
        print(f"  % missing joint_number: {row.get('pct_missing_joint_number', 'N/A')}")
        print(f"  % missing distance_raw_m: {row.get('pct_missing_distance_raw_m', 'N/A')}")
        print(f"  % missing clock_position_deg: {row.get('pct_missing_clock_position_deg', 'N/A')}")
        print(f"  clock_parse_fail_count: {row.get('clock_parse_fail_count', 'N/A')}")
        print(f"  unique feature_type_raw: {row.get('unique_feature_type_raw_count', 'N/A')}")
        print(f"  distance_raw_m range: [{row.get('distance_raw_m_min')}, {row.get('distance_raw_m_max')}]")
        if row.get("pct_rows_non_monotonic_distance", 0) > 1.0:
            print(f"  WARNING: non-monotonic distance >1%")
        print()
