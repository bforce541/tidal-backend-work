"""
Robust schema discovery for ILI Excel data.

- Normalize headers: strip whitespace, lowercase, remove \\n, collapse spaces
- Regex mapping to canonical fields
- Unit inference from header
- Raise clear error if required field cannot be mapped
- Emit schema_report.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Canonical field names (required for MVP)
JOINT_NUMBER = "joint_number"
DISTANCE_ALONG_LINE = "distance_along_line"  # log dist / wheel count
DIST_TO_US_WELD = "dist_to_us_weld"
DIST_TO_DS_WELD = "dist_to_ds_weld"  # optional
FEATURE_TYPE = "feature_type"
DEPTH_PERCENT = "depth_percent"
LENGTH_IN = "length_in"
WIDTH_IN = "width_in"
CLOCK = "clock"

# Output canonical (after unit conversion)
DISTANCE = "distance_raw_m"  # meters
RELATIVE_POSITION = "dist_to_us_weld_m"
DISTANCE_CORRECTED = "distance_corrected_m"
CLOCK_POSITION = "clock_position_deg"
LENGTH_MM = "length_mm"
WIDTH_MM = "width_mm"
RUN_YEAR = "run_year"
ROW_INDEX = "row_index"

REQUIRED_CANONICAL = [
    JOINT_NUMBER,
    DISTANCE_ALONG_LINE,
    FEATURE_TYPE,
]
# Strongly recommended (warn if missing)
RECOMMENDED_CANONICAL = [
    DIST_TO_US_WELD,
    DEPTH_PERCENT,
    LENGTH_IN,
    WIDTH_IN,
    CLOCK,
]

# Regex patterns (applied to normalized header) -> canonical field
# Order matters: more specific first
HEADER_PATTERNS = [
    (r"joint\s*number|j\.?\s*no\.?", JOINT_NUMBER),
    (r"log\s*dist|wheel\s*count|ili\s*wheel|odometer|distance\s*along", DISTANCE_ALONG_LINE),
    (r"to\s*u/?s\s*w\.?|dist.*u/?s\s*gw|distance\s*to\s*u/?s", DIST_TO_US_WELD),
    (r"to\s*d/?s\s*w\.?|dist.*d/?s\s*gw|distance\s*to\s*d/?s", DIST_TO_DS_WELD),
    (r"\bevent\b|event\s*desc|feature\s*type|event\s*description", FEATURE_TYPE),
    (r"metal\s*loss\s*depth|depth\s*\[?\s*%|depth\s*percent", DEPTH_PERCENT),
    (r"length\s*\[?\s*in|length\s*\[", LENGTH_IN),
    (r"width\s*\[?\s*in|width\s*\[", WIDTH_IN),
    (r"o['\']?clock|clock\s*\[|hh\s*:\s*mm", CLOCK),
]


def normalize_header(h: str) -> str:
    """Strip whitespace, lowercase, remove \\n, collapse spaces."""
    if pd.isna(h):
        return ""
    s = str(h).replace("\n", " ").replace("\r", " ").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def infer_unit_from_header(header: str) -> str:
    """Infer unit from header: [ft]=feet, [in]=inches, [%]=percent."""
    h = normalize_header(header)
    if "[ft]" in h or "ft]" in h or r"ft\." in h or "ft." in h:
        return "ft"
    if "[in]" in h or "in]" in h:
        return "in"
    if "[%]" in h or "%]" in h:
        return "percent"
    return "unknown"


def map_header_to_canonical(header: str) -> Optional[tuple[str, str]]:
    """
    Map normalized header to (canonical_field, inferred_unit).
    Returns None if no match.
    """
    h = normalize_header(header)
    if not h:
        return None
    for pat, canon in HEADER_PATTERNS:
        if re.search(pat, h, re.I):
            unit = infer_unit_from_header(header)
            return (canon, unit)
    return None


def discover_and_map_sheet(df: pd.DataFrame, sheet_name: str) -> dict[str, tuple[str, str]]:
    """
    For each column, find canonical mapping and unit.
    Returns: {original_col: (canonical_field, unit)}
    """
    result = {}
    for col in df.columns:
        orig = str(col)
        mapped = map_header_to_canonical(orig)
        if mapped:
            result[orig] = mapped
    return result


def build_column_map(
    df: pd.DataFrame, sheet_name: str
) -> tuple[dict[str, str], dict[str, str], list[str], list[str]]:
    """
    Build mapping: original_col -> canonical_field.
    Returns: (rename_map, unit_map, missing_required, missing_recommended)
    """
    discovered = discover_and_map_sheet(df, sheet_name)
    rename = {}
    units: dict[str, str] = {}
    seen_canon = set()
    for orig, (canon, unit) in discovered.items():
        if canon not in seen_canon:
            rename[orig] = canon
            units[canon] = unit
            seen_canon.add(canon)
    missing_req = [c for c in REQUIRED_CANONICAL if c not in rename.values()]
    missing_rec = [c for c in RECOMMENDED_CANONICAL if c not in rename.values()]
    return rename, units, missing_req, missing_rec


def parse_run(
    df: pd.DataFrame, year: int, sheet_name: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Parse run into canonical columns. Raise ValueError if required field missing.
    Returns: (parsed_df, schema_info for report).
    """
    rename, units, missing_req, missing_rec = build_column_map(df, sheet_name)
    if missing_req:
        raise ValueError(
            f"Sheet '{sheet_name}': required canonical fields could not be mapped: {missing_req}. "
            f"Available columns: {list(df.columns)}"
        )
    out = df.rename(columns=rename).copy()
    out[RUN_YEAR] = year
    out[ROW_INDEX] = out.index
    schema_info = {
        "sheet": sheet_name,
        "year": year,
        "column_map": {k: v for k, v in rename.items()},
        "units": units,
        "missing_recommended": missing_rec,
    }
    return out, schema_info


def load_and_parse_runs(
    path: str | Path,
    run_years: list[int],
) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    """
    Load Excel, parse requested run sheets.
    Returns: (parsed_runs {year: df}, full_schema_report).
    """
    xl = pd.ExcelFile(path)
    report: dict[str, Any] = {
        "source": str(path),
        "sheets_found": xl.sheet_names,
        "run_years_requested": run_years,
        "runs": {},
        "errors": [],
    }
    result = {}
    for year in run_years:
        sheet = str(year)
        if sheet not in xl.sheet_names:
            report["errors"].append(f"Sheet '{sheet}' not found")
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        parsed, info = parse_run(df, year, sheet)
        result[year] = parsed
        report["runs"][str(year)] = info
    return result, report


def write_schema_report(report: dict[str, Any], out_path: Path) -> None:
    """Write schema_report.json to output dir."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)


# Legacy aliases for backward compat during refactor
DISTANCE_LEGACY = DISTANCE
RELATIVE_POSITION_LEGACY = "relative_position"
