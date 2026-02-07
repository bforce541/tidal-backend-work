"""
Normalize ILI data: distances -> m, clock -> degrees, depth -> %, length/width -> mm.

Works with schema.py canonical output.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import FT_TO_M, IN_TO_MM
from src.schema import (
    CLOCK,
    DEPTH_PERCENT,
    DISTANCE_ALONG_LINE,
    DIST_TO_DS_WELD,
    DIST_TO_US_WELD,
    FEATURE_TYPE,
    JOINT_NUMBER,
    LENGTH_IN,
    WIDTH_IN,
)

# Output columns (SI units)
DISTANCE = "distance_raw_m"
RELATIVE_POSITION = "dist_to_us_weld_m"
DISTANCE_CORRECTED = "distance_corrected_m"
CLOCK_POSITION = "clock_position_deg"
LENGTH_MM = "length_mm"
WIDTH_MM = "width_mm"
RUN_YEAR = "run_year"
ROW_INDEX = "row_index"

logger = logging.getLogger(__name__)


def _clock_to_degrees(v: str | float) -> Optional[float]:
    """
    Convert clock (e.g. '09:00', '12:00') to degrees 0-360.
    12 o'clock = 0, 3 o'clock = 90, 6 = 180, 9 = 270.
    Robust: handles hh:mm, hh:mm:ss, numeric degrees.
    """
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if 0 <= f <= 360 else (f % 360)
    s = str(v).strip().replace("\n", " ").replace("\r", " ")
    if not s:
        return None
    parts = s.replace(":", " ").replace(".", " ").split()
    try:
        h = int(float(parts[0])) if parts else 0
        m = int(float(parts[1])) if len(parts) > 1 else 0
        deg = (h % 12) * 30 + m / 2.0
        return deg % 360
    except (ValueError, IndexError):
        return None


def _convert_ft_to_m(x: float) -> float:
    return float(x) * FT_TO_M if pd.notna(x) else float("nan")


def _convert_in_to_mm(x: float) -> float:
    return float(x) * IN_TO_MM if pd.notna(x) else float("nan")


def _convert_to_m(val: float, unit: str) -> float:
    """Convert distance to meters. Unit from header: ft, in, unknown->assume ft."""
    if pd.isna(val):
        return float("nan")
    if unit == "in":
        return float(val) * IN_TO_MM / 1000  # in -> m via mm
    return float(val) * FT_TO_M  # ft or unknown


def _convert_to_mm(val: float, unit: str) -> float:
    """Convert length/width to mm."""
    if pd.isna(val):
        return float("nan")
    if unit == "ft":
        return float(val) * FT_TO_M * 1000  # ft -> m -> mm
    return float(val) * IN_TO_MM  # in or unknown


def normalize_run(df: pd.DataFrame, year: int, units: Optional[dict] = None) -> pd.DataFrame:
    """
    Normalize parsed run to SI units.
    df: parsed from schema (canonical column names)
    units: optional {canonical_field: "ft"|"in"|"percent"}
    """
    out = df.copy()
    if units is None:
        units = {}

    # Distance along line -> meters
    if DISTANCE_ALONG_LINE in out.columns:
        u = units.get(DISTANCE_ALONG_LINE, "ft")
        out[DISTANCE] = out[DISTANCE_ALONG_LINE].apply(lambda x: _convert_to_m(x, u))
    else:
        out[DISTANCE] = float("nan")

    # Dist to u/s weld -> meters
    if DIST_TO_US_WELD in out.columns:
        u = units.get(DIST_TO_US_WELD, "ft")
        out[RELATIVE_POSITION] = out[DIST_TO_US_WELD].apply(lambda x: _convert_to_m(x, u))
    else:
        out[RELATIVE_POSITION] = float("nan")

    # Clock -> degrees
    if CLOCK in out.columns:
        out[CLOCK_POSITION] = out[CLOCK].apply(_clock_to_degrees)
        out = out.drop(columns=[CLOCK], errors="ignore")
    else:
        out[CLOCK_POSITION] = float("nan")

    # Depth -> percent (ensure numeric)
    if DEPTH_PERCENT in out.columns:
        out[DEPTH_PERCENT] = pd.to_numeric(out[DEPTH_PERCENT], errors="coerce")
    else:
        out[DEPTH_PERCENT] = float("nan")

    # Length, width -> mm
    if LENGTH_IN in out.columns:
        u = units.get(LENGTH_IN, "in")
        out[LENGTH_MM] = out[LENGTH_IN].apply(lambda x: _convert_to_mm(x, u))
        out = out.drop(columns=[LENGTH_IN], errors="ignore")
    else:
        out[LENGTH_MM] = float("nan")
    if WIDTH_IN in out.columns:
        u = units.get(WIDTH_IN, "in")
        out[WIDTH_MM] = out[WIDTH_IN].apply(lambda x: _convert_to_mm(x, u))
        out = out.drop(columns=[WIDTH_IN], errors="ignore")
    else:
        out[WIDTH_MM] = float("nan")

    return out


def normalize_all(
    parsed: dict[int, pd.DataFrame],
    schema_report: Optional[dict] = None,
) -> dict[int, pd.DataFrame]:
    """Normalize all parsed runs. Use schema_report for unit info."""
    units_by_year = {}
    if schema_report and "runs" in schema_report:
        for y, info in schema_report["runs"].items():
            units_by_year[int(y)] = info.get("units", {})
    result = {}
    for year, df in parsed.items():
        units = units_by_year.get(year, {})
        result[year] = normalize_run(df, year, units)
    return result
