"""Normalize ILI data: distances -> m, clock -> degrees, depth -> % wall, length/width -> mm."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import FT_TO_M, IN_TO_MM
from src.schema import (
    CLOCK_POSITION,
    DEPTH_PERCENT,
    DISTANCE,
    FEATURE_TYPE,
    JOINT_NUMBER,
    LENGTH_MM,
    RELATIVE_POSITION,
    RUN_YEAR,
    ROW_INDEX,
    WIDTH_MM,
    ParsedRun,
)

logger = logging.getLogger(__name__)


def _clock_to_degrees(v: str | float) -> Optional[float]:
    """Convert clock (e.g. '09:00', '12:00') to degrees 0-360.
    12 o'clock = 0, 3 o'clock = 90, 6 = 180, 9 = 270.
    """
    if pd.isna(v):
        return None
    if isinstance(v, (int, float)):
        # Assume already degrees if numeric
        return float(v) if 0 <= v <= 360 else None
    s = str(v).strip()
    if not s:
        return None
    # Parse hh:mm or hh:mm:ss
    parts = s.replace(":", " ").split()
    try:
        h = int(parts[0]) if parts else 0
        m = int(parts[1]) if len(parts) > 1 else 0
        # 12 o'clock = 0, 1 = 30, ... 11 = 330
        deg = (h % 12) * 30 + m / 2.0
        return deg % 360
    except (ValueError, IndexError):
        return None


def _ft_to_m(x: float) -> float:
    return float(x) * FT_TO_M if pd.notna(x) else float("nan")


def _in_to_mm(x: float) -> float:
    return float(x) * IN_TO_MM if pd.notna(x) else float("nan")


def normalize_run(parsed: ParsedRun) -> pd.DataFrame:
    """Normalize a parsed run to canonical units."""
    df = parsed.df.copy()

    # Distance (log dist / wheel count) -> meters
    if DISTANCE in df.columns:
        df[DISTANCE] = df[DISTANCE].apply(_ft_to_m)
    else:
        logger.warning(f"Run {parsed.year}: no distance column")

    # Relative position (to u/s weld) -> meters
    if RELATIVE_POSITION in df.columns:
        df[RELATIVE_POSITION] = df[RELATIVE_POSITION].apply(_ft_to_m)
    else:
        df[RELATIVE_POSITION] = float("nan")

    # Clock -> degrees
    if "clock_raw" in df.columns:
        df[CLOCK_POSITION] = df["clock_raw"].apply(_clock_to_degrees)
        df = df.drop(columns=["clock_raw"], errors="ignore")
    else:
        df[CLOCK_POSITION] = float("nan")

    # Depth already % - ensure numeric
    if DEPTH_PERCENT in df.columns:
        df[DEPTH_PERCENT] = pd.to_numeric(df[DEPTH_PERCENT], errors="coerce")
    else:
        df[DEPTH_PERCENT] = float("nan")

    # Length, width: in -> mm
    if "length_in" in df.columns:
        df[LENGTH_MM] = df["length_in"].apply(_in_to_mm)
        df = df.drop(columns=["length_in"], errors="ignore")
    else:
        df[LENGTH_MM] = float("nan")
    if "width_in" in df.columns:
        df[WIDTH_MM] = df["width_in"].apply(_in_to_mm)
        df = df.drop(columns=["width_in"], errors="ignore")
    else:
        df[WIDTH_MM] = float("nan")

    return df


def normalize_all(parsed_runs: dict[int, ParsedRun]) -> dict[int, pd.DataFrame]:
    """Normalize all runs."""
    return {year: normalize_run(p) for year, p in parsed_runs.items()}
