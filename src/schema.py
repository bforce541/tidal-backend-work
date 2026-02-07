"""Column mapping and schema discovery for ILI Excel data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Canonical field names
DISTANCE = "distance"  # axial position, m (raw until alignment)
JOINT_NUMBER = "joint_number"
RELATIVE_POSITION = "relative_position"  # distance to u/s weld, m
FEATURE_TYPE = "feature_type"
CLOCK_POSITION = "clock_position"  # degrees 0-360
DEPTH_PERCENT = "depth_percent"
LENGTH_MM = "length_mm"
WIDTH_MM = "width_mm"
RUN_YEAR = "run_year"
ROW_INDEX = "row_index"

# Column mappings per sheet (Excel col -> canonical)
# 2007
MAP_2007 = {
    "J. no.": JOINT_NUMBER,
    "log dist. [ft]": DISTANCE,
    "to u/s w. [ft]": RELATIVE_POSITION,
    "event": FEATURE_TYPE,
    "depth [%]": DEPTH_PERCENT,
    "length [in]": "length_in",
    "width [in]": "width_in",
    "o'clock": "clock_raw",
}

# 2015
MAP_2015 = {
    "J. no.": JOINT_NUMBER,
    "Log Dist. [ft]": DISTANCE,
    "to u/s w. [ft]": RELATIVE_POSITION,
    "Event Description": FEATURE_TYPE,
    "Depth [%]": DEPTH_PERCENT,
    "Length [in]": "length_in",
    "Width [in]": "width_in",
    "O'clock": "clock_raw",
}

# 2022
MAP_2022 = {
    "Joint Number": JOINT_NUMBER,
    "ILI Wheel Count \n[ft.]": DISTANCE,
    "Distance to U/S GW \n[ft]": RELATIVE_POSITION,
    "Event Description": FEATURE_TYPE,
    "Metal Loss Depth \n[%]": DEPTH_PERCENT,
    "Length [in]": "length_in",
    "Width [in]": "width_in",
    "O'clock\n[hh:mm]": "clock_raw",
}

YEAR_TO_MAP = {2007: MAP_2007, 2015: MAP_2015, 2022: MAP_2022}


@dataclass
class ParsedRun:
    """Parsed raw data for one run (before normalization)."""

    year: int
    df: pd.DataFrame
    column_map: dict[str, str]


def load_excel(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load all sheets from Excel."""
    xl = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}


def parse_run(df: pd.DataFrame, year: int) -> ParsedRun:
    """Parse a run sheet into canonical column names (raw values)."""
    m = YEAR_TO_MAP[year]
    rename = {}
    for ex_col, canon in m.items():
        if ex_col in df.columns:
            rename[ex_col] = canon
    out = df.rename(columns=rename).copy()
    out[RUN_YEAR] = year
    out[ROW_INDEX] = out.index
    return ParsedRun(year=year, df=out, column_map=m)


def discover_schema(path: str | Path) -> dict[str, list[str]]:
    """Inspect Excel and return sheet -> columns mapping."""
    d = load_excel(path)
    return {name: list(df.columns) for name, df in d.items()}


def get_runs(path: str | Path) -> dict[int, ParsedRun]:
    """Load and parse 2007, 2015, 2022 run sheets."""
    data = load_excel(path)
    result = {}
    for year in [2007, 2015, 2022]:
        sheet = str(year)
        if sheet in data:
            result[year] = parse_run(data[sheet], year)
    return result
