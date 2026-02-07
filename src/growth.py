"""
Growth calculation: depth/length/width rates for matched pairs only.

- depth_rate, length_rate, width_rate (%/yr or mm/yr)
- growth_flag: outlier_rate when depth_rate > MAX_DEPTH_RATE_OUTLIER
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import MAX_DEPTH_RATE_OUTLIER
from src.normalize import DEPTH_PERCENT, LENGTH_MM, WIDTH_MM


def growth_rate(v_old: float, v_new: float, years: float) -> Optional[float]:
    """(v_new - v_old) / years. Returns None if invalid."""
    if pd.isna(v_old) or pd.isna(v_new) or years <= 0:
        return None
    return (float(v_new) - float(v_old)) / years


def compute_growth_for_matches(
    matches: pd.DataFrame,
    prev_year: int,
    later_year: int,
    max_depth_rate: float = MAX_DEPTH_RATE_OUTLIER,
) -> pd.DataFrame:
    """
    For each matched pair, compute depth_rate, length_rate, width_rate.
    Add growth_flag = "outlier_rate" when |depth_rate| > max_depth_rate.
    """
    years = later_year - prev_year
    if years <= 0:
        return matches.copy()
    out = matches.copy()
    out["years"] = years
    out["depth_rate"] = out.apply(
        lambda r: growth_rate(r["prev_depth_percent"], r["later_depth_percent"], years),
        axis=1,
    )
    out["length_rate"] = out.apply(
        lambda r: growth_rate(r["prev_length_mm"], r["later_length_mm"], years),
        axis=1,
    )
    out["width_rate"] = out.apply(
        lambda r: growth_rate(r["prev_width_mm"], r["later_width_mm"], years),
        axis=1,
    )
    out["growth_flag"] = out["depth_rate"].apply(
        lambda r: "outlier_rate" if r is not None and abs(r) > max_depth_rate else ""
    )
    return out
