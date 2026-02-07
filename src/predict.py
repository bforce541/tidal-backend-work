"""5-year projection with linear extrapolation and risk flagging."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from config import MAX_DEPTH_RATE_PER_YEAR, PREDICTION_HORIZON_YEARS, RISK_DEPTH_THRESHOLDS
from src.growth import RATE_SOURCE_07_15, RATE_SOURCE_15_22
from src.schema import DEPTH_PERCENT

LAST_OBSERVED_YEAR = 2022



def project_depth(
    depth_current: float,
    rate: Optional[float],
    years: float,
    cap: float = 100.0,
    floor: float = 0.0,
    conservative_clamp: bool = True,
) -> float:
    """
    Linear extrapolation: depth_current + rate * years.
    Conservative clamp: limit rate to MAX_DEPTH_RATE_PER_YEAR if extrapolating.
    """
    if pd.isna(depth_current):
        return float("nan")
    if rate is None or pd.isna(rate):
        return depth_current
    if conservative_clamp and abs(rate) > MAX_DEPTH_RATE_PER_YEAR:
        rate = MAX_DEPTH_RATE_PER_YEAR if rate > 0 else -MAX_DEPTH_RATE_PER_YEAR
    pred = depth_current + rate * years
    return max(floor, min(cap, pred))


def flag_risk(depth: float, thresholds: List[float]) -> Optional[str]:
    """Flag if depth exceeds any threshold."""
    if pd.isna(depth):
        return None
    exceeded = [t for t in sorted(thresholds, reverse=True) if depth >= t]
    if not exceeded:
        return None
    return f"depth >= {exceeded[0]}%"


def _get_most_recent_valid_rate(row: dict) -> Optional[float]:
    """
    Use most recent valid growth rate per rate_source_interval.
    Prefer 15_22, fallback to 07_15.
    """
    interval = row.get("rate_source_interval")
    if interval == RATE_SOURCE_15_22 and pd.notna(row.get("depth_rate_15_22")):
        return row["depth_rate_15_22"]
    if interval == RATE_SOURCE_07_15 and pd.notna(row.get("depth_rate_07_15")):
        return row["depth_rate_07_15"]
    if pd.notna(row.get("depth_rate_15_22")):
        return row["depth_rate_15_22"]
    if pd.notna(row.get("depth_rate_07_15")):
        return row["depth_rate_07_15"]
    return None


def _get_current_depth(row: dict) -> float:
    """Get current (last observed) depth for projection."""
    last = row.get("last_seen_year")
    if last == 2022 and pd.notna(row.get("depth_2022")):
        return row["depth_2022"]
    if last == 2015 and pd.notna(row.get("depth_2015")):
        return row["depth_2015"]
    if pd.notna(row.get("depth_2022")):
        return row["depth_2022"]
    if pd.notna(row.get("depth_2015")):
        return row["depth_2015"]
    return float("nan")


def predict_tracks(
    tracks: pd.DataFrame,
    horizon_years: float = PREDICTION_HORIZON_YEARS,
    risk_thresholds: List[float] = None,
    conservative_clamp: bool = True,
) -> pd.DataFrame:
    """
    Project each tracked anomaly forward.
    Uses most recent valid growth rate (per rate_source_interval) with conservative clamp.
    """
    if risk_thresholds is None:
        risk_thresholds = RISK_DEPTH_THRESHOLDS
    pred_year = LAST_OBSERVED_YEAR + horizon_years
    out = tracks.copy()
    out["depth_rate_used"] = out.apply(_get_most_recent_valid_rate, axis=1)
    out["depth_current"] = out.apply(_get_current_depth, axis=1)
    out["depth_predicted"] = out.apply(
        lambda r: project_depth(
            r["depth_current"],
            r["depth_rate_used"],
            horizon_years,
            conservative_clamp=conservative_clamp,
        ),
        axis=1,
    )
    out["prediction_year"] = pred_year
    out["risk_flag"] = out["depth_predicted"].apply(lambda d: flag_risk(d, risk_thresholds))
    return out
