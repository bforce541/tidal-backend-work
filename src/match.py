"""Anomaly matching: match anomalies across runs with deterministic scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import AXIAL_DISTANCE_TOLERANCE_M, CLOCK_POSITION_TOLERANCE_DEG, DEPTH_SIMILARITY_TOLERANCE
from src.schema import (
    CLOCK_POSITION,
    DEPTH_PERCENT,
    FEATURE_TYPE,
    LENGTH_MM,
    WIDTH_MM,
)

# Anomaly event patterns (metal loss, cluster, etc.)
ANOMALY_PATTERNS = [
    "metal loss",
    "metal loss-manufacturing",
    "metal loss manufacturing",
    "metal loss manufacturing anomaly",
    "cluster",
    "cluster ",
    "metal loss anomaly",
    "seam weld manufacturing anomaly",
    "metal loss manufacturing anomaly",
    "seam weld anomaly",
]


def is_anomaly(event: str) -> bool:
    """Check if event is an anomaly (metal loss, cluster, etc.)."""
    if pd.isna(event):
        return False
    s = str(event).strip().lower()
    return any(p in s for p in ANOMALY_PATTERNS)


def _deg_diff(a: float, b: float) -> float:
    """Angular difference in degrees (0-180)."""
    if pd.isna(a) or pd.isna(b):
        return 0
    d = abs((a - b + 180) % 360 - 180)
    return d


def _depth_sim(d1: float, d2: float) -> float:
    """Depth similarity: 1 = same, 0 = very different. Handles missing."""
    if pd.isna(d1) and pd.isna(d2):
        return 1.0
    if pd.isna(d1) or pd.isna(d2):
        return 0.5
    if d1 == 0 and d2 == 0:
        return 1.0
    if d1 == 0 or d2 == 0:
        return 0.3
    r = min(d1, d2) / max(d1, d2)
    return r


def score_match(
    r_prev: dict,
    r_later: dict,
    axial_tol: float,
    clock_tol: float,
) -> float:
    """
    Deterministic score: higher = better match.
    Weighted: distance (closer better), clock (closer better), depth similarity.
    """
    d_axial = abs(r_prev.get("distance_corrected", 0) - r_later.get("distance_corrected", 0))
    if d_axial > axial_tol:
        return -1e9
    clock_p = r_prev.get(CLOCK_POSITION)
    clock_l = r_later.get(CLOCK_POSITION)
    d_clock = _deg_diff(clock_p if pd.notna(clock_p) else 0, clock_l if pd.notna(clock_l) else 0)
    if d_clock > clock_tol:
        return -1e9
    depth_sim = _depth_sim(r_prev.get(DEPTH_PERCENT), r_later.get(DEPTH_PERCENT))
    # Score: penalize distance and clock, reward depth similarity
    return -d_axial - 0.01 * d_clock + 10 * depth_sim


def match_anomalies(
    df_prev: pd.DataFrame,
    df_later: pd.DataFrame,
    axial_tol: float = AXIAL_DISTANCE_TOLERANCE_M,
    clock_tol: float = CLOCK_POSITION_TOLERANCE_DEG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match anomalies between previous and later run.
    Returns: Matches, New (in later), Missing (in prev), ambiguous resolved as best match.
    """
    prev = df_prev[df_prev[FEATURE_TYPE].apply(is_anomaly)].copy()
    later = df_later[df_later[FEATURE_TYPE].apply(is_anomaly)].copy()
    prev = prev.reset_index(drop=True)
    later = later.reset_index(drop=True)

    matches = []
    used_prev = set()
    used_later = set()

    for i, row_later in later.iterrows():
        candidates = []
        for j, row_prev in prev.iterrows():
            if j in used_prev:
                continue
            s = score_match(row_prev.to_dict(), row_later.to_dict(), axial_tol, clock_tol)
            if s > -1e8:
                candidates.append((s, j, row_prev))
        if not candidates:
            # New anomaly
            used_later.add(i)
            continue
        # Pick best
        candidates.sort(key=lambda x: -x[0])
        best_s, best_j, best_prev = candidates[0]
        used_prev.add(best_j)
        used_later.add(i)
        matches.append({
            "prev_idx": best_j,
            "later_idx": i,
            "prev_row": best_prev.to_dict(),
            "later_row": row_later.to_dict(),
        })

    # Build output tables
    match_rows = []
    for m in matches:
        pr, lr = m["prev_row"], m["later_row"]
        match_rows.append({
            "prev_idx": m["prev_idx"],
            "later_idx": m["later_idx"],
            "prev_distance_corrected": pr.get("distance_corrected"),
            "later_distance_corrected": lr.get("distance_corrected"),
            "prev_depth_percent": pr.get(DEPTH_PERCENT),
            "later_depth_percent": lr.get(DEPTH_PERCENT),
            "prev_length_mm": pr.get(LENGTH_MM),
            "later_length_mm": lr.get(LENGTH_MM),
            "prev_width_mm": pr.get(WIDTH_MM),
            "later_width_mm": lr.get(WIDTH_MM),
        })
    df_matches = pd.DataFrame(match_rows)

    new_rows = later.loc[[i for i in range(len(later)) if i not in used_later]]
    missing_rows = prev.loc[[j for j in range(len(prev)) if j not in used_prev]]

    return df_matches, new_rows, missing_rows


def run_matching(
    aligned: dict[int, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Run matching for 2007<->2015 and 2015<->2022."""
    result = {}
    for (y_prev, y_later) in [(2007, 2015), (2015, 2022)]:
        if y_prev not in aligned or y_later not in aligned:
            continue
        m, new, missing = match_anomalies(aligned[y_prev], aligned[y_later])
        result[f"Matches_{y_prev}_{y_later}"] = m
        result[f"New_{y_prev}_{y_later}"] = new
        result[f"Missing_{y_prev}_{y_later}"] = missing
    return result
