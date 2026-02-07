"""
Anomaly matching: deterministic, explainable.

- Hard filters: distance, clock, feature type (family)
- Score: w_d*|Δd| + w_c*Δθ + w_dep*|Δdepth| + w_len*|Δlen| + w_wid*|Δwid| (lower = better)
- Ambiguous: top2 score diff < epsilon => do not match, record both candidates
- Output: ids, raw+corrected distances, deltas, score, reason flags
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    AXIAL_TOL_M,
    CLOCK_TOL_DEG,
    MATCH_SCORE_EPSILON,
    W_CLOCK,
    W_DEPTH,
    W_DIST,
    W_LENGTH,
    W_WIDTH,
)
from src.normalize import (
    CLOCK_POSITION,
    DEPTH_PERCENT,
    DISTANCE_CORRECTED,
    FEATURE_TYPE,
    LENGTH_MM,
    WIDTH_MM,
)
from src.schema import FEATURE_TYPE as FT

# Feature family mapping for type compatibility
FEATURE_FAMILIES = {
    "METAL_LOSS": [
        "metal loss", "corrosion", "external ml", "internal ml",
        "metal loss-manufacturing", "metal loss manufacturing",
        "metal loss manufacturing anomaly", "metal loss anomaly",
        "seam weld manufacturing anomaly", "metal loss manufacturing anomaly",
        "seam weld anomaly", "metal loss depth",
    ],
    "CLUSTER": ["cluster"],
}
# Welds excluded from anomaly matching
WELD_FAMILY = ["weld", "gw", "girth", "girth weld", "girthweld"]
UNKNOWN = "UNKNOWN"


def to_feature_family(event: str) -> str:
    """Map event/feature_type to normalized family."""
    if pd.isna(event):
        return UNKNOWN
    s = str(event).strip().lower()
    for fam, patterns in FEATURE_FAMILIES.items():
        if any(p in s for p in patterns):
            return fam
    if any(p in s for p in WELD_FAMILY):
        return "WELD"  # excluded from matching
    return UNKNOWN


def is_anomaly(event: str) -> bool:
    """Exclude welds; anomalies are METAL_LOSS, CLUSTER, or UNKNOWN (allowed but penalized)."""
    fam = to_feature_family(event)
    return fam != "WELD" and fam in ("METAL_LOSS", "CLUSTER", UNKNOWN)


def _type_compatible(t_prev: str, t_later: str) -> bool:
    """Hard filter: same family (or UNKNOWN matches)."""
    a = to_feature_family(t_prev)
    b = to_feature_family(t_later)
    if a == "WELD" or b == "WELD":
        return False
    if a == UNKNOWN or b == UNKNOWN:
        return True  # allow but will be penalized in score
    return a == b


def _circular_clock_diff(a: float, b: float) -> float:
    """Circular angular difference in degrees (0-180)."""
    if pd.isna(a) or pd.isna(b):
        return 0
    d = abs((float(a) - float(b) + 180) % 360 - 180)
    return d


def score_match(
    r_prev: dict, r_later: dict,
    w_d: float = W_DIST, w_c: float = W_CLOCK,
    w_dep: float = W_DEPTH, w_len: float = W_LENGTH, w_wid: float = W_WIDTH,
) -> float:
    """
    score = w_d*|Δd| + w_c*Δθ + w_dep*|Δdepth| + w_len*|Δlen| + w_wid*|Δwid|
    Lower = better.
    """
    dd = abs(r_prev.get(DISTANCE_CORRECTED, 0) - r_later.get(DISTANCE_CORRECTED, 0))
    dtheta = _circular_clock_diff(
        r_prev.get(CLOCK_POSITION),
        r_later.get(CLOCK_POSITION),
    )
    dep_p = r_prev.get(DEPTH_PERCENT) or 0
    dep_l = r_later.get(DEPTH_PERCENT) or 0
    ddep = abs(dep_p - dep_l) if pd.notna(dep_p) and pd.notna(dep_l) else 0
    len_p = r_prev.get(LENGTH_MM) or 0
    len_l = r_later.get(LENGTH_MM) or 0
    dlen = abs(len_p - len_l) if pd.notna(len_p) and pd.notna(len_l) else 0
    wid_p = r_prev.get(WIDTH_MM) or 0
    wid_l = r_later.get(WIDTH_MM) or 0
    dwid = abs(wid_p - wid_l) if pd.notna(wid_p) and pd.notna(wid_l) else 0
    # UNKNOWN penalty
    ft_col = FEATURE_TYPE if FEATURE_TYPE in r_prev else FT
    pen = 10.0 if to_feature_family(r_prev.get(ft_col, "")) == UNKNOWN or to_feature_family(r_later.get(ft_col, "")) == UNKNOWN else 0
    return w_d * dd + w_c * dtheta + w_dep * ddep + w_len * dlen + w_wid * dwid + pen


def _hard_filter(
    row_prev: dict, row_later: dict,
    axial_tol: float, clock_tol: float,
) -> bool:
    """Hard filters: distance, clock, type."""
    dd = abs(row_prev.get(DISTANCE_CORRECTED, 0) - row_later.get(DISTANCE_CORRECTED, 0))
    if dd > axial_tol:
        return False
    dtheta = _circular_clock_diff(
        row_prev.get(CLOCK_POSITION),
        row_later.get(CLOCK_POSITION),
    )
    if dtheta > clock_tol:
        return False
    ft = FEATURE_TYPE if FEATURE_TYPE in row_prev else FT
    if not _type_compatible(row_prev.get(ft), row_later.get(ft)):
        return False
    return True


def match_anomalies(
    df_prev: pd.DataFrame,
    df_later: pd.DataFrame,
    prev_year: int,
    later_year: int,
    axial_tol: float = AXIAL_TOL_M,
    clock_tol: float = CLOCK_TOL_DEG,
    score_epsilon: float = MATCH_SCORE_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Match: for each anomaly in later run, find candidates in previous run.
    Returns: Matches, New_{later}_vs_{prev}, Missing_{prev}_vs_{later}, Ambiguous.
    """
    ft_col = FEATURE_TYPE if FEATURE_TYPE in df_prev.columns else FT
    prev = df_prev[df_prev[ft_col].apply(is_anomaly)].copy()
    later = df_later[df_later[ft_col].apply(is_anomaly)].copy()
    prev = prev.reset_index(drop=True)
    later = later.reset_index(drop=True)

    dc_prev = prev[DISTANCE_CORRECTED].values

    matches = []
    ambiguous_list = []
    used_prev = set()
    used_later = set()

    for i, row_later in later.iterrows():
        d_later = row_later.get(DISTANCE_CORRECTED)
        if pd.isna(d_later):
            continue
        mask = (np.abs(dc_prev - d_later) <= axial_tol) & ~np.isnan(dc_prev)
        candidates = []
        for j in np.where(mask)[0]:
            if j in used_prev:
                continue
            row_prev = prev.iloc[j]
            if not _hard_filter(row_prev.to_dict(), row_later.to_dict(), axial_tol, clock_tol):
                continue
            s = score_match(row_prev.to_dict(), row_later.to_dict())
            candidates.append((s, int(j), row_prev))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        best_s, best_j, best_prev = candidates[0]
        if len(candidates) >= 2 and abs(candidates[1][0] - best_s) < score_epsilon:
            ambiguous_list.append({
                "later_idx": i,
                "prev_idx_candidate_1": best_j,
                "prev_idx_candidate_2": candidates[1][1],
                "score_1": best_s,
                "score_2": candidates[1][0],
                "later_row": row_later.to_dict(),
            })
            continue
        used_prev.add(best_j)
        used_later.add(i)
        pr, lr = best_prev.to_dict(), row_later.to_dict()
        dc_p = pr.get(DISTANCE_CORRECTED)
        dc_l = lr.get(DISTANCE_CORRECTED)
        matches.append({
            "prev_idx": best_j,
            "later_idx": i,
            "prev_year": prev_year,
            "later_year": later_year,
            "prev_distance_raw_m": pr.get("distance_raw_m"),
            "later_distance_raw_m": lr.get("distance_raw_m"),
            "prev_distance_corrected_m": dc_p,
            "later_distance_corrected_m": dc_l,
            "delta_distance_m": (dc_l - dc_p) if pd.notna(dc_p) and pd.notna(dc_l) else None,
            "prev_depth_percent": pr.get(DEPTH_PERCENT),
            "later_depth_percent": lr.get(DEPTH_PERCENT),
            "prev_length_mm": pr.get(LENGTH_MM),
            "later_length_mm": lr.get(LENGTH_MM),
            "prev_width_mm": pr.get(WIDTH_MM),
            "later_width_mm": lr.get(WIDTH_MM),
            "score": best_s,
            "reason_flags": "",
        })

    df_matches = pd.DataFrame(matches)
    new_rows = later.loc[[i for i in range(len(later)) if i not in used_later]]
    missing_rows = prev.loc[[j for j in range(len(prev)) if j not in used_prev]]
    df_ambiguous = pd.DataFrame(ambiguous_list) if ambiguous_list else pd.DataFrame()
    return df_matches, new_rows, missing_rows, df_ambiguous


def run_matching_two(
    aligned: dict[int, pd.DataFrame],
    prev_year: int,
    later_year: int,
) -> dict[str, pd.DataFrame]:
    """MVP: matching for exactly two runs."""
    m, new, miss, amb = match_anomalies(
        aligned[prev_year], aligned[later_year],
        prev_year, later_year,
    )
    return {
        f"Matches_{prev_year}_{later_year}": m,
        f"New_{later_year}_vs_{prev_year}": new,
        f"Missing_{prev_year}_vs_{later_year}": miss,
        f"Ambiguous_{prev_year}_{later_year}": amb,
    }
