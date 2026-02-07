"""
Weld-anchored alignment: girth welds as hard anchors, segment-wise linear mapping.

- Weld detection: "weld", "gw", "girth" (case-insensitive); fallback to joint boundaries
- Explicit scale/offset per segment with degenerate_segment + monotonic flags
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from config import SEGMENT_LENGTH_MIN_M
from src.normalize import DISTANCE, DISTANCE_CORRECTED, RELATIVE_POSITION
from src.schema import FEATURE_TYPE, JOINT_NUMBER

# Use normalize column names
RUN_YEAR = "run_year"
WELD_UID = "weld_uid"
WELD_INDEX = "weld_index"
SEGMENT_ID = "segment_id"

# Weld event patterns (case-insensitive)
WELD_PATTERNS = ["weld", "gw", "girth"]


def is_weld(event: str) -> bool:
    """Detect explicit weld markers."""
    if pd.isna(event):
        return False
    s = str(event).strip().lower()
    return any(p in s for p in WELD_PATTERNS)


def get_welds_explicit(df: pd.DataFrame, year: int) -> Optional[pd.DataFrame]:
    """Extract welds from explicit weld events. Returns None if none found."""
    if FEATURE_TYPE not in df.columns:
        return None
    mask = df[FEATURE_TYPE].apply(is_weld)
    if not mask.any():
        return None
    w = df[mask].copy()
    dist_col = DISTANCE if DISTANCE in df.columns else "distance_raw_m"
    if dist_col not in w.columns:
        return None
    w = w.dropna(subset=[dist_col])
    if len(w) == 0:
        return None
    w = w.sort_values(dist_col).reset_index(drop=True)
    w[WELD_INDEX] = w.index
    w["distance_raw_m"] = w[dist_col]
    return w


def get_welds_from_joints(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Fallback: derive weld positions from joint boundaries.
    Uses joint_number ordering and dist_to_us_weld where available.
    """
    dist_col = DISTANCE if DISTANCE in df.columns else "distance_raw_m"
    if JOINT_NUMBER not in df.columns or dist_col not in df.columns:
        return pd.DataFrame()
    # Get unique joints with min distance per joint (upstream boundary)
    grp = df.dropna(subset=[JOINT_NUMBER]).groupby(JOINT_NUMBER)[dist_col].min().sort_values()
    if len(grp) == 0:
        return pd.DataFrame()
    rows = []
    for i, (jn, d) in enumerate(grp.items()):
        rows.append({
            WELD_UID: f"W{i}",
            WELD_INDEX: i,
            "distance_raw_m": d,
            JOINT_NUMBER: jn,
        })
    return pd.DataFrame(rows)


def get_welds(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Get weld table: prefer explicit markers, else joint boundaries."""
    w = get_welds_explicit(df, year)
    if w is not None and len(w) > 0:
        w[WELD_UID] = [f"W{i}" for i in range(len(w))]
        return w
    w = get_welds_from_joints(df, year)
    if len(w) > 0:
        return w
    return pd.DataFrame()


def build_weld_map(
    welds_prev: pd.DataFrame,
    welds_later: pd.DataFrame,
    prev_year: int,
    later_year: int,
) -> pd.DataFrame:
    """
    Build WeldMap_{later}_to_{prev} with:
    weld_uid_prev, weld_uid_later, distance_prev_m, distance_later_m,
    segment_id, scale, offset, segment_length_prev_m, segment_length_later_m,
    align_flag (degenerate_segment when segment length < min)
    """
    d_col = "distance_raw_m"
    n = min(len(welds_prev), len(welds_later))
    rows = []
    for i in range(n):
        dp_i = welds_prev[d_col].iloc[i]
        dp_i1 = welds_prev[d_col].iloc[i + 1] if i + 1 < len(welds_prev) else dp_i
        dl_i = welds_later[d_col].iloc[i]
        dl_i1 = welds_later[d_col].iloc[i + 1] if i + 1 < len(welds_later) else dl_i
        seg_len_prev = dp_i1 - dp_i
        seg_len_later = dl_i1 - dl_i
        degenerate = abs(seg_len_later) < SEGMENT_LENGTH_MIN_M
        if degenerate:
            scale, offset = 1.0, 0.0
            align_flag = "degenerate_segment"
        else:
            scale = seg_len_prev / seg_len_later
            offset = dp_i - scale * dl_i
            align_flag = ""
        uid_prev = welds_prev[WELD_UID].iloc[i] if WELD_UID in welds_prev.columns else f"W{i}"
        uid_lat = welds_later[WELD_UID].iloc[i] if WELD_UID in welds_later.columns else f"W{i}"
        rows.append({
            "weld_uid_prev": uid_prev,
            "weld_uid_later": uid_lat,
            "distance_prev_m": dp_i,
            "distance_later_m": dl_i,
            "segment_id": i,
            "scale": scale,
            "offset": offset,
            "segment_length_prev_m": seg_len_prev,
            "segment_length_later_m": seg_len_later,
            "align_flag": align_flag,
        })
    return pd.DataFrame(rows)


def _segment_transform(
    d_later_lo: float, d_later_hi: float,
    d_prev_lo: float, d_prev_hi: float,
    d_raw: float,
) -> tuple[float, str]:
    """
    scale = (d_prev[i+1]-d_prev[i]) / (d_later[i+1]-d_later[i])
    offset = d_prev[i] - scale * d_later[i]
    distance_corrected_m = scale * distance_raw_m + offset
    Returns (corrected, align_flag)
    """
    seg_len_later = d_later_hi - d_later_lo
    seg_len_prev = d_prev_hi - d_prev_lo
    if abs(seg_len_later) < SEGMENT_LENGTH_MIN_M:
        return d_raw, "degenerate_segment"
    scale = seg_len_prev / seg_len_later
    offset = d_prev_lo - scale * d_later_lo
    corrected = scale * d_raw + offset
    return corrected, ""


def align_run_to_reference(
    df: pd.DataFrame,
    welds_prev: pd.DataFrame,
    welds_later: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Align later run distances to previous run coordinate system.
    Returns (aligned_df, alignment_flags for monotonic violations).
    """
    d_prev = welds_prev["distance_raw_m"].values
    d_later = welds_later["distance_raw_m"].values
    dist_col = DISTANCE if DISTANCE in df.columns else "distance_raw_m"
    d_raw = df[dist_col].values
    n = min(len(d_prev), len(d_later))
    if n < 2:
        out = df.copy()
        out[DISTANCE_CORRECTED] = df[dist_col].copy()
        return out, []
    corrected = []
    flags = []
    prev_c = None
    for idx, d in enumerate(d_raw):
        if pd.isna(d):
            corrected.append(float("nan"))
            continue
        seg_idx = 0
        for i in range(n - 1):
            if d_later[i] <= d <= d_later[i + 1]:
                seg_idx = i
                break
            if d < d_later[0]:
                seg_idx = 0
                break
            if d > d_later[n - 1]:
                seg_idx = n - 2
                break
        c, flag = _segment_transform(
            d_later[seg_idx], d_later[seg_idx + 1],
            d_prev[seg_idx], d_prev[seg_idx + 1],
            d,
        )
        corrected.append(c)
        if prev_c is not None and not pd.isna(prev_c) and not pd.isna(c) and c < prev_c - 1e-9:
            flags.append({"row": idx, "reason": "monotonic_violation", "prev": prev_c, "curr": c})
        prev_c = c
    out = df.copy()
    out[DISTANCE_CORRECTED] = corrected
    return out, flags


def align_two_runs(
    normalized: dict[int, pd.DataFrame],
    prev_year: int,
    later_year: int,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, list]:
    """
    MVP: Align exactly two runs. prev = reference, later = aligned to prev.
    Returns: (aligned_dfs, weld_map, alignment_flags)
    """
    if prev_year not in normalized or later_year not in normalized:
        raise ValueError(f"Runs {prev_year} and {later_year} must exist in normalized data")
    welds_prev = get_welds(normalized[prev_year], prev_year)
    welds_later = get_welds(normalized[later_year], later_year)
    if len(welds_prev) < 2 or len(welds_later) < 2:
        aligned = {prev_year: normalized[prev_year].copy(), later_year: normalized[later_year].copy()}
        aligned[prev_year][DISTANCE_CORRECTED] = normalized[prev_year][DISTANCE]
        aligned[later_year][DISTANCE_CORRECTED] = normalized[later_year][DISTANCE]
        weld_map = pd.DataFrame()
        return aligned, weld_map, []
    weld_map = build_weld_map(welds_prev, welds_later, prev_year, later_year)
    aligned_prev = normalized[prev_year].copy()
    aligned_prev[DISTANCE_CORRECTED] = normalized[prev_year][DISTANCE]
    aligned_later, flags = align_run_to_reference(
        normalized[later_year], welds_prev, welds_later
    )
    return {prev_year: aligned_prev, later_year: aligned_later}, weld_map, flags


def align_all(
    normalized: dict[int, pd.DataFrame],
) -> tuple[dict[int, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Multi-run: chain alignments 2015->2007, 2022->2015, then 2022->2007."""
    welds = {y: get_welds(df, y) for y, df in normalized.items()}
    for y in welds:
        w = welds[y]
        if len(w) > 0 and WELD_UID not in w.columns:
            w = w.copy()
            w[WELD_UID] = [f"W{i}" for i in range(len(w))]
            welds[y] = w
    weld_maps = {}
    aligned = {}
    # 2007 baseline
    if 2007 in normalized:
        aligned[2007] = normalized[2007].copy()
        aligned[2007][DISTANCE_CORRECTED] = normalized[2007][DISTANCE]
    # 2015 -> 2007
    if 2007 in welds and 2015 in welds and len(welds[2007]) >= 2 and len(welds[2015]) >= 2:
        weld_maps["WeldMap_2015_to_2007"] = build_weld_map(welds[2007], welds[2015], 2007, 2015)
        aligned[2015], _ = align_run_to_reference(normalized[2015], welds[2007], welds[2015])
    elif 2015 in normalized:
        aligned[2015] = normalized[2015].copy()
        aligned[2015][DISTANCE_CORRECTED] = normalized[2015][DISTANCE]
    # 2022 -> 2015
    if 2015 in welds and 2022 in welds and len(welds[2015]) >= 2 and len(welds[2022]) >= 2:
        weld_maps["WeldMap_2022_to_2015"] = build_weld_map(welds[2015], welds[2022], 2015, 2022)
        aligned[2022], _ = align_run_to_reference(normalized[2022], welds[2015], welds[2022])
        # Chain 2022 -> 2007
        if 2007 in welds and len(welds[2007]) >= 2 and len(welds[2015]) >= 2:
            d07 = welds[2007]["distance_raw_m"].values if "distance_raw_m" in welds[2007].columns else welds[2007].iloc[:, 0].values
            d15 = welds[2015]["distance_raw_m"].values if "distance_raw_m" in welds[2015].columns else welds[2015].iloc[:, 0].values
            n = min(len(d07), len(d15))
            if n >= 2:
                dc = aligned[2022][DISTANCE_CORRECTED].values
                out = []
                for d in dc:
                    if pd.isna(d):
                        out.append(float("nan"))
                        continue
                    seg = 0
                    for i in range(n - 1):
                        if d15[i] <= d <= d15[i + 1]:
                            seg = i
                            break
                        if d < d15[0]: seg = 0; break
                        if d > d15[n - 1]: seg = n - 2; break
                    c, _ = _segment_transform(d15[seg], d15[seg + 1], d07[seg], d07[seg + 1], d)
                    out.append(c)
                aligned[2022] = aligned[2022].copy()
                aligned[2022][DISTANCE_CORRECTED] = out
    elif 2022 in normalized:
        aligned[2022] = normalized[2022].copy()
        aligned[2022][DISTANCE_CORRECTED] = normalized[2022][DISTANCE]
    return aligned, weld_maps
