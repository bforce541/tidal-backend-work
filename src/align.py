"""Weld-anchored alignment: girth welds as hard anchors, segment-wise linear mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.schema import DISTANCE, FEATURE_TYPE, JOINT_NUMBER, RELATIVE_POSITION, RUN_YEAR

# Girth weld event names (case-insensitive patterns)
WELD_PATTERNS = ["girth weld", "girthweld", "girth weld anomaly"]


def is_weld(event: str) -> bool:
    """Check if event is a girth weld."""
    if pd.isna(event):
        return False
    s = str(event).strip().lower()
    return any(p in s for p in WELD_PATTERNS)


def get_welds(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Extract girth welds with distance, ordered by position."""
    mask = df[FEATURE_TYPE].apply(is_weld)
    w = df[mask][[DISTANCE, JOINT_NUMBER, RELATIVE_POSITION, FEATURE_TYPE]].copy()
    w = w.dropna(subset=[DISTANCE])
    w = w.sort_values(DISTANCE).reset_index(drop=True)
    w[RUN_YEAR] = year
    w["weld_idx"] = w.index
    return w


def build_weld_map(welds_from: pd.DataFrame, welds_to: pd.DataFrame) -> pd.DataFrame:
    """
    Build weld correspondence: for each weld in 'to' run, find matched weld in 'from' run.
    Uses ordered position: 1st weld <-> 1st weld, etc.
    """
    from_year = welds_from[RUN_YEAR].iloc[0]
    to_year = welds_to[RUN_YEAR].iloc[0]
    n = min(len(welds_from), len(welds_to))
    rows = []
    for i in range(n):
        rows.append({
            f"weld_idx_{to_year}": i,
            f"weld_idx_{from_year}": i,
            f"distance_{to_year}": welds_to[DISTANCE].iloc[i],
            f"distance_{from_year}": welds_from[DISTANCE].iloc[i],
        })
    return pd.DataFrame(rows)


def _segment_transform(
    d_from_lo: float, d_from_hi: float,
    d_to_lo: float, d_to_hi: float,
    d_raw: float
) -> float:
    """
    Linear map: [d_from_lo, d_from_hi] -> [d_to_lo, d_to_hi].
    Maps d_raw (in 'from' space) to corrected position in 'to' space.
    """
    if d_from_hi <= d_from_lo:
        return d_raw
    t = (d_raw - d_from_lo) / (d_from_hi - d_from_lo)
    t = max(0, min(1, t))
    return d_to_lo + t * (d_to_hi - d_to_lo)


def align_run_to_reference(
    df: pd.DataFrame,
    welds_ref: pd.DataFrame,
    welds_run: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align a run's distances to the reference run's coordinate system.
    Uses segment-wise linear stretch/compression between matched welds.
    """
    year = df[RUN_YEAR].iloc[0]
    ref_year = welds_ref[RUN_YEAR].iloc[0]
    run_year = welds_run[RUN_YEAR].iloc[0]

    d_ref = welds_ref[DISTANCE].values
    d_run = welds_run[DISTANCE].values
    n = min(len(d_ref), len(d_run))
    if n < 2:
        # Not enough welds; use identity
        df = df.copy()
        df["distance_corrected"] = df[DISTANCE].copy()
        return df

    result = df.copy()
    dist_raw = df[DISTANCE].values
    corrected = []

    for d in dist_raw:
        if pd.isna(d):
            corrected.append(float("nan"))
            continue
        # Find segment: between weld i and i+1
        seg_idx = -1
        for i in range(n - 1):
            if d_run[i] <= d <= d_run[i + 1]:
                seg_idx = i
                break
            if d < d_run[0]:
                seg_idx = -1  # before first weld
                break
            if d > d_run[n - 1]:
                seg_idx = n - 2  # after last weld
                break
        if seg_idx < 0:
            # Extrapolate from first segment
            seg_idx = 0
        i = seg_idx
        d_from_lo, d_from_hi = d_run[i], d_run[i + 1]
        d_to_lo, d_to_hi = d_ref[i], d_ref[i + 1]
        c = _segment_transform(d_from_lo, d_from_hi, d_to_lo, d_to_hi, d)
        corrected.append(c)
    result["distance_corrected"] = corrected
    return result


def align_all(
    normalized: dict[int, pd.DataFrame],
) -> tuple[dict[int, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Align 2015 -> 2007 and 2022 -> 2015 (then 2022 to 2007 via chaining).
    Returns: (aligned_dfs, weld_maps)
    """
    welds = {year: get_welds(df, year) for year, df in normalized.items()}
    weld_maps = {}
    aligned = {}

    # 2015 -> 2007
    if 2007 in welds and 2015 in welds:
        weld_maps["WeldMap_2015_to_2007"] = build_weld_map(welds[2007], welds[2015])
        aligned[2015] = align_run_to_reference(normalized[2015], welds[2007], welds[2015])
    else:
        aligned[2015] = normalized[2015].copy()
        aligned[2015]["distance_corrected"] = normalized[2015][DISTANCE]

    # 2022 -> 2015
    if 2015 in welds and 2022 in welds:
        weld_maps["WeldMap_2022_to_2015"] = build_weld_map(welds[2015], welds[2022])
        aligned[2022] = align_run_to_reference(normalized[2022], welds[2015], welds[2022])
    else:
        aligned[2022] = normalized[2022].copy()
        aligned[2022]["distance_corrected"] = normalized[2022][DISTANCE]

    # 2007 baseline: corrected = raw (reference)
    aligned[2007] = normalized[2007].copy()
    aligned[2007]["distance_corrected"] = normalized[2007][DISTANCE]

    # Chain 2022 -> 2007: first 2022->2015, then map 2015 coords to 2007
    if 2007 in welds and 2015 in welds and 2022 in welds:
        # 2022 distances are in 2015 space after align_run_to_reference(2022, 2015, 2022)
        # We need 2022 in 2007 space. Re-align: treat 2015 as reference, 2022 already in 2015 space
        # Actually: align_run_to_reference(df_2022, welds_2015, welds_2022) puts df_2022 distances
        # into 2015's coordinate system. So aligned[2022].distance_corrected is in 2015 coords.
        # To get 2022 in 2007 coords: we need to map from 2015 coords to 2007.
        # aligned[2015].distance_corrected is in 2007 coords. So we need a transform 2015_pos -> 2007_pos.
        # That transform is implicitly: segment between welds in 2015 maps to segment in 2007.
        # So: for each row in 2022, distance_corrected is in 2015. Apply 2015->2007 transform.
        df_2022 = aligned[2022]
        df_2015_aligned = aligned[2015]
        # Build 2015->2007 transform from weld positions
        w07 = welds[2007]
        w15 = welds[2015]
        d07 = w07[DISTANCE].values
        d15 = w15[DISTANCE].values
        n = min(len(d07), len(d15))
        if n >= 2:
            d_2022_in_15 = df_2022["distance_corrected"].values
            d_2022_in_07 = []
            for d in d_2022_in_15:
                if pd.isna(d):
                    d_2022_in_07.append(float("nan"))
                    continue
                seg_idx = 0
                for i in range(n - 1):
                    if d15[i] <= d <= d15[i + 1]:
                        seg_idx = i
                        break
                    if d < d15[0]:
                        seg_idx = 0
                        break
                    if d > d15[n - 1]:
                        seg_idx = n - 2
                        break
                i = seg_idx
                c = _segment_transform(d15[i], d15[i + 1], d07[i], d07[i + 1], d)
                d_2022_in_07.append(c)
            aligned[2022] = aligned[2022].copy()
            aligned[2022]["distance_corrected"] = d_2022_in_07

    return aligned, weld_maps
