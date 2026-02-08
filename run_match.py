#!/usr/bin/env python3
"""
Broad anomaly matching between two runs (e.g. 2015 -> 2022).

Reads anomalies_clean.csv, finds candidates within 30m, scores them,
outputs top-k candidates, best match per later anomaly, unmatched, and summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Tolerances and weights (DO NOT over-filter)
DISTANCE_TOL_M = 30.0
W_DIST = 1.0
W_CLOCK = 0.2
W_DEPTH = 0.5
W_LENGTH = 0.05
W_WIDTH = 0.05
TYPE_PENALTY_SAME = 0
TYPE_PENALTY_OTHER = 10
TYPE_PENALTY_DIFF = 25
# Reason/needs_review thresholds (labeling only)
REASON_HIGH_SCORE = 40.0
REASON_LOW_GAP_THRESHOLD = 5.0
REASON_AMBIGUOUS_GOOD_MAX_SCORE = 20.0

# Alignment
MIN_ANCHORS = 30
ROLLING_MEDIAN_WINDOW = 25


def _build_anchor_pairs(
    all_runs_path: Path,
    prev_year: int,
    later_year: int,
) -> tuple[pd.DataFrame | None, bool]:
    """
    Load all_runs_clean, get weld (or fallback) anchors, inner-join on joint_number.
    Returns (anchors_df with joint_number, prev_dist_m, later_dist_m) or (None, False) if not enough.
    """
    df = pd.read_csv(all_runs_path, low_memory=False)
    need = {"run_year", "joint_number", "distance_raw_m", "feature_type_norm"}
    if not need.issubset(df.columns):
        return None, False
    prev_df = df[df["run_year"] == prev_year].copy()
    later_df = df[df["run_year"] == later_year].copy()
    if prev_df.empty or later_df.empty:
        return None, False

    # 1) Try weld rows
    prev_weld = prev_df[prev_df["feature_type_norm"].astype(str).str.lower() == "weld"].dropna(subset=["joint_number", "distance_raw_m"])
    later_weld = later_df[later_df["feature_type_norm"].astype(str).str.lower() == "weld"].dropna(subset=["joint_number", "distance_raw_m"])
    if len(prev_weld) >= MIN_ANCHORS and len(later_weld) >= MIN_ANCHORS:
        prev_anch = prev_weld.groupby("joint_number", as_index=False)["distance_raw_m"].first()
        later_anch = later_weld.groupby("joint_number", as_index=False)["distance_raw_m"].first()
        prev_anch = prev_anch.rename(columns={"distance_raw_m": "prev_dist_m"})
        later_anch = later_anch.rename(columns={"distance_raw_m": "later_dist_m"})
        anchors = prev_anch.merge(later_anch, on="joint_number", how="inner")
        if len(anchors) >= MIN_ANCHORS:
            return anchors, True
    # 2) Fallback: all rows with joint_number + distance_raw_m
    prev_any = prev_df.dropna(subset=["joint_number", "distance_raw_m"]).groupby("joint_number", as_index=False)["distance_raw_m"].first()
    later_any = later_df.dropna(subset=["joint_number", "distance_raw_m"]).groupby("joint_number", as_index=False)["distance_raw_m"].first()
    prev_any = prev_any.rename(columns={"distance_raw_m": "prev_dist_m"})
    later_any = later_any.rename(columns={"distance_raw_m": "later_dist_m"})
    anchors = prev_any.merge(later_any, on="joint_number", how="inner")
    if len(anchors) < MIN_ANCHORS:
        return None, False
    return anchors, True


def _fit_alignment(anchors: pd.DataFrame) -> tuple[callable, pd.DataFrame, dict]:
    """
    Sort by prev_dist_m, compute offset = later - prev, smooth with rolling median.
    Return (offset_interp_func, anchors_with_offset_smoothed, metadata).
    interp_func(prev_dist_m) returns offset (clamped at ends).
    """
    a = anchors.sort_values("prev_dist_m").reset_index(drop=True)
    a["offset_m"] = a["later_dist_m"] - a["prev_dist_m"]
    a["offset_smoothed_m"] = a["offset_m"].rolling(window=ROLLING_MEDIAN_WINDOW, min_periods=1, center=True).median()
    # Fill any leading/trailing NaN from rolling with first/last valid
    a["offset_smoothed_m"] = a["offset_smoothed_m"].ffill().bfill()
    if a["offset_smoothed_m"].isna().any():
        a["offset_smoothed_m"] = a["offset_smoothed_m"].fillna(a["offset_m"])
    xp = a["prev_dist_m"].values.astype(float)
    fp = a["offset_smoothed_m"].values.astype(float)
    def offset_fn(prev_d: np.ndarray | float) -> np.ndarray | float:
        out = np.interp(np.atleast_1d(np.asarray(prev_d, dtype=float)), xp, fp)
        return out[0] if np.isscalar(prev_d) else out
    meta = {
        "anchor_count": len(a),
        "anchor_range_prev_m": {"min": float(xp.min()), "max": float(xp.max())},
        "offset_stats": {
            "min": float(fp.min()),
            "median": float(np.median(fp)),
            "max": float(fp.max()),
        },
        "method": "piecewise_linear_rolling_median",
    }
    return offset_fn, a, meta


def _circular_clock_diff_deg(a: pd.Series, b: pd.Series) -> pd.Series:
    """Clock difference in [0, 180] degrees; NaN if either input is NaN."""
    out = np.full(len(a), np.nan, dtype=float)
    both = pd.notna(a) & pd.notna(b)
    if not both.any():
        return pd.Series(out)
    d = np.abs(a.values.astype(float) - b.values.astype(float))
    d = np.minimum(d, 360 - d)
    out = np.where(both, d, np.nan)
    return pd.Series(out)


def _type_penalty(later_norm: pd.Series, prev_norm: pd.Series) -> pd.Series:
    """0 if match, 10 if either is 'other', 25 if differ and neither other."""
    later = later_norm.astype(str).str.strip().str.lower()
    prev = prev_norm.astype(str).str.strip().str.lower()
    match = (later == prev)
    either_other = (later == "other") | (prev == "other")
    out = np.where(match, TYPE_PENALTY_SAME, np.where(either_other, TYPE_PENALTY_OTHER, TYPE_PENALTY_DIFF))
    return pd.Series(out, index=later_norm.index)


def find_candidates_vectorized(
    later: pd.DataFrame,
    prev: pd.DataFrame,
    dist_tol_m: float = DISTANCE_TOL_M,
    *,
    prev_dist_col: str = "distance_aligned_m",
) -> pd.DataFrame:
    """
    For each later anomaly, find all prev anomalies with abs(prev_aligned - later_raw) <= dist_tol_m.
    prev must have prev_dist_col (e.g. distance_aligned_m); later uses distance_raw_m.
    """
    prev_d_col = prev_dist_col if prev_dist_col in prev.columns else "distance_raw_m"
    prev_sorted = prev.sort_values(prev_d_col).reset_index(drop=True)
    later_sorted = later.sort_values("distance_raw_m").reset_index(drop=True)
    prev_dist = prev_sorted[prev_d_col].values.astype(float)
    later_dist = later_sorted["distance_raw_m"].values.astype(float)

    rows = []
    for i in range(len(later_sorted)):
        later_d = later_dist[i]
        left = np.searchsorted(prev_dist, later_d - dist_tol_m, side="left")
        right = np.searchsorted(prev_dist, later_d + dist_tol_m, side="right")
        for j in range(left, right):
            rows.append({"later_idx": i, "prev_idx": j})
    if not rows:
        return pd.DataFrame()

    pair_df = pd.DataFrame(rows)
    later_cols = later_sorted.add_prefix("later_").reset_index(drop=False).rename(columns={"index": "later_idx"})
    prev_cols = prev_sorted.add_prefix("prev_").reset_index(drop=False).rename(columns={"index": "prev_idx"})
    pair_df = pair_df.merge(later_cols, on="later_idx", how="left")
    pair_df = pair_df.merge(prev_cols, on="prev_idx", how="left")
    return pair_df


def score_candidates(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Add score components and total_score. Uses aligned distance for delta_distance_m when present."""
    pair_df["later_distance_raw_m"] = pair_df["later_distance_raw_m"].astype(float)
    pair_df["prev_distance_raw_m"] = pair_df["prev_distance_raw_m"].astype(float)
    if "prev_distance_aligned_m" in pair_df.columns:
        pair_df["prev_distance_aligned_m"] = pd.to_numeric(pair_df["prev_distance_aligned_m"], errors="coerce")
        pair_df["delta_distance_m"] = (pair_df["later_distance_raw_m"] - pair_df["prev_distance_aligned_m"]).abs()
    else:
        pair_df["delta_distance_m"] = (pair_df["later_distance_raw_m"] - pair_df["prev_distance_raw_m"]).abs()

    # Clock (circular diff); 0 if missing
    clock_diff = _circular_clock_diff_deg(
        pair_df["later_clock_position_deg"],
        pair_df["prev_clock_position_deg"],
    )
    pair_df["clock_diff_deg"] = clock_diff.fillna(0)

    # Depth, length, width (0 if missing)
    later_d = pd.to_numeric(pair_df["later_depth_percent"], errors="coerce")
    prev_d = pd.to_numeric(pair_df["prev_depth_percent"], errors="coerce")
    pair_df["delta_depth_pct"] = (later_d - prev_d).fillna(0).abs()

    later_len = pd.to_numeric(pair_df["later_length_mm"], errors="coerce")
    prev_len = pd.to_numeric(pair_df["prev_length_mm"], errors="coerce")
    pair_df["delta_length_mm"] = (later_len - prev_len).fillna(0).abs()

    later_w = pd.to_numeric(pair_df["later_width_mm"], errors="coerce")
    prev_w = pd.to_numeric(pair_df["prev_width_mm"], errors="coerce")
    pair_df["delta_width_mm"] = (later_w - prev_w).fillna(0).abs()

    pair_df["type_penalty"] = _type_penalty(
        pair_df["later_feature_type_norm"],
        pair_df["prev_feature_type_norm"],
    )

    pair_df["total_score"] = (
        W_DIST * pair_df["delta_distance_m"].abs()
        + W_CLOCK * pair_df["clock_diff_deg"]
        + W_DEPTH * pair_df["delta_depth_pct"]
        + W_LENGTH * pair_df["delta_length_mm"]
        + W_WIDTH * pair_df["delta_width_mm"]
        + pair_df["type_penalty"]
    )
    return pair_df


CANDIDATES_OUT_COLUMNS = [
    "later_anomaly_id", "prev_anomaly_id", "later_run_year", "prev_run_year",
    "later_distance_raw_m", "prev_distance_raw_m", "prev_distance_aligned_m", "delta_distance_m",
    "later_clock_deg", "prev_clock_deg", "clock_diff_deg",
    "later_depth_percent", "prev_depth_percent", "delta_depth_pct",
    "later_length_mm", "prev_length_mm", "delta_length_mm",
    "later_width_mm", "prev_width_mm", "delta_width_mm",
    "later_feature_type_norm", "prev_feature_type_norm",
    "later_feature_type_raw", "prev_feature_type_raw",
    "type_penalty", "total_score", "rank_within_later",
]
BEST_MATCH_OUT_COLUMNS = [
    "later_anomaly_id", "best_prev_anomaly_id", "best_total_score",
    "second_best_score", "score_gap", "needs_review", "reason",
    "prev_used_by_n", "later_candidate_count",
]


def build_output_tables(
    pair_df: pd.DataFrame,
    later: pd.DataFrame,
    prev_run_year: int,
    later_run_year: int,
    topk: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build candidates (topk per later), best_matches, unmatched.
    """
    if pair_df.empty:
        candidates_out = pd.DataFrame(columns=CANDIDATES_OUT_COLUMNS)
        best_matches = pd.DataFrame(columns=BEST_MATCH_OUT_COLUMNS)
        unmatched = later.copy()
        return candidates_out, best_matches, unmatched

    # Rank within each later anomaly by total_score
    pair_df = pair_df.sort_values(["later_anomaly_id", "total_score"]).reset_index(drop=True)
    pair_df["rank_within_later"] = pair_df.groupby("later_anomaly_id").cumcount() + 1
    candidates_topk = pair_df[pair_df["rank_within_later"] <= topk].copy()

    # Ensure prev_distance_aligned_m exists (identity if no alignment)
    if "prev_distance_aligned_m" not in candidates_topk.columns:
        candidates_topk["prev_distance_aligned_m"] = candidates_topk["prev_distance_raw_m"]

    # Candidate table columns (rename for output)
    candidates_out = candidates_topk[[
        "later_anomaly_id", "prev_anomaly_id",
        "later_run_year", "prev_run_year",
        "later_distance_raw_m", "prev_distance_raw_m", "prev_distance_aligned_m", "delta_distance_m",
        "later_clock_position_deg", "prev_clock_position_deg", "clock_diff_deg",
        "later_depth_percent", "prev_depth_percent", "delta_depth_pct",
        "later_length_mm", "prev_length_mm", "delta_length_mm",
        "later_width_mm", "prev_width_mm", "delta_width_mm",
        "later_feature_type_norm", "prev_feature_type_norm",
        "later_feature_type_raw", "prev_feature_type_raw",
        "type_penalty", "total_score", "rank_within_later",
    ]].copy()
    candidates_out = candidates_out.rename(columns={
        "later_clock_position_deg": "later_clock_deg",
        "prev_clock_position_deg": "prev_clock_deg",
    })
    candidates_out = candidates_out[CANDIDATES_OUT_COLUMNS]

    # Best match per later: rank 1 only
    best = pair_df[pair_df["rank_within_later"] == 1].copy()
    later_with_candidates = set(best["later_anomaly_id"])
    second = pair_df[pair_df["rank_within_later"] == 2][["later_anomaly_id", "total_score"]].rename(
        columns={"total_score": "second_best_score"}
    )
    best = best.merge(second, on="later_anomaly_id", how="left")
    best["score_gap"] = best["second_best_score"] - best["total_score"]

    # Reason / needs_review (5 cases)
    def _reason_and_review(row: pd.Series) -> tuple[bool, str]:
        single = pd.isna(row["second_best_score"])
        gap = row["score_gap"]
        scr = row["total_score"]
        if single:
            return (bool(scr > REASON_HIGH_SCORE), "single_candidate")
        if gap < REASON_LOW_GAP_THRESHOLD and scr <= REASON_AMBIGUOUS_GOOD_MAX_SCORE:
            return (False, "ambiguous_but_good")
        if gap < REASON_LOW_GAP_THRESHOLD and scr > REASON_AMBIGUOUS_GOOD_MAX_SCORE:
            return (True, "low_gap")
        if scr > REASON_HIGH_SCORE:
            return (True, "high_score")
        return (False, "ok")

    rev_reas = best.apply(_reason_and_review, axis=1)
    best["needs_review"] = [r[0] for r in rev_reas]
    best["reason"] = [r[1] for r in rev_reas]

    # prev_used_by_n: how many later anomalies selected this best_prev_anomaly_id
    prev_use_count = best["prev_anomaly_id"].value_counts().to_dict()
    best["prev_used_by_n"] = best["prev_anomaly_id"].map(prev_use_count).astype(int)
    # later_candidate_count: number of candidate rows for this later in candidates file
    later_cand_count = candidates_topk["later_anomaly_id"].value_counts().to_dict()
    best["later_candidate_count"] = best["later_anomaly_id"].map(later_cand_count).astype(int)

    best_matches = best[[
        "later_anomaly_id", "prev_anomaly_id", "total_score",
        "second_best_score", "score_gap", "needs_review", "reason",
        "prev_used_by_n", "later_candidate_count",
    ]].rename(columns={"prev_anomaly_id": "best_prev_anomaly_id", "total_score": "best_total_score"})

    # Unmatched: later anomalies with zero candidates
    unmatched = later[~later["anomaly_id"].isin(later_with_candidates)].copy()

    return candidates_out, best_matches, unmatched


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Broad anomaly matching between two runs")
    parser.add_argument("--input", "-i", default="data_ready/anomalies_clean.csv", help="Anomalies CSV path")
    parser.add_argument("--all-runs", default="data_ready/all_runs_clean.csv", help="All-runs CSV for weld anchors")
    parser.add_argument("--out", "-o", default="output/matching", help="Output directory")
    parser.add_argument("--prev", type=int, default=2015, help="Previous run year")
    parser.add_argument("--later", type=int, default=2022, help="Later run year")
    parser.add_argument("--topk", type=int, default=5, help="Top-k candidates per later anomaly")
    args = parser.parse_args()

    input_path = Path(args.input)
    all_runs_path = Path(args.all_runs)
    out_dir = Path(args.out)
    prev_year = args.prev
    later_year = args.later
    topk = args.topk

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path, low_memory=False)
    required = {"anomaly_id", "run_year", "distance_raw_m", "feature_type_norm", "feature_type_raw",
                "clock_position_deg", "depth_percent", "length_mm", "width_mm"}
    if not required.issubset(df.columns):
        print(f"ERROR: Missing columns: {required - set(df.columns)}", file=sys.stderr)
        return 1

    prev = df[df["run_year"] == prev_year].copy()
    later = df[df["run_year"] == later_year].copy()
    if prev.empty or later.empty:
        print(f"ERROR: No data for prev={prev_year} or later={later_year}", file=sys.stderr)
        return 1

    prev["prev_run_year"] = prev_year
    later["later_run_year"] = later_year

    # Weld-anchored alignment (prev -> later frame)
    alignment_meta = {"alignment_used": False, "anchor_count": 0, "anchor_range_prev_m": None, "offset_stats": None, "method": None}
    anchors_df_for_csv = None

    if all_runs_path.exists():
        anchors, _ = _build_anchor_pairs(all_runs_path, prev_year, later_year)
        if anchors is not None and len(anchors) >= MIN_ANCHORS:
            offset_fn, anchors_smoothed, meta = _fit_alignment(anchors)
            alignment_meta = {"alignment_used": True, **meta}
            anchors_df_for_csv = anchors_smoothed
            prev_d = pd.to_numeric(prev["distance_raw_m"], errors="coerce")
            offset_vals = offset_fn(prev_d.values)
            prev["distance_aligned_m"] = prev_d.values + np.atleast_1d(offset_vals)
        else:
            if anchors is not None and len(anchors) > 0:
                print(f"WARNING: Only {len(anchors)} anchors (need {MIN_ANCHORS}); using identity alignment.", file=sys.stderr)
            prev["distance_aligned_m"] = pd.to_numeric(prev["distance_raw_m"], errors="coerce")
    else:
        prev["distance_aligned_m"] = pd.to_numeric(prev["distance_raw_m"], errors="coerce")
    if "distance_aligned_m" not in prev.columns or prev["distance_aligned_m"].isna().all():
        prev["distance_aligned_m"] = pd.to_numeric(prev["distance_raw_m"], errors="coerce")

    pair_df = find_candidates_vectorized(later, prev, dist_tol_m=DISTANCE_TOL_M)
    if pair_df.empty:
        pair_df = pd.DataFrame()
    else:
        pair_df = score_candidates(pair_df)

    candidates_out, best_matches, unmatched = build_output_tables(
        pair_df, later, prev_year, later_year, topk
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    prev_s = str(prev_year)
    later_s = str(later_year)

    candidates_out.to_csv(out_dir / f"candidates_{prev_s}_{later_s}.csv", index=False)
    best_matches.to_csv(out_dir / f"best_matches_{prev_s}_{later_s}.csv", index=False)
    unmatched.to_csv(out_dir / f"unmatched_{later_s}.csv", index=False)

    if anchors_df_for_csv is not None:
        anchors_df_for_csv.to_csv(out_dir / f"alignment_{prev_s}_{later_s}_anchors.csv", index=False)

    # Summary
    n_later = len(later)
    n_with_candidate = len(best_matches)
    has_candidate_pct = (n_with_candidate / n_later * 100) if n_later else 0
    avg_candidates = (candidates_out.groupby("later_anomaly_id").size().mean()) if not candidates_out.empty else 0
    median_best = best_matches["best_total_score"].median() if not best_matches.empty else float("nan")
    n_review = int(best_matches["needs_review"].sum()) if not best_matches.empty else 0
    needs_review_pct = (n_review / n_with_candidate * 100) if n_with_candidate else 0

    reason_counts = best_matches["reason"].value_counts().to_dict() if not best_matches.empty else {}
    top_prev_pileups = (
        best_matches["best_prev_anomaly_id"].value_counts().head(10).to_dict()
        if not best_matches.empty else {}
    )

    score_components = {}
    if not candidates_out.empty:
        for col in ["delta_distance_m", "clock_diff_deg", "delta_depth_pct", "delta_length_mm", "delta_width_mm", "type_penalty", "total_score"]:
            if col in candidates_out.columns:
                score_components[col] = {
                    "min": float(candidates_out[col].min()),
                    "max": float(candidates_out[col].max()),
                    "mean": float(candidates_out[col].mean()),
                }

    summary = {
        "prev_run_year": prev_year,
        "later_run_year": later_year,
        "later_anomalies_count": n_later,
        "with_at_least_one_candidate_count": n_with_candidate,
        "has_candidate_pct": round(has_candidate_pct, 2),
        "unmatched_count": len(unmatched),
        "avg_candidates_per_later_topk": round(float(avg_candidates), 2),
        "median_best_total_score": float(median_best) if pd.notna(median_best) else None,
        "needs_review_count": n_review,
        "needs_review_pct": round(needs_review_pct, 2),
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "top_prev_pileups": {str(k): int(v) for k, v in top_prev_pileups.items()},
        "score_component_stats": score_components,
        "alignment_used": alignment_meta.get("alignment_used", False),
        "anchor_count": alignment_meta.get("anchor_count", 0),
        "anchor_range_prev_m": alignment_meta.get("anchor_range_prev_m"),
        "offset_stats": alignment_meta.get("offset_stats"),
        "method": alignment_meta.get("method"),
    }
    with open(out_dir / f"summary_{prev_s}_{later_s}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("Matching summary")
    print("---------------")
    print(f"  Later anomalies count: {n_later}")
    print(f"  has_candidate_pct: {has_candidate_pct:.1f}%")
    print(f"  needs_review_pct: {needs_review_pct:.1f}%")
    print("  reason_counts:", reason_counts)
    top5_pileups = (
        best_matches["best_prev_anomaly_id"].value_counts().head(5).to_dict()
        if not best_matches.empty else {}
    )
    print("  Top 5 prev pileups (best_prev_anomaly_id -> count):", top5_pileups)
    print(f"Outputs: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
