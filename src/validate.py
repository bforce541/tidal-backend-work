"""
Quick sanity checks for MVP. Run automatically in MVP mode.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MATCH_RATE_WARN_PCT = 10.0


def run_validation(
    aligned: dict[int, pd.DataFrame],
    matches: dict[str, pd.DataFrame],
    weld_map: pd.DataFrame,
    prev_year: int,
    later_year: int,
) -> None:
    """Run sanity checks, log warnings."""
    warnings = []
    # Required columns
    for year, df in aligned.items():
        if "distance_corrected_m" not in df.columns and "distance_corrected" not in df.columns:
            if "distance_raw_m" in df.columns:
                pass  # MVP may use distance_raw_m as corrected when no alignment
            else:
                warnings.append(f"Run {year}: no distance_corrected column")
    # Corrected distances exist
    for year, df in aligned.items():
        dc = df.get("distance_corrected_m", df.get("distance_corrected", pd.Series()))
        if len(dc) > 0 and dc.isna().all():
            warnings.append(f"Run {year}: all corrected distances are NaN")
    # Weld segments cover anomalies
    if len(weld_map) > 0 and "distance_prev_m" in weld_map.columns:
        dmin = weld_map["distance_prev_m"].min()
        dmax = weld_map["distance_prev_m"].max()
        for year, df in aligned.items():
            dc = df.get("distance_corrected_m", df.get("distance_corrected", pd.Series()))
            valid = dc.dropna()
            if len(valid) > 0:
                out = ((valid < dmin) | (valid > dmax)).sum()
                if out > len(valid) * 0.5:
                    warnings.append(f"Run {year}: many anomalies outside weld range")
    # Match rate
    m_key = f"Matches_{prev_year}_{later_year}"
    n_key = f"New_{later_year}_vs_{prev_year}"
    if m_key in matches and n_key in matches:
        n_matched = len(matches[m_key])
        n_new = len(matches[n_key])
        total_later = n_matched + n_new
        if total_later > 0:
            rate = n_matched / total_later * 100
            if rate < MATCH_RATE_WARN_PCT:
                warnings.append(
                    f"Match rate {rate:.1f}% < {MATCH_RATE_WARN_PCT}% - check tolerances or feature type mapping"
                )
    for w in warnings:
        logger.warning(f"Validate: {w}")
