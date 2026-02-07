"""Render stacked ILI Data Alignment image in baseline format."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.align import is_weld
from src.match import is_anomaly
from src.schema import FEATURE_TYPE


def _is_valve(event: str) -> bool:
    return event is not None and "valve" in str(event).lower()


def _is_tee(event: str) -> bool:
    return event is not None and "tee" in str(event).lower()


def _is_bend(event: str) -> bool:
    return event is not None and "bend" in str(event).lower()


def _draw_valve(ax, x: float, y: float, size: float = 0.03):
    """Yellow bow-tie valve symbol."""
    angles = [45, 135, 225, 315]
    xs = [x + size * np.cos(np.radians(a)) for a in angles]
    ys = [y + size * 0.6 * np.sin(np.radians(a)) for a in angles]
    ax.plot(xs[:2] + [x] + xs[2:], ys[:2] + [y] + ys[2:], color="gold", lw=2, zorder=3)
    ax.plot(xs[2:] + [x] + xs[:2], ys[2:] + [y] + ys[:2], color="gold", lw=2, zorder=3)
    circle = plt.Circle((x, y), size * 0.4, color="white", ec="gold", lw=1, zorder=4)
    ax.add_patch(circle)


def _draw_tee(ax, x: float, y: float, size: float = 0.03):
    """Light blue circle with T."""
    circle = plt.Circle((x, y), size, color="lightblue", ec="steelblue", lw=1, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, "T", ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=5)


def _draw_bend(ax, x: float, y: float, size: float = 0.025):
    """Red circle."""
    circle = plt.Circle((x, y), size, color="red", ec="darkred", lw=1, zorder=4)
    ax.add_patch(circle)


def _draw_anomaly(ax, x: float, y: float, is_new: bool = False, size: float = 0.02):
    """Beige oval (existing) or red-outline oval (new)."""
    w, h = size * 2, size * 1.2
    e = mpatches.Ellipse((x, y), w, h, angle=45, fill=True, zorder=4)
    if is_new:
        e.set_facecolor("mistyrose")
        e.set_edgecolor("red")
        e.set_linewidth(2)
    else:
        e.set_facecolor("wheat")
        e.set_edgecolor("tan")
        e.set_linewidth(1)
    ax.add_patch(e)


def _draw_girth_weld(ax, x: float, y: float, y_bottom: float, height: float = 0.15):
    """Gray vertical bar."""
    ax.plot([x, x], [y, y - height], color="gray", lw=4, solid_capstyle="butt", zorder=2)


def _draw_pipeline_row(
    ax,
    df: pd.DataFrame,
    y: float,
    label: str,
    is_baseline: bool,
    x_min: float,
    x_max: float,
    pos_col: str = "distance_corrected",
    predicted_new_mask: Optional[pd.Series] = None,
):
    """Draw one pipeline row with features."""
    ax.plot([0, 1], [y, y], color="black", lw=3, solid_capstyle="round", zorder=1)
    if x_max <= x_min:
        x_range = 1
    else:
        x_range = x_max - x_min

    def x_norm(d):
        if pd.isna(d):
            return 0.5
        return (d - x_min) / x_range if x_range > 0 else 0.5

    for _, row in df.iterrows():
        pos = row.get(pos_col, row.get("distance", 0))
        x = x_norm(pos)
        if x < 0 or x > 1:
            continue
        event = row.get(FEATURE_TYPE, "")
        if is_weld(event):
            _draw_girth_weld(ax, x, y, y - 0.12)
        elif _is_valve(event):
            _draw_valve(ax, x, y)
        elif _is_tee(event):
            _draw_tee(ax, x, y)
        elif _is_bend(event):
            _draw_bend(ax, x, y)
        elif is_anomaly(event):
            is_new = predicted_new_mask is not None and predicted_new_mask.get(row.name, False)
            _draw_anomaly(ax, x, y, is_new=is_new)

    ax.text(-0.08, y, label, ha="right", va="center", fontsize=10, color="steelblue")
    if is_baseline:
        ax.text(1.08, y, "Baseline", ha="left", va="center", fontsize=10, color="green")


def render_stacked(
    aligned: dict[int, pd.DataFrame],
    predicted_tracks: Optional[pd.DataFrame] = None,
    pred_year: int = 2027,
    out_path: Optional[Path] = None,
) -> None:
    """
    Render stacked ILI runs in baseline format.
    Rows: 2007 (baseline), 2015, 2022, Predicted {pred_year}.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.2, 1.25)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")
    ax.set_title("ILI Data Alignment", fontsize=18, color="darkblue", pad=20)
    ax.text(0.5, 1.05, "Same pipeline features appear at different reported locations in each ILI run",
            ha="center", va="top", fontsize=11, color="gray", transform=ax.transAxes)

    all_pos = []
    for df in aligned.values():
        if "distance_corrected" in df.columns:
            all_pos.extend(df["distance_corrected"].dropna().tolist())
        else:
            all_pos.extend(df.get("distance", pd.Series()).dropna().tolist())
    x_min = min(all_pos) if all_pos else 0
    x_max = max(all_pos) if all_pos else 1

    rows = []
    for year in [2007, 2015, 2022]:
        if year in aligned:
            rows.append((year, aligned[year], f"ILI Run ({year})", year == 2007, None))

    if predicted_tracks is not None and len(predicted_tracks) > 0:
        df_2022 = aligned.get(2022)
        if df_2022 is not None:
            pred_df = df_2022.copy()
            pred_df["distance_corrected"] = pred_df.get("distance_corrected", pred_df.get("distance"))
            pred_new = {}
            for _, r in predicted_tracks[predicted_tracks["risk_flag"].notna()].iterrows():
                idx = r.get("idx_2022")
                if pd.notna(idx):
                    pred_new[idx] = True
            rows.append((pred_year, pred_df, f"Predicted {pred_year}", False, pred_new))

    for i, (year, df, label, is_baseline, pred_new) in enumerate(rows):
        y = 0.85 - i * 0.22
        _draw_pipeline_row(ax, df, y, label, is_baseline, x_min, x_max, predicted_new_mask=pred_new)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def save_tables(
    weld_maps: dict[str, pd.DataFrame],
    matches: dict[str, pd.DataFrame],
    tracks: pd.DataFrame,
    summary: pd.DataFrame,
    predicted: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save all output tables as CSV and Excel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables = {
        **weld_maps,
        **matches,
        "Tracks_2007_2015_2022": tracks,
        "Summary": summary,
        "Predicted": predicted,
    }
    for name, df in all_tables.items():
        if df is None:
            continue
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
    excel_path = out_dir / "ili_outputs.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        for name, df in all_tables.items():
            if df is not None and (len(df) > 0 or name == "Summary"):
                df.to_excel(w, sheet_name=name[:31], index=False)
