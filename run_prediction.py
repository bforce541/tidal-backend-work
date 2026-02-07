#!/usr/bin/env python3
"""
ILI Future Year Prediction - Entry point.

Pipeline:
  1. Load & parse Excel
  2. Normalize (m, degrees, % wall, mm)
  3. Weld-anchored alignment
  4. Anomaly matching
  5. Growth calculation & tracks
  6. 5-year prediction & risk flagging
  7. Output tables (CSV + Excel) + stacked image
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from config import PREDICTION_HORIZON_YEARS, RISK_DEPTH_THRESHOLDS
from src.align import align_all, get_welds
from src.growth import build_summary, build_tracks, compute_growth_rates
from src.match import run_matching
from src.normalize import normalize_all
from src.predict import predict_tracks
from src.render import render_stacked, save_tables
from src.schema import get_runs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ILI Future Year Prediction")
    parser.add_argument("--input", "-i", default="data/ILIDataV2.xlsx", help="Input Excel path")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON_YEARS,
                        help="Prediction horizon in years")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1

    logger.info("Loading Excel...")
    parsed = get_runs(input_path)
    if not parsed:
        logger.error("No run data found (expected 2007, 2015, 2022 sheets)")
        return 1

    logger.info("Normalizing...")
    normalized = normalize_all(parsed)

    logger.info("Aligning (weld-anchored)...")
    aligned, weld_maps = align_all(normalized)

    logger.info("Matching anomalies...")
    matches = run_matching(aligned)

    logger.info("Computing growth & tracks...")
    m07_15 = matches.get("Matches_2007_2015", pd.DataFrame())
    m15_22 = matches.get("Matches_2015_2022", pd.DataFrame())
    m07_15, m15_22 = compute_growth_rates(m07_15, m15_22)
    tracks = build_tracks(m07_15, m15_22)
    summary = build_summary(matches, tracks)

    logger.info("Predicting...")
    pred_year = 2022 + args.horizon
    predicted = predict_tracks(
        tracks,
        horizon_years=args.horizon,
        risk_thresholds=RISK_DEPTH_THRESHOLDS,
    )

    logger.info("Saving outputs...")
    save_tables(weld_maps, matches, tracks, summary, predicted, tables_dir)

    logger.info("Rendering stacked image...")
    img_path = output_dir / "ili_prediction.png"
    render_stacked(aligned, predicted_tracks=predicted, pred_year=pred_year, out_path=img_path)

    logger.info(f"Done. Tables in {tables_dir}, image at {img_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
