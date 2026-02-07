"""Configurable parameters for ILI prediction pipeline."""

# Prediction
PREDICTION_HORIZON_YEARS = 5
RISK_DEPTH_THRESHOLDS = [40, 60]  # % wall thickness

# Runs (baseline 2007, alignment 2007 -> 2015 -> 2022)
BASELINE_YEAR = 2007
RUN_YEARS = [2007, 2015, 2022]

# Matching tolerances (post-alignment)
AXIAL_DISTANCE_TOLERANCE_M = 2.0  # meters
CLOCK_POSITION_TOLERANCE_DEG = 30  # degrees
DEPTH_SIMILARITY_TOLERANCE = 0.2  # relative difference (e.g. 20%)

# Unit conversions (to canonical: m, degrees 0-360, % wall, mm)
FT_TO_M = 0.3048
IN_TO_MM = 25.4
