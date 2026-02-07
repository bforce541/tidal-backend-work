"""Configurable parameters for ILI prediction pipeline."""

# Prediction
PREDICTION_HORIZON_YEARS = 5
RISK_DEPTH_THRESHOLDS = [40, 60]  # % wall thickness
MAX_DEPTH_RATE_PER_YEAR = 2.0  # conservative clamp: max %/year for extrapolation

# Runs (baseline 2007, alignment 2007 -> 2015 -> 2022)
BASELINE_YEAR = 2007
RUN_YEARS = [2007, 2015, 2022]

# Matching tolerances (post-alignment)
AXIAL_DISTANCE_TOLERANCE_M = 50.0  # meters
AXIAL_TOL_M = AXIAL_DISTANCE_TOLERANCE_M
CLOCK_POSITION_TOLERANCE_DEG = 30  # degrees
CLOCK_TOL_DEG = CLOCK_POSITION_TOLERANCE_DEG
DEPTH_SIMILARITY_TOLERANCE = 0.2  # relative difference
MATCH_SCORE_EPSILON = 0.5  # if top two scores within this → Ambiguous

# Matching score weights (lower score = better)
W_DIST = 1.0
W_CLOCK = 0.01
W_DEPTH = 1.0
W_LENGTH = 0.1
W_WIDTH = 0.1

# Growth sanity
MAX_DEPTH_RATE_OUTLIER = 5.0  # %/yr - flag as outlier_rate

# Segment alignment
SEGMENT_LENGTH_MIN_M = 0.01  # guard: treat shorter segments as identity

# Unit conversions (to canonical: m, degrees 0-360, % wall, mm)
FT_TO_M = 0.3048
IN_TO_MM = 25.4
