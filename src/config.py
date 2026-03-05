"""
Central configuration for the Six Nations Predictor.
Edit this file to update ELO ratings, model parameters, or file paths.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data")
DB_FILE    = os.path.join(DATA_DIR, "six_nations_squads_FINAL.csv")
OUTPUT_DIR = BASE_DIR   # match_ready_squads.csv stays in root by default


def match_file_path(filename: str) -> str:
    """Returns the full path for a match Excel file inside the data folder."""
    return os.path.join(DATA_DIR, filename)


# ── World Rugby ELO Ratings ────────────────────────────────────────────────────
# Update these before each tournament with current rankings.
WR_ELO = {
    "Ireland":  87.97,
    "France":   87.24,
    "England":  89.41,
    "Scotland": 80.22,
    "Italy":    78.98,
    "Wales":    74.23,
}

# ── Model Hyperparameters ──────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Environmental
    "HOME_ADVANTAGE":         2.0,
    "STD_DEV":               13.5,

    # Hybrid model weights  (must sum to 1.0)
    "WEIGHT_MODEL_TACTICAL":  0.65,
    "WEIGHT_MODEL_ELO":       0.35,

    # Tactical unit weights  (must sum to 1.0)
    "WEIGHT_PACK":            0.40,
    "WEIGHT_CONTROL":         0.35,
    "WEIGHT_STRIKE":          0.25,

    # Starting XV vs Bench   (must sum to 1.0)
    "WEIGHT_STARTERS":        0.80,
    "WEIGHT_BENCH":           0.20,

    # Pack weighting inside simulate_match
    "PACK_STARTER_WEIGHT":    0.70,
    "PACK_BENCH_WEIGHT":      0.30,

    # Scaling factors
    "TACTICAL_SCALING_FACTOR": 0.85,
    "ELO_SCALING":             0.70,

    # Physical mismatch (continuous linear ramp)
    "MISMATCH_THRESHOLD":      4.0,   # below this → no bonus
    "MISMATCH_MAX_DIFF":      10.0,   # at this diff → full bonus
    "MISMATCH_BONUS":          5.0,   # maximum bonus in points
}