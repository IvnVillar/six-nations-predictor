"""
Central configuration for the Six Nations Predictor.
Edit this file to update ELO ratings, model parameters, or file paths.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
HISTORICAL_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "historical_data"))
RESULTS_FILE     = os.path.join(HISTORICAL_DIR, "results.csv")
DB_FILE          = os.path.join(DATA_DIR, "six_nations_squads_FINAL.csv")
OUTPUT_DIR       = BASE_DIR


def match_file_path(filename: str, season: str = None) -> str:
    """
    Returns the full path for a match Excel file.
    - If season is provided, looks in historical_data/<season>/
    - Otherwise looks in data/ (current season workflow)
    """
    if season:
        return os.path.join(HISTORICAL_DIR, str(season), filename)
    return os.path.join(DATA_DIR, filename)


# ── World Rugby ELO Ratings — by season ───────────────────────────────────────
# Each entry is the published ranking immediately before that tournament.
# Update WR_ELO_CURRENT before each new tournament.
WR_ELO = {
    "2024": {
        "Ireland":  90.57,
        "France":   87.81,
        "England":  85.46,
        "Scotland": 83.43,
        "Italy":    75.93,
        "Wales":    80.64,
    },
    "2025": {
        "Ireland":  90.78,
        "France":   88.51,
        "England":  82.31,
        "Scotland": 83.34,
        "Italy":    78.64,
        "Wales":    74.01,
    },
    "2026": {
        "Ireland":  87.97,
        "France":   87.24,
        "England":  89.41,
        "Scotland": 80.22,
        "Italy":    78.98,
        "Wales":    74.23,
    },
}

# Convenience alias for the current season (used by match_predictor.py standalone)
CURRENT_SEASON   = "2026"
WR_ELO_CURRENT   = WR_ELO[CURRENT_SEASON]


def get_elo(season: str = None) -> dict:
    """Return ELO ratings for a given season, falling back to current."""
    if season and str(season) in WR_ELO:
        return WR_ELO[str(season)]
    return WR_ELO_CURRENT


# ── Model Hyperparameters ──────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Environmental
    "HOME_ADVANTAGE":          5.0,
    "STD_DEV":                13.5,

    # Hybrid model weights  (must sum to 1.0)
    "WEIGHT_MODEL_TACTICAL":   0.50,
    "WEIGHT_MODEL_ELO":        0.50,

    # Tactical unit weights  (must sum to 1.0)
    "WEIGHT_PACK":             0.40,
    "WEIGHT_CONTROL":          0.35,
    "WEIGHT_STRIKE":           0.25,

    # Starting XV vs Bench   (must sum to 1.0)
    "WEIGHT_STARTERS":         0.80,
    "WEIGHT_BENCH":            0.20,

    # Pack weighting inside simulate_match
    "PACK_STARTER_WEIGHT":     0.70,
    "PACK_BENCH_WEIGHT":       0.30,

    # Scaling factors
    "TACTICAL_SCALING_FACTOR": 0.85,
    "ELO_SCALING":             0.70,

    # Physical mismatch (continuous linear ramp)
    "MISMATCH_THRESHOLD":      4.0,
    "MISMATCH_MAX_DIFF":      10.0,
    "MISMATCH_BONUS":          5.0,
}