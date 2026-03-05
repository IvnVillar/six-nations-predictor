"""
match_predictor.py — Runs the hybrid Six Nations match prediction.

Usage:
    python match_predictor.py                        # reads match_ready_squads.csv
    python match_predictor.py France-Ireland.xlsx    # derive home/away from filename
"""

import sys
import math
import pandas as pd

from config import DEFAULT_CONFIG, WR_ELO
from squad_builder import get_teams_from_filename


#  Tactical calculation helpers 

def _unit_average(players: pd.DataFrame, unit: str, params: dict) -> float:
    """
    Return the average skill for a tactical unit within a player subset.

    Parameters
    ----------
    players : pd.DataFrame
        Already filtered to the relevant group (starters or bench).
    unit : str
        One of 'Pack', 'Control', 'Strike'.
    params : dict
        Model configuration.

    Notes
    -----
    - Pack    → all Forwards
    - Control → shirt numbers 9-10; falls back to all Backs if none present
    - Strike  → Backs excluding shirt numbers 9-10
    """
    if players.empty:
        return 70.0

    if unit == "Pack":
        group = players[players["position_group"] == "Forwards"]
        return float(group["skill"].mean()) if not group.empty else 70.0

    backs = players[players["position_group"] == "Backs"]

    if unit == "Control":
        halfbacks = backs[backs["shirt_number"].isin([9, 10])]
        if not halfbacks.empty:
            return float(halfbacks["skill"].mean())
        return float(backs["skill"].mean()) if not backs.empty else 70.0

    if unit == "Strike":
        strike = backs[~backs["shirt_number"].isin([9, 10])]
        if not strike.empty:
            return float(strike["skill"].mean())
        return float(backs["skill"].mean()) if not backs.empty else 70.0

    return 70.0


def _tactical_score(group: pd.DataFrame, params: dict) -> float:
    """Compute the weighted tactical score for one group (starters or bench)."""
    return (
        _unit_average(group, "Pack",    params) * params["WEIGHT_PACK"]    +
        _unit_average(group, "Control", params) * params["WEIGHT_CONTROL"] +
        _unit_average(group, "Strike",  params) * params["WEIGHT_STRIKE"]
    )


def get_tactical_metrics(df: pd.DataFrame, country: str, params: dict) -> dict | None:
    """
    Compute the combined tactical score and pack breakdown for one team.

    Returns None if the country is missing from the DataFrame.
    """
    squad = df[df["country"] == country]
    if squad.empty:
        return None

    starters = squad[squad["starting"] == 1]
    bench    = squad[squad["starting"] == 0]

    score_xv    = _tactical_score(starters, params)
    score_bench = _tactical_score(bench,    params)

    total = (
        score_xv    * params["WEIGHT_STARTERS"] +
        score_bench * params["WEIGHT_BENCH"]
    )

    pack_xv    = _unit_average(starters, "Pack", params)
    pack_bench = _unit_average(bench,    "Pack", params)

    return {
        "total_score": total,
        "pack_xv":     pack_xv,
        "pack_bench":  pack_bench,
    }


def _mismatch_bonus(pack_diff: float, params: dict) -> float:
    """
    Return a continuous physical-mismatch bonus (positive favours home).

    Below MISMATCH_THRESHOLD  → no bonus.
    Between threshold and max → linear ramp up to MISMATCH_BONUS.
    Above max                 → capped at MISMATCH_BONUS.
    """
    threshold = params["MISMATCH_THRESHOLD"]
    max_diff  = params["MISMATCH_MAX_DIFF"]
    max_bonus = params["MISMATCH_BONUS"]

    abs_diff = abs(pack_diff)
    if abs_diff <= threshold:
        return 0.0

    ramp  = min(abs_diff - threshold, max_diff - threshold)
    bonus = (ramp / (max_diff - threshold)) * max_bonus
    return bonus if pack_diff > 0 else -bonus


#  Main simulation 

def simulate_match(
    home: str,
    away: str,
    df: pd.DataFrame,
    params: dict | None = None,
) -> dict:
    """
    Simulate a match and return a prediction dict.

    Parameters
    ----------
    home, away : str
        Team names (must match values in df['country'] and WR_ELO keys).
    df : pd.DataFrame
        match_ready_squads DataFrame (from squad_builder).
    params : dict, optional
        Model config. Defaults to DEFAULT_CONFIG if not provided.

    Returns
    -------
    dict
        Keys: home, away, margin, prob, debug.
        On error: {"error": "<message>"}.
    """
    if params is None:
        params = DEFAULT_CONFIG

    h = get_tactical_metrics(df, home, params)
    a = get_tactical_metrics(df, away, params)

    if h is None or a is None:
        missing = home if h is None else away
        return {"error": f"No squad data found for '{missing}' in the DataFrame."}

    # Tactical margin
    margin_tactical = (h["total_score"] - a["total_score"]) * params["TACTICAL_SCALING_FACTOR"]

    # Physical mismatch (continuous)
    pack_home = (
        h["pack_xv"]    * params["PACK_STARTER_WEIGHT"] +
        h["pack_bench"] * params["PACK_BENCH_WEIGHT"]
    )
    pack_away = (
        a["pack_xv"]    * params["PACK_STARTER_WEIGHT"] +
        a["pack_bench"] * params["PACK_BENCH_WEIGHT"]
    )
    pack_diff = pack_home - pack_away
    margin_tactical += _mismatch_bonus(pack_diff, params)

    # ELO margin
    elo_h = WR_ELO.get(home, 75.0)
    elo_a = WR_ELO.get(away, 75.0)
    margin_elo = (elo_h - elo_a) * params["ELO_SCALING"]

    # Hybrid fusion
    final_margin = (
        margin_tactical * params["WEIGHT_MODEL_TACTICAL"] +
        margin_elo      * params["WEIGHT_MODEL_ELO"] +
        params["HOME_ADVANTAGE"]
    )

    # Win probability via normal CDF
    z = final_margin / params["STD_DEV"]
    prob_home = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return {
        "home":   home,
        "away":   away,
        "margin": final_margin,
        "prob":   prob_home,
        "debug": {
            "tactical":  margin_tactical,
            "elo":        margin_elo,
            "pack_diff":  pack_diff,
        },
    }


def print_result(res: dict, params: dict) -> None:
    """Pretty-print one match prediction."""
    home, away = res["home"], res["away"]
    winner = home if res["margin"] > 0 else away
    prob   = res["prob"] if res["margin"] > 0 else 1 - res["prob"]

    print(f"Match:             {home} vs {away}")
    print(f"Predicted Winner:  {winner.upper()}")
    print(f"Predicted Margin:  {abs(res['margin']):.1f} points")
    print(f"Win Probability:   {prob * 100:.1f}%")
    print(f"\nBreakdown:")
    print(f"  Tactical Advantage (Squad): {res['debug']['tactical']:+.1f} pts")
    print(f"  Historical Advantage (ELO): {res['debug']['elo']:+.1f} pts")

    pack_diff = res["debug"]["pack_diff"]
    if abs(pack_diff) > params["MISMATCH_THRESHOLD"]:
        dominant = home if pack_diff > 0 else away
        bonus    = abs(_mismatch_bonus(pack_diff, params))
        print(f"  Pack mismatch bonus ({dominant}): +{bonus:.1f} pts")

    print("-" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_FILE = "match_ready_squads.csv"

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"\nData loaded: {INPUT_FILE} ({len(df)} records)")
    except FileNotFoundError:
        print(f"ERROR: '{INPUT_FILE}' not found. Run squad_builder.py first.")
        sys.exit(1)

    print("\nSIX NATIONS MATCH PREDICTION\n" + "=" * 60)

    # Determine home/away order from the source filename when passed as argument,
    # otherwise fall back to the order of the CSV (first team in file = home).
    if len(sys.argv) > 1:
        try:
            home, away = get_teams_from_filename(sys.argv[1])
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    else:
        teams = list(df["country"].unique())
        if len(teams) != 2:
            print(f"ERROR: Expected 2 teams in CSV, found {len(teams)}: {teams}")
            sys.exit(1)
        # Use the filename-based convention: first alphabetically is NOT reliable,
        # so we warn the user.
        home, away = teams[0], teams[1]
        print(
            f"WARNING: Home/away order inferred from CSV row order ({home} = home). "
            "Pass the match filename as an argument for reliable ordering.\n"
        )

    res = simulate_match(home, away, df)

    if "error" in res:
        print(f"ERROR: {res['error']}")
        sys.exit(1)

    print_result(res, DEFAULT_CONFIG)