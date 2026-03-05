"""
optimize_params.py — Grid search over tactical_weight and home_advantage.

Evaluates configurations against real Six Nations results and reports
RMSE, MAE, and winner-prediction accuracy.
"""

import copy
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from match_predictor import simulate_match
from squad_builder import build_match_squads

# ── Ground truth ───────────────────────────────────────────────────────────────

REAL_RESULTS = [
    {"home": "France",   "away": "Ireland",  "home_score": 36, "away_score": 14, "match_file": "France-Ireland.xlsx"},
    {"home": "Italy",    "away": "Scotland", "home_score": 18, "away_score": 15, "match_file": "Italy-Scotland.xlsx"},
    {"home": "England",  "away": "Wales",    "home_score": 48, "away_score":  7, "match_file": "England-Wales.xlsx"},
    {"home": "Scotland", "away": "England",  "home_score": 31, "away_score": 20, "match_file": "Scotland-England.xlsx"},
    {"home": "Wales",    "away": "France",   "home_score": 12, "away_score": 54, "match_file": "Wales-France.xlsx"},
    {"home": "Ireland",  "away": "Italy",    "home_score": 20, "away_score": 13, "match_file": "Ireland-Italy.xlsx"},
]


# ── Metrics ────────────────────────────────────────────────────────────────────

def calculate_metrics(predictions: list[dict], real_results: list[dict]) -> dict:
    """
    Compute evaluation metrics comparing predicted vs real margins.

    Parameters
    ----------
    predictions : list of dict
        Each must have a 'margin' key (positive = home win).
    real_results : list of dict
        Each must have 'home_score' and 'away_score'.

    Returns
    -------
    dict
        rmse, mae, max_error, correct_winners, accuracy.
    """
    if not predictions:
        raise ValueError("Predictions list is empty.")

    errors = []
    correct = 0

    for pred, real in zip(predictions, real_results):
        real_margin = real["home_score"] - real["away_score"]
        pred_margin = pred["margin"]

        errors.append(abs(pred_margin - real_margin))

        if (
            (real_margin >  0 and pred_margin >  0)
            or (real_margin <  0 and pred_margin <  0)
            or (real_margin == 0 and abs(pred_margin) < 3)
        ):
            correct += 1

    errors = np.array(errors)

    return {
        "rmse":            float(np.sqrt(np.mean(errors ** 2))),
        "mae":             float(np.mean(errors)),
        "max_error":       float(np.max(errors)),
        "correct_winners": int(correct),
        "accuracy":        correct / len(real_results) * 100.0,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_params(tactical_weight: float, home_advantage: float) -> dict | None:
    """
    Run all REAL_RESULTS matches under a given parameter configuration.

    Returns a metrics dict, or None if any match fails to load.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["WEIGHT_MODEL_TACTICAL"] = tactical_weight
    config["WEIGHT_MODEL_ELO"]      = round(1.0 - tactical_weight, 4)
    config["HOME_ADVANTAGE"]        = home_advantage

    predictions = []

    for match in REAL_RESULTS:
        try:
            df   = build_match_squads(match["match_file"])
            pred = simulate_match(match["home"], match["away"], df, config)
            if "error" in pred:
                raise RuntimeError(pred["error"])
            predictions.append(pred)
        except Exception as exc:
            print(f"  Error processing {match['match_file']}: {exc}")
            return None

    return calculate_metrics(predictions, REAL_RESULTS)


# ── Grid search ────────────────────────────────────────────────────────────────

def grid_search() -> tuple[dict, pd.DataFrame]:
    """
    Search over tactical_weight and home_advantage.

    Returns
    -------
    tuple
        (best_params dict, full results DataFrame sorted by RMSE)
    """
    print("=" * 70)
    print("GRID SEARCH OPTIMIZATION — Six Nations Predictor")
    print("=" * 70)
    print(f"\nEvaluating {len(REAL_RESULTS)} matches:")
    for i, m in enumerate(REAL_RESULTS, 1):
        print(f"  {i}. {m['home']} vs {m['away']}: {m['home_score']}-{m['away_score']}")

    tactical_weights = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    home_advantages  = [0.0,  1.0,  2.0,  3.0,  4.0,  5.0]
    total = len(tactical_weights) * len(home_advantages)

    print(f"\nTesting {total} parameter combinations...\n" + "=" * 70)

    results     = []
    best_rmse   = float("inf")
    best_params = None

    for i, tactical in enumerate(tactical_weights, 1):
        for home_adv in home_advantages:
            metrics = evaluate_params(tactical, home_adv)
            if metrics is None:
                continue

            row = {
                "tactical_weight": tactical,
                "elo_weight":       round(1.0 - tactical, 2),
                "home_advantage":   home_adv,
                **metrics,
            }
            results.append(row)

            if metrics["rmse"] < best_rmse:
                best_rmse   = metrics["rmse"]
                best_params = row

        print(f"  Progress: {i * len(home_advantages)}/{total} combinations evaluated...")

    if not results:
        raise RuntimeError("No parameter combination could be evaluated successfully.")

    results_df = pd.DataFrame(results).sort_values("rmse")

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY RMSE\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(results_df.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"\n  Tactical Weight : {best_params['tactical_weight']:.2f}")
    print(f"  ELO Weight      : {best_params['elo_weight']:.2f}")
    print(f"  Home Advantage  : {best_params['home_advantage']:.1f} pts")
    print(f"\n  RMSE            : {best_params['rmse']:.2f} pts")
    print(f"  MAE             : {best_params['mae']:.2f} pts")
    print(f"  Max Error       : {best_params['max_error']:.2f} pts")
    print(f"  Correct Winners : {best_params['correct_winners']}/{len(REAL_RESULTS)}")
    print(f"  Accuracy        : {best_params['accuracy']:.1f}%")

    # Comparison with current defaults
    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT PARAMETERS")
    print("=" * 70)

    current = evaluate_params(
        DEFAULT_CONFIG["WEIGHT_MODEL_TACTICAL"],
        DEFAULT_CONFIG["HOME_ADVANTAGE"],
    )
    print(f"\n  Current  → Tactical: {DEFAULT_CONFIG['WEIGHT_MODEL_TACTICAL']:.2f}, "
          f"Home Adv: {DEFAULT_CONFIG['HOME_ADVANTAGE']:.1f} | "
          f"RMSE: {current['rmse']:.2f} | Accuracy: {current['accuracy']:.1f}%")
    print(f"  Optimised → Tactical: {best_params['tactical_weight']:.2f}, "
          f"Home Adv: {best_params['home_advantage']:.1f} | "
          f"RMSE: {best_params['rmse']:.2f} | Accuracy: {best_params['accuracy']:.1f}%")

    improvement = (current["rmse"] - best_params["rmse"]) / current["rmse"] * 100
    print(f"\n  RMSE improvement: {improvement:+.1f}%")

    output_csv = "optimization_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    print("=" * 70 + "\n")

    return best_params, results_df


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        grid_search()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as exc:
        import traceback
        print(f"\nError during optimization: {exc}")
        traceback.print_exc()