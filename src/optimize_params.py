"""
optimize_params.py — Grid search over tactical_weight and home_advantage.

Reads match results from historical_data/results.csv and evaluates
configurations against all available seasons.
"""

import copy
import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, RESULTS_FILE
from match_predictor import simulate_match
from squad_builder import build_match_squads


# ── Load results from CSV ──────────────────────────────────────────────────────

def load_results() -> list[dict]:
    """
    Load match results from historical_data/results.csv.

    Expected columns: season, round, home, away, home_score, away_score, match_file
    """
    try:
        df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_FILE}\n"
            "Create historical_data/results.csv with columns: "
            "season, round, home, away, home_score, away_score, match_file"
        )

    required = {"season", "round", "home", "away", "home_score", "away_score", "match_file"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {missing}")

    return df.to_dict(orient="records")


# ── Metrics ────────────────────────────────────────────────────────────────────

def calculate_metrics(predictions: list[dict], real_results: list[dict]) -> dict:
    if not predictions:
        raise ValueError("Predictions list is empty.")

    errors  = []
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

def evaluate_params(
    tactical_weight: float,
    home_advantage: float,
    real_results: list[dict],
) -> dict | None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["WEIGHT_MODEL_TACTICAL"] = tactical_weight
    config["WEIGHT_MODEL_ELO"]      = round(1.0 - tactical_weight, 4)
    config["HOME_ADVANTAGE"]        = home_advantage

    predictions = []

    for match in real_results:
        try:
            df   = build_match_squads(match["match_file"], season=str(match["season"]))
            pred = simulate_match(
                match["home"], match["away"], df, config,
                season=str(match["season"])
            )
            if "error" in pred:
                raise RuntimeError(pred["error"])
            predictions.append(pred)
        except Exception as exc:
            print(f"  Error processing {match['match_file']} ({match['season']}): {exc}")
            return None

    return calculate_metrics(predictions, real_results)


# ── Grid search ────────────────────────────────────────────────────────────────

def grid_search() -> tuple[dict, pd.DataFrame]:
    print("=" * 70)
    print("GRID SEARCH OPTIMIZATION — Six Nations Predictor")
    print("=" * 70)

    real_results = load_results()

    # Summary by season
    seasons = sorted(set(str(m["season"]) for m in real_results))
    print(f"\nSeasons loaded: {', '.join(seasons)}")
    print(f"Total matches:  {len(real_results)}\n")
    for m in real_results:
        print(f"  {m['season']} R{m['round']:>2}  {m['home']} vs {m['away']}: "
              f"{m['home_score']}-{m['away_score']}")

    tactical_weights = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    home_advantages  = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    total = len(tactical_weights) * len(home_advantages)

    print(f"\nTesting {total} parameter combinations...\n" + "=" * 70)

    results     = []
    best_rmse   = float("inf")
    best_params = None

    for i, tactical in enumerate(tactical_weights, 1):
        for home_adv in home_advantages:
            metrics = evaluate_params(tactical, home_adv, real_results)
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

        if i % 3 == 0:
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
    print(f"  Correct Winners : {best_params['correct_winners']}/{len(real_results)}")
    print(f"  Accuracy        : {best_params['accuracy']:.1f}%")

    # Comparison with current defaults
    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT PARAMETERS")
    print("=" * 70)

    current = evaluate_params(
        DEFAULT_CONFIG["WEIGHT_MODEL_TACTICAL"],
        DEFAULT_CONFIG["HOME_ADVANTAGE"],
        real_results,
    )
    if current:
        print(f"\n  Current   → Tactical: {DEFAULT_CONFIG['WEIGHT_MODEL_TACTICAL']:.2f}, "
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

    # Apply best params and regenerate dashboard
    DEFAULT_CONFIG["WEIGHT_MODEL_TACTICAL"] = best_params["tactical_weight"]
    DEFAULT_CONFIG["WEIGHT_MODEL_ELO"]      = best_params["elo_weight"]
    DEFAULT_CONFIG["HOME_ADVANTAGE"]        = best_params["home_advantage"]

    print("\nGenerating dashboard with optimised parameters...")
    try:
        from generate_dashboard import generate
        generate()
    except Exception as exc:
        print(f"Warning: dashboard generation failed: {exc}")

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