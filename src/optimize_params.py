import os
import copy
import numpy as np
import pandas as pd

from match_predictor import simulate_match, DEFAULT_CONFIG
from squad_builder import (
    normalize_name,
    match_player_data,
    get_country_fallback_skill,
    get_teams_from_filename,
)

REAL_RESULTS = [
    {
        "home": "France",
        "away": "Ireland",
        "home_score": 36,
        "away_score": 14,
        "match_file": "France-Ireland.xlsx",
    },
    {
        "home": "Italy",
        "away": "Scotland",
               "home_score": 18,
        "away_score": 15,
        "match_file": "Italy-Scotland.xlsx",
    },
    {
        "home": "England",
        "away": "Wales",
        "home_score": 48,
        "away_score": 7,
        "match_file": "England-Wales.xlsx",
    },
    {
        "home": "Scotland",
        "away": "England",
        "home_score": 31,
        "away_score": 20,
        "match_file": "Scotland-England.xlsx",
    },
    {
        "home": "Wales",
        "away": "France",
        "home_score": 12,
        "away_score": 54,
        "match_file": "Wales-France.xlsx",
    },
    {
        "home": "Ireland",
        "away": "Italy",
        "home_score": 20,
        "away_score": 13,
        "match_file": "Ireland-Italy.xlsx",
    },
]


def calculate_metrics(predictions, real_results):
    """
    Compute evaluation metrics for match margin predictions.

    Parameters
    ----------
    predictions : list of dict
        Each prediction must contain a 'margin' key (home - away).
    real_results : list of dict
        Each result must contain 'home_score' and 'away_score'.

    Returns
    -------
    dict
        Contains rmse, mae, max_error, correct_winners, accuracy.
    """
    if not predictions:
        raise ValueError("Predictions list is empty.")

    errors = []
    correct_winners = 0

    for pred, real in zip(predictions, real_results):
        real_margin = real["home_score"] - real["away_score"]
        pred_margin = pred["margin"]

        error = abs(pred_margin - real_margin)
        errors.append(error)

        if (
            (real_margin > 0 and pred_margin > 0)
            or (real_margin < 0 and pred_margin < 0)
            or (real_margin == 0 and abs(pred_margin) < 3)
        ):
            correct_winners += 1

    errors = np.array(errors)

    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "correct_winners": int(correct_winners),
        "accuracy": correct_winners / len(real_results) * 100.0,
    }


def _infer_position_group(jersey_number):
    """
    Infer position group ('Forwards' or 'Backs') from jersey number.
    """
    forwards_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20}
    return "Forwards" if jersey_number in forwards_numbers else "Backs"


def load_match_squads(match_file):
    """
    Load and process match squads from an Excel file.

    Parameters
    ----------
    match_file : str
        Match file name, e.g. "France-Wales.xlsx".

    Returns
    -------
    pandas.DataFrame
        Columns: country, name, position_group, skill, starting, shirt_number.
    """
    db_file = "data/six_nations_squads_FINAL.csv"
    match_path = os.path.join("data", match_file)

    if not os.path.exists(match_path):
        raise FileNotFoundError(f"Match file not found: {match_path}")

    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file not found: {db_file}")

    df_db = pd.read_csv(db_file, encoding="utf-8")

    team1, team2 = get_teams_from_filename(match_path)

    df_match = pd.read_excel(match_path, engine="openpyxl")
    if "Number" not in df_match.columns:
        raise KeyError(f"'Number' column not found in {match_path}")

    df_match = df_match.dropna(subset=["Number"])

    cols = list(df_match.columns)
    if len(cols) < 3:
        raise ValueError(
            f"Expected at least 3 columns (Team1, Number, Team2) in {match_path}, "
            f"got {len(cols)}."
        )

    col_team1, col_number, col_team2 = cols[0], "Number", cols[2]

    match_squad = []
    fallback_t1 = get_country_fallback_skill(team1, df_db)
    fallback_t2 = get_country_fallback_skill(team2, df_db)

    for _, row in df_match.iterrows():
        try:
            jersey_number = int(row[col_number])
        except (ValueError, TypeError):
            continue

        is_starting = 1 if jersey_number <= 15 else 0

        if col_team1 in row and not pd.isna(row[col_team1]):
            player1 = row[col_team1]
            data1 = match_player_data(player1, team1, df_db)

            if data1 is not None:
                match_squad.append(
                    {
                        "country": team1,
                        "name": data1["name"],
                        "position_group": data1["position_group"],
                        "skill": data1["skill"],
                        "starting": is_starting,
                        "shirt_number": jersey_number,
                    }
                )
            else:
                pos = _infer_position_group(jersey_number)
                match_squad.append(
                    {
                        "country": team1,
                        "name": player1,
                        "position_group": pos,
                        "skill": fallback_t1,
                        "starting": is_starting,
                        "shirt_number": jersey_number,
                    }
                )

        if col_team2 in row and not pd.isna(row[col_team2]):
            player2 = row[col_team2]
            data2 = match_player_data(player2, team2, df_db)

            if data2 is not None:
                match_squad.append(
                    {
                        "country": team2,
                        "name": data2["name"],
                        "position_group": data2["position_group"],
                        "skill": data2["skill"],
                        "starting": is_starting,
                        "shirt_number": jersey_number,
                    }
                )
            else:
                pos = _infer_position_group(jersey_number)
                match_squad.append(
                    {
                        "country": team2,
                        "name": player2,
                        "position_group": pos,
                        "skill": fallback_t2,
                        "starting": is_starting,
                        "shirt_number": jersey_number,
                    }
                )

    return pd.DataFrame(match_squad)


def evaluate_params(tactical_weight, home_advantage):
    """
    Evaluate a parameter configuration against REAL_RESULTS.

    Parameters
    ----------
    tactical_weight : float
        Weight for the tactical model.
    home_advantage : float
        Home advantage in points.

    Returns
    -------
    dict or None
        Metrics dict if successful, None if any match fails to process.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["WEIGHT_MODEL_TACTICAL"] = tactical_weight
    config["WEIGHT_MODEL_ELO"] = 1.0 - tactical_weight
    config["HOME_ADVANTAGE"] = home_advantage

    predictions = []

    for match in REAL_RESULTS:
        try:
            df = load_match_squads(match["match_file"])
            pred = simulate_match(
                match["home"],
                match["away"],
                df,
                config,
            )
            predictions.append(pred)
        except Exception as exc:
            print(f"Error processing {match['match_file']}: {exc}")
            return None

    return calculate_metrics(predictions, REAL_RESULTS)


def grid_search():
    """
    Run a simple grid search over tactical_weight and home_advantage.

    Returns
    -------
    tuple
        (best_params: dict, results_df: pandas.DataFrame)
    """
    print("=" * 70)
    print("GRID SEARCH OPTIMIZATION - Six Nations Predictor")
    print("=" * 70)
    print(f"\nEvaluating {len(REAL_RESULTS)} matches (first 2 rounds)")
    print("\nMatches:")
    for i, match in enumerate(REAL_RESULTS, 1):
        result = f"{match['home_score']}-{match['away_score']}"
        print(f"  {i}. {match['home']} vs {match['away']}: {result}")

    tactical_weights = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    home_advantages = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    total_combinations = len(tactical_weights) * len(home_advantages)
    print(f"\nTesting {total_combinations} parameter combinations...")
    print("=" * 70)

    results = []
    best_rmse = float("inf")
    best_params = None

    counter = 0
    for tactical in tactical_weights:
        for home_adv in home_advantages:
            counter += 1

            metrics = evaluate_params(tactical, home_adv)
            if metrics is None:
                continue

            result = {
                "tactical_weight": tactical,
                "elo_weight": round(1.0 - tactical, 2),
                "home_advantage": home_adv,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_error": metrics["max_error"],
                "correct_winners": metrics["correct_winners"],
                "accuracy_%": metrics["accuracy"],
            }
            results.append(result)

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_params = result

            if counter % 5 == 0:
                print(
                    f"Progress: {counter}/{total_combinations} combinations evaluated..."
                )

    if not results:
        raise RuntimeError("No parameter combination could be evaluated successfully.")

    results_df = pd.DataFrame(results).sort_values("rmse")

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\nTop 10 configurations by RMSE:\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(results_df.head(10).to_string(index=False))

    print("\n\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"\nTactical Weight:    {best_params['tactical_weight']:.2f}")
    print(f"ELO Weight:         {best_params['elo_weight']:.2f}")
    print(f"Home Advantage:     {best_params['home_advantage']:.1f} points")
    print("\nPerformance:")
    print(f"  RMSE:             {best_params['rmse']:.2f} points")
    print(f"  MAE:              {best_params['mae']:.2f} points")
    print(f"  Max Error:        {best_params['max_error']:.2f} points")
    print(
        f"  Correct Winners:  {best_params['correct_winners']}/{len(REAL_RESULTS)}"
    )
    print(f"  Accuracy:         {best_params['accuracy_%']:.1f}%")

    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT PARAMETERS")
    print("=" * 70)

    current_metrics = evaluate_params(
        DEFAULT_CONFIG["WEIGHT_MODEL_TACTICAL"],
        DEFAULT_CONFIG["HOME_ADVANTAGE"],
    )

    print("\nCurrent parameters:")
    print(
        f"  Tactical: {DEFAULT_CONFIG['WEIGHT_MODEL_TACTICAL']:.2f}, "
        f"Home Adv: {DEFAULT_CONFIG['HOME_ADVANTAGE']:.1f}"
    )
    print(f"  RMSE: {current_metrics['rmse']:.2f} points")
    print(f"  Accuracy: {current_metrics['accuracy']:.1f}%")

    print("\nOptimized parameters:")
    print(
        f"  Tactical: {best_params['tactical_weight']:.2f}, "
        f"Home Adv: {best_params['home_advantage']:.1f}"
    )
    print(f"  RMSE: {best_params['rmse']:.2f} points")
    print(f"  Accuracy: {best_params['accuracy_%']:.1f}%")

    improvement = (
        (current_metrics["rmse"] - best_params["rmse"]) / current_metrics["rmse"]
    ) * 100
    print(f"\nRMSE improvement: {improvement:+.1f}%")

    results_df.to_csv("optimization_results.csv", index=False)
    print("\nResults saved to: optimization_results.csv")
    print("=" * 70 + "\n")

    return best_params, results_df


if __name__ == "__main__":
    try:
        grid_search()
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as exc:
        print(f"\nError during optimization: {exc}")
        import traceback

        traceback.print_exc()
