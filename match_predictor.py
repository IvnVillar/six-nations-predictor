import pandas as pd
import numpy as np
import math

# Configuration
DEFAULT_CONFIG = {
    # Environmental factors
    "HOME_ADVANTAGE": 2.0,      
    "STD_DEV": 13.5,            

    # Hybrid model weights
    "WEIGHT_MODEL_TACTICAL": 0.65, 
    "WEIGHT_MODEL_ELO": 0.35,      

    # Internal tactical weights (units)
    "WEIGHT_PACK": 0.40,
    "WEIGHT_CONTROL": 0.35,
    "WEIGHT_STRIKE": 0.25,
    
    # Starting XV vs Bench weight
    "WEIGHT_STARTERS": 0.80,
    "WEIGHT_BENCH": 0.20,

    # Fine-tuning factors
    "TACTICAL_SCALING_FACTOR": 0.85,
    "ELO_SCALING": 0.7,          
    "MISMATCH_THRESHOLD": 4.0,   
    "MISMATCH_BONUS": 5.0        
}

# Current ELO rankings (World Rugby)
WR_ELO = {
    "Ireland": 87.97, 
    "France": 87.24, 
    "England": 89.41, 
    "Scotland": 80.22, 
    "Italy": 78.98, 
    "Wales": 74.23
}


def get_unit_average(df_subset, unit_type):
    """Calculates average skill for a unit group (Pack/Control/Strike)"""
    if df_subset.empty:
        return 70.0
    
    if unit_type == "Pack":
        # All forwards
        pack = df_subset[df_subset['position_group'] == 'Forwards']
        return pack['skill'].mean() if not pack.empty else 70.0
    
    elif unit_type == "Control":
        # Halfbacks: try shirt number first (9-10)
        if 'shirt_number' in df_subset.columns:
            halfbacks = df_subset[df_subset['shirt_number'].isin([9, 10])]
            if not halfbacks.empty:
                return halfbacks['skill'].mean()
        
        # Fallback: average of all backs
        backs = df_subset[df_subset['position_group'] == 'Backs']
        return backs['skill'].mean() if not backs.empty else 70.0
        
    elif unit_type == "Strike":
        # Backs except halfbacks
        backs = df_subset[df_subset['position_group'] == 'Backs']
        if 'shirt_number' in df_subset.columns:
            strike = backs[~backs['shirt_number'].isin([9, 10])]
            if not strike.empty:
                return strike['skill'].mean()
        return backs['skill'].mean() if not backs.empty else 70.0
    
    return 70.0


def get_tactical_metrics(df, country, params):
    """Calculates combined strength (Starting XV + Bench) and returns key metrics"""
    squad = df[df['country'] == country]
    if squad.empty:
        return None

    # Separate starting XV and bench
    starters = squad[squad['starting'] == 1]
    bench = squad[squad['starting'] == 0]

    # Calculate starting XV strength
    xv_pack = get_unit_average(starters[starters['position_group']=='Forwards'], "Pack")
    xv_control = get_unit_average(starters[starters['position_group']=='Backs'], "Control")
    xv_strike = get_unit_average(starters[starters['position_group']=='Backs'], "Strike")
    
    score_xv = (xv_pack * params["WEIGHT_PACK"] + 
                xv_control * params["WEIGHT_CONTROL"] + 
                xv_strike * params["WEIGHT_STRIKE"])

    # Calculate bench strength
    bn_pack = get_unit_average(bench[bench['position_group']=='Forwards'], "Pack")
    bn_control = get_unit_average(bench[bench['position_group']=='Backs'], "Control")
    bn_strike = get_unit_average(bench[bench['position_group']=='Backs'], "Strike")

    score_bench = (bn_pack * params["WEIGHT_PACK"] + 
                   bn_control * params["WEIGHT_CONTROL"] + 
                   bn_strike * params["WEIGHT_STRIKE"])

    # Final weighting
    total_score = (score_xv * params["WEIGHT_STARTERS"] + 
                   score_bench * params["WEIGHT_BENCH"])

    return {
        "total_score": total_score,
        "pack_xv": xv_pack,
        "pack_bench": bn_pack
    }


def simulate_match(home, away, df, params=DEFAULT_CONFIG):
    """Simulates a match and returns prediction of margin and win probability"""
    # 1. Get tactical metrics
    h_metrics = get_tactical_metrics(df, home, params)
    a_metrics = get_tactical_metrics(df, away, params)
    
    if not h_metrics or not a_metrics:
        return {"error": f"Missing data in CSV for {home} or {away}"}

    # 2. Tactical calculation
    margin_tactical = ((h_metrics["total_score"] - a_metrics["total_score"]) * 
                       params["TACTICAL_SCALING_FACTOR"])
    
    # Physical mismatch bonus (starting pack + bench)
    pack_total_home = 0.7 * h_metrics["pack_xv"] + 0.3 * h_metrics["pack_bench"]
    pack_total_away = 0.7 * a_metrics["pack_xv"] + 0.3 * a_metrics["pack_bench"]
    pack_diff = pack_total_home - pack_total_away
    
    if pack_diff > params["MISMATCH_THRESHOLD"]:
        margin_tactical += params["MISMATCH_BONUS"]
    elif pack_diff < -params["MISMATCH_THRESHOLD"]:
        margin_tactical -= params["MISMATCH_BONUS"]

    # 3. ELO calculation
    elo_h = WR_ELO.get(home, 75.0)
    elo_a = WR_ELO.get(away, 75.0)
    margin_elo = (elo_h - elo_a) * params["ELO_SCALING"]

    # 4. Hybrid fusion
    final_margin = (margin_tactical * params["WEIGHT_MODEL_TACTICAL"] + 
                    margin_elo * params["WEIGHT_MODEL_ELO"] + 
                    params["HOME_ADVANTAGE"])

    # 5. Calculate probabilities
    z_score = final_margin / params["STD_DEV"]
    prob_home = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

    return {
        "home": home,
        "away": away,
        "margin": final_margin,
        "prob": prob_home,
        "debug": {
            "tactical": margin_tactical,
            "elo": margin_elo,
            "pack_diff": pack_diff
        }
    }


# Main execution
if __name__ == "__main__":
    INPUT_FILE = "match_ready_squads.csv"
    
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"\nData loaded: {INPUT_FILE} ({len(df)} records)")
    except FileNotFoundError:
        print(f"ERROR: Cannot find '{INPUT_FILE}'. Run prepare_match.py first")
        exit()

    print("\nSIX NATIONS MATCH PREDICTION\n")
    
    teams_available = df['country'].unique()
    print(f"Teams found in CSV: {teams_available}")

    if len(teams_available) != 2:
        print(f"ERROR: Expected exactly 2 teams in CSV, found {len(teams_available)}: {teams_available}")
        exit()

    home, away = teams_available[0], teams_available[1]
    matches = [(home, away)]

    for h, a in matches:
        res = simulate_match(h, a, df)
        
        if "error" in res:
            print(res["error"])
            continue

        winner = h if res['margin'] > 0 else a
        prob = res['prob'] if res['margin'] > 0 else 1 - res['prob']
        
        print(f"Match: {h} vs {a}")
        print(f"Predicted Winner: {winner.upper()}")
        print(f"Predicted Margin: {abs(res['margin']):.1f} points")
        print(f"Win Probability: {prob*100:.1f}%")
        
        print(f"\nBreakdown:")
        print(f"Tactical Advantage (Squad): {res['debug']['tactical']:+.1f} points")
        print(f"Historical Advantage (ELO): {res['debug']['elo']:+.1f} points")
        
        if abs(res['debug']['pack_diff']) > DEFAULT_CONFIG["MISMATCH_THRESHOLD"]:
            dom = h if res['debug']['pack_diff'] > 0 else a
            print(f"ALERT: {dom} has critical physical advantage in the pack.")
            
        print("-" * 60)
