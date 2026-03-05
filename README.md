# Six Nations Match Predictor — Hybrid Model (Tactical + ELO)

A rugby match prediction system that combines tactical squad analysis with World Rugby ELO rankings to forecast Six Nations match outcomes.

## Overview

This project predicts Six Nations fixtures using a hybrid model that blends two complementary signals:

- **Tactical squad strength** — derived from a player skill database, split across pack, control (halfbacks), and strike (backs)
- **World Rugby ELO ratings** — historical team strength over time
- **Starting XV vs bench differentiation** — bench impact weighted at 20%
- **Physical mismatch detection** — continuous pack dominance bonus

Hyperparameter tuning is supported via a built-in grid search module backtested against real results.

---

## Project Structure

```
six-nations-predictor/
│
├── data/
│   ├── six_nations_squads_FINAL.csv    # Player database (skills, positions, clubs)
│   ├── England-Wales.xlsx              # Matchday squads (one file per fixture)
│   ├── France-Ireland.xlsx
│   ├── Ireland-Italy.xlsx
│   ├── Italy-Scotland.xlsx
│   ├── Scotland-England.xlsx
│   └── Wales-France.xlsx
│
├── src/
│   ├── config.py                       # Central config: paths, ELO ratings, hyperparameters
│   ├── squad_builder.py                # Builds match_ready_squads.csv from a fixture file
│   ├── match_predictor.py              # Runs the hybrid prediction model
│   ├── optimize_params.py             # Grid search over model hyperparameters
│   └── generate_dashboard.py          # Generates dashboard.html from live match data
│
├── match_ready_squads.csv              # ⚡ Generated — not committed
├── optimization_results.csv           # ⚡ Generated — not committed
├── dashboard.html                      # ⚡ Generated — not committed
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/six-nations-predictor.git
cd six-nations-predictor

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

```text
pandas>=1.0.0
numpy>=1.18.0
openpyxl>=3.0.0
scikit-learn>=0.22.0
```

---

## Data Inputs

### Player database

`data/six_nations_squads_FINAL.csv` contains:

- `country` — national team (France, Ireland, etc.)
- `name` — player name
- `position_group` — Forwards or Backs
- `club` — current club
- `league_tier` — PREM, TOP14, URC_A, or URC_B
- `rank_top100` — World Rugby top 100 ranking (if applicable)
- `skill` — numeric rating derived from ranking or league tier
- `role` — SPINE (key player) or STD
- `status` — SQUAD

Player skill is assigned as follows:
1. If the player appears in the World Rugby top 100, skill is calculated from their ranking position
2. Otherwise, a fallback is applied based on their club's league tier: PREM → 76, TOP14 → 78, URC_A → 77, URC_B → 71

### Match lineup files

Each `data/<Home>-<Away>.xlsx` file contains:

- `Number` — shirt number (1–23)
- `<Home team>` — player names for the home side
- `<Away team>` — player names for the away side

The filename determines home/away assignment — the first team named is always home.

---

## Pipeline

### Step 1 — Build match squads

```bash
python src/squad_builder.py France-Ireland.xlsx
```

This script:
- Loads the player database and the specified fixture file
- Matches player names using a three-level lookup: exact → partial → initial + surname
- Falls back to a league-tier skill for players not found in the database
- Tags players as starting (shirt 1–15) or bench (16–23)
- Validates that each team has exactly 15 starters and no duplicate shirt numbers
- Writes `match_ready_squads.csv`

### Step 2 — Run prediction

```bash
python src/match_predictor.py France-Ireland.xlsx
```

This script:
1. Loads `match_ready_squads.csv`
2. Computes tactical metrics per team (pack, control, strike) for starters and bench
3. Computes an ELO-based margin from current World Rugby ratings
4. Fuses tactical and ELO margins using configurable weights
5. Adds home advantage
6. Converts the final margin to a win probability via normal CDF
7. Prints the predicted winner, margin, probability, and breakdown

---

## Model Details

### Tactical component

Three unit scores are computed per team:

| Unit | Players | Default weight |
|------|---------|---------------|
| Pack | All forwards | 40% |
| Control | Halfbacks (shirt 9–10) | 35% |
| Strike | Remaining backs | 25% |

Starting XV and bench are scored separately, then combined:

```
TOTAL_TACTICAL = STARTERS × 0.80 + BENCH × 0.20
TACTICAL_MARGIN = (HOME - AWAY) × TACTICAL_SCALING_FACTOR (0.85)
```

A continuous physical mismatch bonus is added when pack difference exceeds a threshold — linearly ramped up to a maximum of 5 points (no abrupt step).

### ELO component

```
ELO_MARGIN = (ELO_HOME - ELO_AWAY) × ELO_SCALING (0.70)
```

ELO ratings are defined in `src/config.py` and should be updated before each tournament.

### Hybrid fusion

```
FINAL_MARGIN = TACTICAL_MARGIN × 0.50 + ELO_MARGIN × 0.50 + HOME_ADVANTAGE (5.0)
P(home win) = Φ(FINAL_MARGIN / STD_DEV)
```

Weights above reflect the optimised configuration from grid search over the 2025 Six Nations (rounds 1–2). All parameters can be tuned in `src/config.py`.

---

## Dashboard

```bash
python src/generate_dashboard.py
```

Generates `dashboard.html` — a self-contained visual report showing:

- **Backtesting chart** — predicted vs actual margin for every match, with winner correctness badges
- **Team strength ranking** — switchable between ELO, Pack, Control, and Strike metrics
- **Model performance KPIs** — accuracy, RMSE, MAE, and current parameter settings

The dashboard is also regenerated automatically at the end of every `optimize_params.py` run, using the best parameters found in that session. Open `dashboard.html` in any browser — no server required.

> `dashboard.html` is listed in `.gitignore` and should not be committed. Regenerate it locally whenever needed.

---

## Hyperparameter Optimisation

```bash
python src/optimize_params.py
```

Runs a grid search over `tactical_weight` and `home_advantage` against real Six Nations results defined in `REAL_RESULTS`. Outputs:

- Top 10 configurations by RMSE
- Best configuration with full metrics (RMSE, MAE, accuracy)
- Comparison against current default parameters
- `optimization_results.csv` with all evaluated combinations

Current best (2025, rounds 1–2): **6/6 correct winners**, RMSE 20.6 pts.

---

## Example Output

```
SIX NATIONS MATCH PREDICTION
============================================================
Match:             France vs Ireland
Predicted Winner:  IRELAND
Predicted Margin:  3.2 points
Win Probability:   62.4%

Breakdown:
  Tactical Advantage (Squad): +1.8 pts
  Historical Advantage (ELO): -0.5 pts
  Pack mismatch bonus (Ireland): +2.1 pts
------------------------------------------------------------
```

---

## Configuration

All model parameters live in `src/config.py`:

```python
WR_ELO = { "Ireland": 87.97, "France": 87.24, ... }  # Update before each tournament

DEFAULT_CONFIG = {
    "HOME_ADVANTAGE":          5.0,
    "WEIGHT_MODEL_TACTICAL":   0.50,
    "WEIGHT_MODEL_ELO":        0.50,
    ...
}
```

---

## Roadmap

**Short term**
- Extend backtesting to full 2023–2025 Six Nations data
- Add unit tests for core functions (name matching, metrics, tactical scoring)
- Log predictions and errors per match for longitudinal analysis

**Medium term**
- CLI flags to specify fixture, config overrides, and output path
- Visualisations: margin distributions, calibration curves, ELO evolution
- Expand to other international tournaments (Autumn Nations, Rugby Championship)

---

## Limitations

- Player skills are an abstract rating, not derived from match performance data
- Hyperparameter tuning is based on a small sample (6 matches)
- Squad data for past seasons is not automatically reconstructed
- Model does not account for injuries, weather, or referee tendencies

---

## Requirements

- Python 3.10+
- pandas, numpy, openpyxl, scikit-learn

---

## License

MIT License — see `LICENSE` for details.