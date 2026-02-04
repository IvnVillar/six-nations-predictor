
# Six Nations Match Predictor - Hybrid Model (Tactical + ELO)

A rugby match prediction system that combines tactical squad analysis with World Rugby ELO rankings to forecast Six Nations match outcomes.

## Overview

This project focuses on a single fixture (France vs Ireland) as a working prototype of a broader Six Nations prediction engine. It uses:

- Tactical squad strength, derived from a player skill database
- World Rugby style ELO rankings for each national team
- A hybrid model that blends tactical and ELO components
- Differentiation between starting XV and bench impact

Hyperparameter tuning and full backtesting are planned but not finalized. For now, the focus is on a clean, reproducible pipeline and a transparent model structure.

---

## Project Structure

Suggested simple layout (everything in the repo root):

```
six-nations-predictor/
├── six_nations_squads_FINAL.csv      # Player database (skills, positions, country)
├── France-Ireland.xlsx               # Matchday 23: France vs Ireland lineup
├── prepare_match.py                  # Script 1: builds match_ready_squads.csv
├── match_predictor.py                # Script 2: runs the hybrid prediction
├── match_ready_squads.csv            # Generated intermediate file (not committed)
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git ignore rules
```

You can later move to a `data/` and `src/` layout if the project grows.

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/six-nations-predictor.git
cd six-nations-predictor

# Create and activate a virtual environment (optional but recommended)
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

`six_nations_squads_FINAL.csv` contains:

- `country`: national team (France, Ireland, etc.)
- `name`: player name
- `position_group`: Forwards or Backs
- `skill`: numeric rating (approximate ability)
- Additional metadata (club, league, rank, role, status)

This file acts as the core source of player-level strength.

### Match lineup

`France-Ireland.xlsx` contains:

- `Number`: shirt number (1–23)
- `France`: player names for France
- `Ireland`: player names for Ireland

It represents the matchday 23 (starting XV + bench) for each team.

---

## Pipeline

### 1. Squad preparation

Run:

```bash
python prepare_match.py
```

This script:

- Loads `six_nations_squads_FINAL.csv`
- Loads `France-Ireland.xlsx`
- Matches player names to the database, with:
  - Normalized name matching (case and whitespace)
  - Partial matching fallback for minor differences
  - Country-specific fallback skill if a player is missing
- Tags:
  - `starting`: 1 for shirt numbers 1–15, 0 for 16–23
  - `shirt_number`: jersey number
- Performs basic validation:
  - Checks that each team has exactly 15 starters
  - Detects duplicate shirt numbers per country
- Writes `match_ready_squads.csv`

Output schema:

- `country`
- `name`
- `position_group` (Forwards / Backs)
- `skill`
- `starting` (1 = starting XV, 0 = bench)
- `shirt_number`

### 2. Match prediction

Run:

```bash
python match_predictor.py
```

This script:

1. Loads `match_ready_squads.csv`
2. Computes tactical metrics:
   - Pack strength (Forwards)
   - Control (halfbacks, 9–10 when identifiable)
   - Strike (backs excluding halfbacks)
   - Separate scores for:
     - Starting XV
     - Bench
   - Aggregates into a single tactical score:
     - Starting XV weight: 80%
     - Bench weight: 20%
3. Computes an ELO-based margin using predefined ratings
4. Combines tactical margin and ELO margin in a hybrid model:
   - Tactical weight: 0.65
   - ELO weight: 0.35
   - Adds a fixed home advantage
5. Converts final margin into a win probability using a normal CDF
6. Prints:
   - Predicted winner
   - Expected margin (points)
   - Win probability
   - Tactical vs ELO breakdown
   - Pack mismatch alerts (if physical gap exceeds a threshold)

---

## Model Details

### Tactical component

For each team, the model computes three unit scores:

- Pack: average skill of all forwards
- Control: average skill of halfbacks (9–10 where identifiable)
- Strike: average skill of remaining backs

These are combined with weights:

- `WEIGHT_PACK = 0.45`
- `WEIGHT_CONTROL = 0.35`
- `WEIGHT_STRIKE = 0.20`

Then the model computes:

- Starting XV score
- Bench score

and combines them as:

- `TOTAL_TACTICAL = STARTERS * 0.80 + BENCH * 0.20`

Finally:

- `TACTICAL_MARGIN = (TOTAL_TACTICAL_HOME - TOTAL_TACTICAL_AWAY) * TACTICAL_SCALING_FACTOR`

with `TACTICAL_SCALING_FACTOR = 0.85`.

A physical mismatch bonus is added if the pack strength difference exceeds a threshold:

- `MISMATCH_THRESHOLD = 4.0`
- `MISMATCH_BONUS = 5.0` points

### ELO component

Uses a simple transformation of World Rugby-style ratings:

- `ELO_MARGIN = (ELO_HOME - ELO_AWAY) * ELO_SCALING`

with:

- `ELO_SCALING = 0.7`

### Hybrid margin and probability

The final predicted margin is:

- `FINAL_MARGIN = TACTICAL_MARGIN * WEIGHT_MODEL_TACTICAL + ELO_MARGIN * WEIGHT_MODEL_ELO + HOME_ADVANTAGE`

with:

- `HOME_ADVANTAGE = 2.0`
- `WEIGHT_MODEL_TACTICAL = 0.65`
- `WEIGHT_MODEL_ELO = 0.35`

Win probability for the home team is:

- `P(home win) = Φ(FINAL_MARGIN / STD_DEV)`

using a normal CDF with:

- `STD_DEV = 13.5` points

---

## Example Output

Example console output for France vs Ireland:

```text
Data loaded: match_ready_squads.csv (46 records)

SIX NATIONS MATCH PREDICTION

Match: France vs Ireland
Predicted Winner: IRELAND
Predicted Margin: 3.2 points
Win Probability: 62.4%

Breakdown:
Tactical Advantage (Squad): +1.8 points
Historical Advantage (ELO): -0.5 points
ALERT: Ireland has critical physical advantage in the pack.
------------------------------------------------------------
```

Values are illustrative; actual numbers depend on the player database and lineup.

---

## Hyperparameter Tuning (Work in Progress)

There is an ongoing effort to:

- Use Six Nations historical data (2023–2025) for backtesting
- Evaluate RMSE, MAE, and match outcome accuracy
- Tune:
  - Tactical vs ELO weights
  - Pack/Control/Strike weights
  - Home advantage
  - Mismatch thresholds and bonuses

For now, parameters are based on rugby domain heuristics rather than fully optimized.

---

## Limitations

- Current public version is calibrated around a single fixture (France vs Ireland).
- Player skills are an abstract rating, not derived from a formal model in this repository.
- Hyperparameter tuning and cross-validation are not yet integrated.
- Squads for past seasons are not automatically reconstructed.

---

## Requirements

- Python 3.7+
- pandas 1.0.0+
- numpy 1.18.0+
- openpyxl 3.0.0+
- scikit-learn 0.22.0+

Install via:

```bash
pip install -r requirements.txt
```

---

## Roadmap

Short term:

- Add a backtesting module against 2023–2025 results
- Introduce a simple hyperparameter search over a small, meaningful grid
- Log predictions and errors per match for analysis

Medium term:

- Generalize to all Six Nations fixtures
- Parameterize input paths and teams from the command line
- Add plotting utilities for margin distributions and calibration

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.
