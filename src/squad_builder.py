"""
squad_builder.py — Builds match_ready_squads.csv from a match Excel file.

Usage:
    python squad_builder.py                          # uses DEFAULT_MATCH_FILE
    python squad_builder.py France-Ireland.xlsx      # explicit match file
"""

import os
import sys
import pandas as pd

from config import DB_FILE, OUTPUT_DIR, match_file_path

# Default match file (can be overridden via CLI argument)
DEFAULT_MATCH_FILE = "France-Ireland.xlsx"

# Jersey numbers that correspond to forward positions in rugby union
_FORWARD_NUMBERS = {1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20}


# ── Shared utilities ───────────────────────────────────────────────────────────

def infer_position_group(jersey_number: int) -> str:
    """
    Infer position group ('Forwards' or 'Backs') from jersey number.

    Rugby union convention:
      1-8   → Forwards (starting)
      9-15  → Backs (starting)
      16-20 → Forwards (bench)
      21-23 → Backs (bench)
    """
    return "Forwards" if jersey_number in _FORWARD_NUMBERS else "Backs"


def normalize_name(name) -> str:
    """Normalise a player name: strip whitespace and lowercase."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def get_teams_from_filename(filename: str):
    """
    Extract (home_team, away_team) from a filename like 'France-Ireland.xlsx'.
    The first team is always considered the HOME side.
    """
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid filename: '{filename}'. Expected format: HomeTeam-AwayTeam.xlsx"
        )
    return parts[0].strip(), parts[1].strip()


def _initial_surname(name: str) -> tuple[str, str]:
    """
    Split a normalised name into (initial, surname).
    Handles both 'firstname surname' and 'F surname' formats.
    Returns ('', '') if the name has fewer than 2 tokens.
    """
    parts = name.split()
    if len(parts) < 2:
        return ("", "")
    return (parts[0][0], parts[-1])   # first char of first token + last token


def match_player_data(player_name: str, country: str, db_df: pd.DataFrame):
    """
    Look up a player in the database and return their row, or None if not found.

    Search order:
      1. Exact match (normalised)
      2. Partial match (one name contains the other — handles middle names / accents)
      3. Initial + surname match (handles 'Finn RUSSELL' ↔ 'F Russell' in DB)
    """
    clean = normalize_name(player_name)
    country_db = db_df[db_df["country"].str.lower() == country.lower()]

    # 1. Exact match
    exact = country_db[country_db["name"].apply(normalize_name) == clean]
    if not exact.empty:
        return exact.iloc[0]

    # 2. Partial match
    for _, row in country_db.iterrows():
        db_name = normalize_name(row["name"])
        if clean in db_name or db_name in clean:
            return row

    # 3. Initial + surname match
    initial_in, surname_in = _initial_surname(clean)
    if initial_in and surname_in:
        for _, row in country_db.iterrows():
            db_name = normalize_name(row["name"])
            initial_db, surname_db = _initial_surname(db_name)
            if surname_in == surname_db and initial_in == initial_db:
                return row

    return None


def get_country_fallback_skill(country: str, db_df: pd.DataFrame) -> float:
    """
    Return the 25th-percentile skill for a country as a conservative fallback
    for players missing from the database.
    """
    country_db = db_df[db_df["country"].str.lower() == country.lower()]
    if country_db.empty:
        return 70.0
    return float(country_db["skill"].quantile(0.25))


# ── Core processing ────────────────────────────────────────────────────────────

def build_match_squads(match_filename: str) -> pd.DataFrame:
    """
    Load a match Excel file, cross-reference the player database, and return
    a DataFrame ready for match_predictor.py.

    Parameters
    ----------
    match_filename : str
        Filename only (e.g. 'France-Ireland.xlsx'). Must live in the data/ folder.

    Returns
    -------
    pd.DataFrame
        Columns: country, name, position_group, skill, starting, shirt_number.
    """
    match_path = match_file_path(match_filename)
    home, away = get_teams_from_filename(match_filename)

    print(f"Teams detected: {home} (home) vs {away} (away)")

    # Load database
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Player database not found: {DB_FILE}")

    try:
        db_df = pd.read_csv(DB_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        db_df = pd.read_csv(DB_FILE, encoding="latin1")

    # Load match lineup
    if not os.path.exists(match_path):
        raise FileNotFoundError(f"Match file not found: {match_path}")

    df_match = pd.read_excel(match_path, engine="openpyxl")

    if "Number" not in df_match.columns:
        raise KeyError(f"'Number' column not found in {match_path}")

    df_match = df_match.dropna(subset=["Number"])

    cols = list(df_match.columns)
    if len(cols) < 3:
        raise ValueError(
            f"Expected at least 3 columns in {match_path}, got {len(cols)}. "
            "Format should be: [HomeTeam, Number, AwayTeam]"
        )

    col_home, col_away = cols[0], cols[2]
    print(f"Columns detected: '{col_home}' | Number | '{col_away}'")

    fallback_home = get_country_fallback_skill(home, db_df)
    fallback_away = get_country_fallback_skill(away, db_df)

    records = []

    print("\nCrossing data and assigning roles (Starting XV vs Bench)...\n")

    for _, row in df_match.iterrows():
        try:
            jersey = int(row["Number"])
        except (ValueError, TypeError):
            continue

        is_starting = 1 if jersey <= 15 else 0

        for player_name, country, fallback, col in [
            (row.get(col_home), home, fallback_home, col_home),
            (row.get(col_away), away, fallback_away, col_away),
        ]:
            if pd.isna(player_name):
                continue

            data = match_player_data(player_name, country, db_df)

            if data is not None:
                records.append({
                    "country":        country,
                    "name":           data["name"],
                    "position_group": data["position_group"],
                    "skill":          data["skill"],
                    "starting":       is_starting,
                    "shirt_number":   jersey,
                })
            else:
                print(
                    f"WARNING: '{player_name}' ({country}) not found in DB. "
                    f"Using fallback skill {fallback:.1f}"
                )
                records.append({
                    "country":        country,
                    "name":           str(player_name),
                    "position_group": infer_position_group(jersey),
                    "skill":          fallback,
                    "starting":       is_starting,
                    "shirt_number":   jersey,
                })

    return pd.DataFrame(records)


def validate_and_save(df: pd.DataFrame, output_file: str) -> None:
    """Run basic integrity checks and write the DataFrame to CSV."""
    print("VALIDATING DATA...\n")

    starters = df[df["starting"] == 1]

    for country in df["country"].unique():
        n = len(starters[starters["country"] == country])
        status = "OK" if n == 15 else "WARNING"
        print(f"{status}: {country} has {n} starting players (expected 15)")

    duplicates = df[df.duplicated(["country", "shirt_number"], keep=False)]
    if not duplicates.empty:
        print("\nWARNING: Duplicate shirt numbers detected:")
        print(duplicates.sort_values(["country", "shirt_number"]))
    else:
        print("OK: No duplicate shirt numbers\n")

    df.to_csv(output_file, index=False)

    teams = df["country"].unique()
    print("=" * 60)
    print(f"SUCCESS: File generated -> {output_file}")
    print(f"Total players: {len(df)}")
    for t in teams:
        print(f"  {t}: {len(df[df['country'] == t])} players")
    print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

def process_match_file(match_filename: str = DEFAULT_MATCH_FILE) -> None:
    output_file = os.path.join(OUTPUT_DIR, "match_ready_squads.csv")

    print(f"Loading player database: {DB_FILE}")
    df = build_match_squads(match_filename)
    validate_and_save(df, output_file)


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MATCH_FILE
    try:
        process_match_file(filename)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)