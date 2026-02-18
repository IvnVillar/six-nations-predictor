import pandas as pd
import numpy as np
import os

# File names
DB_FILE = "six_nations_squads_FINAL.csv"
MATCH_FILE = "England-Wales.xlsx"  
OUTPUT_FILE = "match_ready_squads.csv"


def normalize_name(name):
    """Cleans names to ensure matches (removes uppercase, extra spaces)"""
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def match_player_data(player_name, country, db_df):
    """Searches for a player in the database and returns their stats"""
    clean_name = normalize_name(player_name)

    # Filter by country to avoid duplicate names between nations
    country_db = db_df[db_df['country'].str.lower() == country.lower()]

    # Exact search (normalized)
    found = country_db[country_db['name'].apply(normalize_name) == clean_name]
    if len(found) > 0:
        return found.iloc[0]

    # If not found, try partial search (useful for accents or middle names)
    for idx, row in country_db.iterrows():
        db_name = normalize_name(row['name'])
        if clean_name in db_name or db_name in clean_name:
            return row

    return None


def get_country_fallback_skill(country, db_df):
    """Calculates fallback skill for missing players (25th percentile)"""
    country_db = db_df[db_df['country'].str.lower() == country.lower()]
    if country_db.empty:
        return 70.0
    return country_db['skill'].quantile(0.25)


def get_teams_from_filename(filename):
    """Extracts team names from a filename like 'France-Ireland.xlsx'"""
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    parts = name.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Nombre de archivo no válido: {filename}. Debe ser equipo1-equipo2.xlsx"
        )
    team1 = parts[0].strip()
    team2 = parts[1].strip()
    return team1, team2


def process_match_file():
    print(f"Loading database: {DB_FILE}")

    # Detect teams from filename
    team1, team2 = get_teams_from_filename(MATCH_FILE)
    print(f"Detectados equipos: {team1} vs {team2}")

    try:
        try:
            df_db = pd.read_csv(DB_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            df_db = pd.read_csv(DB_FILE, encoding='latin1')
    except FileNotFoundError:
        print("ERROR: Cannot find six_nations_squads_FINAL.csv")
        return

    print(f"Loading match lineup: {MATCH_FILE}")
    df_match = pd.DataFrame()

    try:
        if MATCH_FILE.endswith('.xlsx') or MATCH_FILE.endswith('.xls'):
            print("Detected Excel format (.xlsx)")
            df_match = pd.read_excel(MATCH_FILE, engine='openpyxl')
        else:
            try:
                df_match = pd.read_csv(MATCH_FILE, encoding='utf-8')
            except UnicodeDecodeError:
                print("Switching encoding to Latin1...")
                df_match = pd.read_csv(MATCH_FILE, encoding='latin1')
            except pd.errors.ParserError:
                print("Format error. Trying semicolon separator...")
                df_match = pd.read_csv(MATCH_FILE, sep=';', encoding='latin1')

        # Aseguramos columna Number y quitamos filas vacías
        df_match = df_match.dropna(subset=['Number'])

    except FileNotFoundError:
        print(f"ERROR: Cannot find file '{MATCH_FILE}'")
        return
    except Exception as e:
        print(f"Critical error reading file: {e}")
        return

    # Detectar nombres reales de columnas de equipos según el Excel
    cols = list(df_match.columns)
    # Asumimos formato [equipo1, 'Number', equipo2]
    if len(cols) < 3:
        print("ERROR: El archivo de alineaciones no tiene al menos 3 columnas (Equipo1, Number, Equipo2)")
        return

    col_team1 = cols[0]
    col_number = 'Number'
    col_team2 = cols[2]

    print(f"Columnas detectadas: {col_team1} (columna equipo1), {col_number} (número), {col_team2} (columna equipo2)")

    match_squad = []

    print("Crossing data and assigning roles (Starting XV vs Bench)...\n")

    # Calculate fallback skills per country
    fallback_t1 = get_country_fallback_skill(team1, df_db)
    fallback_t2 = get_country_fallback_skill(team2, df_db)

    for _, row in df_match.iterrows():
        try:
            dorsal = int(row[col_number])
        except Exception:
            continue

        is_starting = 1 if dorsal <= 15 else 0

        # Equipo 1
        if col_team1 in row and not pd.isna(row[col_team1]):
            player1 = row[col_team1]
            data1 = match_player_data(player1, team1, df_db)

            if data1 is not None:
                match_squad.append({
                    "country": team1,
                    "name": data1['name'],
                    "position_group": data1['position_group'],
                    "skill": data1['skill'],
                    "starting": is_starting,
                    "shirt_number": dorsal
                })
            else:
                print(f"WARNING: Missing data for {player1} ({team1}). Using skill {fallback_t1:.1f}")
                pos = "Forwards" if dorsal in [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20] else "Backs"
                match_squad.append({
                    "country": team1,
                    "name": player1,
                    "position_group": pos,
                    "skill": fallback_t1,
                    "starting": is_starting,
                    "shirt_number": dorsal
                })

        # Equipo 2
        if col_team2 in row and not pd.isna(row[col_team2]):
            player2 = row[col_team2]
            data2 = match_player_data(player2, team2, df_db)

            if data2 is not None:
                match_squad.append({
                    "country": team2,
                    "name": data2['name'],
                    "position_group": data2['position_group'],
                    "skill": data2['skill'],
                    "starting": is_starting,
                    "shirt_number": dorsal
                })
            else:
                print(f"WARNING: Missing data for {player2} ({team2}). Using skill {fallback_t2:.1f}")
                pos = "Forwards" if dorsal in [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20] else "Backs"
                match_squad.append({
                    "country": team2,
                    "name": player2,
                    "position_group": pos,
                    "skill": fallback_t2,
                    "starting": is_starting,
                    "shirt_number": dorsal
                })

    # Save CSV
    df_ready = pd.DataFrame(match_squad)

    # Validation
    print("VALIDATING DATA...\n")
    starters = df_ready[df_ready['starting'] == 1]

    for country in df_ready['country'].unique():
        n_start = len(starters[starters['country'] == country])
        if n_start != 15:
            print(f"WARNING: {country} does not have 15 starting players: has {n_start}")
        else:
            print(f"OK: {country} has 15 starting players")

    duplicates = df_ready[df_ready.duplicated(['country', 'shirt_number'], keep=False)]
    if not duplicates.empty:
        print("\nWARNING: Duplicate shirt numbers detected:")
        print(duplicates.sort_values(['country', 'shirt_number']))
    else:
        print("OK: No duplicate shirt numbers\n")

    df_ready.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print(f"SUCCESS: File generated: {OUTPUT_FILE}")
    print(f"Total players processed: {len(df_ready)}")
    print(f" - {team1}: {len(df_ready[df_ready['country'] == team1])} players")
    print(f" - {team2}: {len(df_ready[df_ready['country'] == team2])} players")
    print("=" * 60)


if __name__ == "__main__":
    process_match_file()

