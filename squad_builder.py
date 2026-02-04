import pandas as pd
import numpy as np

# File names
DB_FILE = "six_nations_squads_FINAL.csv"
MATCH_FILE = "France-Ireland.xlsx"
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


def process_match_file():
    print(f"Loading database: {DB_FILE}")
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

        df_match = df_match.dropna(subset=['Number'])
        
    except FileNotFoundError:
        print(f"ERROR: Cannot find file '{MATCH_FILE}'")
        return
    except Exception as e:
        print(f"Critical error reading file: {e}")
        return

    match_squad = []
    print("Crossing data and assigning roles (Starting XV vs Bench)...\n")

    # Calculate fallback skills per country
    fallback_fr = get_country_fallback_skill("France", df_db)
    fallback_ie = get_country_fallback_skill("Ireland", df_db)

    for _, row in df_match.iterrows():
        try:
            dorsal = int(row['Number'])
        except:
            continue 
            
        is_starting = 1 if dorsal <= 15 else 0

        # Process France
        if 'France' in row and not pd.isna(row['France']):
            player_fr = row['France']
            data = match_player_data(player_fr, "France", df_db)
            
            if data is not None:
                match_squad.append({
                    "country": "France",
                    "name": data['name'],
                    "position_group": data['position_group'],
                    "skill": data['skill'],
                    "starting": is_starting,
                    "shirt_number": dorsal
                })
            else:
                print(f"WARNING: Missing data for {player_fr} (FR). Using skill {fallback_fr:.1f}")
                pos = "Forwards" if dorsal in [1,2,3,4,5,6,7,8,16,17,18,19,20] else "Backs"
                match_squad.append({
                    "country": "France",
                    "name": player_fr,
                    "position_group": pos,
                    "skill": fallback_fr,
                    "starting": is_starting,
                    "shirt_number": dorsal
                })

        # Process Ireland
        if 'Ireland' in row and not pd.isna(row['Ireland']):
            player_ie = row['Ireland']
            data = match_player_data(player_ie, "Ireland", df_db)
            
            if data is not None:
                match_squad.append({
                    "country": "Ireland",
                    "name": data['name'],
                    "position_group": data['position_group'],
                    "skill": data['skill'],
                    "starting": is_starting,
                    "shirt_number": dorsal
                })
            else:
                print(f"WARNING: Missing data for {player_ie} (IE). Using skill {fallback_ie:.1f}")
                pos = "Forwards" if dorsal in [1,2,3,4,5,6,7,8,16,17,18,19,20] else "Backs"
                match_squad.append({
                    "country": "Ireland",
                    "name": player_ie,
                    "position_group": pos,
                    "skill": fallback_ie,
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
    print(f"  - France: {len(df_ready[df_ready['country']=='France'])} players")
    print(f"  - Ireland: {len(df_ready[df_ready['country']=='Ireland'])} players")
    print("=" * 60)


if __name__ == "__main__":
    process_match_file()
