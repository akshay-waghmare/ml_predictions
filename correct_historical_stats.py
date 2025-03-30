import os
import pandas as pd
import re
from fuzzywuzzy import process  # For fuzzy name matching

def load_aggregated_data(file_path='aggregated_match_data.csv'):
    """Load the aggregated match data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded aggregated data with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading aggregated data: {e}")
        return None

def identify_missing_stats(df):
    """Identify rows with missing or zero bowling stats."""
    # Define columns to check
    bowling_stats_cols = [
        'bowler1_historical_average', 'bowler1_historical_economy', 'bowler1_historical_strike_rate',
        'bowler2_historical_average', 'bowler2_historical_economy', 'bowler2_historical_strike_rate'
    ]
    
    # Create masks for rows with missing or zero stats
    missing_mask = df[bowling_stats_cols].isnull().any(axis=1)
    zero_mask = (df[bowling_stats_cols] == 0).any(axis=1)
    
    # Combine masks and filter DataFrame
    problem_rows = df[missing_mask | zero_mask].copy()
    
    print(f"Identified {len(problem_rows)} rows with missing or zero bowling stats.")
    
    # Extract unique match IDs with problems
    problem_match_ids = problem_rows['match_id'].unique()
    print(f"These rows span {len(problem_match_ids)} unique match IDs.")
    
    return problem_rows, problem_match_ids

def identify_missing_batsman_stats(df):
    """Identify rows with missing or zero batsman stats."""
    # Define columns to check
    batting_stats_cols = [
        'batsman1_historical_average', 'batsman1_historical_strike_rate',
        'batsman2_historical_average', 'batsman2_historical_strike_rate'
    ]
    
    # Create masks for rows with missing or zero stats
    missing_mask = df[batting_stats_cols].isnull().any(axis=1)
    zero_mask = (df[batting_stats_cols] == 0).any(axis=1)
    
    # Combine masks and filter DataFrame
    problem_rows = df[missing_mask | zero_mask].copy()
    
    print(f"Identified {len(problem_rows)} rows with missing or zero batsman stats.")
    
    # Extract unique match IDs with problems
    problem_match_ids = problem_rows['match_id'].unique()
    print(f"These rows span {len(problem_match_ids)} unique match IDs.")
    
    return problem_rows, problem_match_ids

def find_match_specific_bowling_stats(base_dirs, match_id):
    """Find bowling stats files for a specific match ID."""
    bowling_stats_files = []
    match_id_str = str(int(match_id))  # Convert to int then string to remove decimals
    
    for base_dir in base_dirs:
        # Construct the expected path pattern
        # First check exact match folder
        specific_match_path = os.path.join(base_dir, match_id_str, "ball_by_ball")
        
        if os.path.exists(specific_match_path):
            # Look for bowling stats files in this specific match folder
            for file in os.listdir(specific_match_path):
                if '_bowling_stats.csv' in file:
                    file_path = os.path.join(specific_match_path, file)
                    bowling_stats_files.append(file_path)
        else:
            # If exact match folder not found, search recursively
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if '_bowling_stats.csv' in file and match_id_str in root:
                        file_path = os.path.join(root, file)
                        bowling_stats_files.append(file_path)
    
    if bowling_stats_files:
        print(f"Found {len(bowling_stats_files)} bowling stats files for match ID {match_id_str}")
    else:
        print(f"No bowling stats files found for match ID {match_id_str}")
        
    return bowling_stats_files

def find_match_specific_batting_stats(base_dirs, match_id):
    """Find batting stats files for a specific match ID."""
    batting_stats_files = []
    match_id_str = str(int(match_id))  # Convert to int then string to remove decimals
    
    for base_dir in base_dirs:
        # Construct the expected path pattern
        # First check exact match folder
        specific_match_path = os.path.join(base_dir, match_id_str, "ball_by_ball")
        
        if os.path.exists(specific_match_path):
            # Look for batting stats files in this specific match folder
            for file in os.listdir(specific_match_path):
                if '_batting_stats.csv' in file:
                    file_path = os.path.join(specific_match_path, file)
                    batting_stats_files.append(file_path)
        else:
            # If exact match folder not found, search recursively
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if '_batting_stats.csv' in file and match_id_str in root:
                        file_path = os.path.join(root, file)
                        batting_stats_files.append(file_path)
    
    if batting_stats_files:
        print(f"Found {len(batting_stats_files)} batting stats files for match ID {match_id_str}")
    else:
        print(f"No batting stats files found for match ID {match_id_str}")
        
    return batting_stats_files

def load_match_bowling_stats(file_paths):
    """Load bowling stats from CSV files for a specific match."""
    team_stats = {}
    
    for file_path in file_paths:
        try:
            # Extract team name from filename
            team_name = os.path.basename(file_path).replace('_bowling_stats.csv', '')
            
            # Load the data
            df = pd.read_csv(file_path)
            
            team_stats[team_name] = df
            print(f"Loaded bowling stats for team {team_name} with {len(df)} bowlers.")
                
        except Exception as e:
            print(f"Error loading bowling stats from {file_path}: {e}")
    
    return team_stats

def load_match_batting_stats(file_paths):
    """Load batting stats from CSV files for a specific match."""
    team_stats = {}
    
    for file_path in file_paths:
        try:
            # Extract team name from filename
            team_name = os.path.basename(file_path).replace('_batting_stats.csv', '')
            
            # Load the data with a more lenient encoding
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
            # Clean up data - handle special characters and whitespace in columns and data
            df.columns = [col.strip() for col in df.columns]
            
            # Clean the player_name column to handle special characters like MS Dhoni(c)
            if 'player_name' in df.columns:
                df['player_name'] = df['player_name'].apply(
                    lambda x: re.sub(r'[^\w\s]', '', str(x)) if pd.notnull(x) else x
                )
            
            team_stats[team_name] = df
            print(f"Loaded batting stats for team {team_name} with {len(df)} batsmen.")
                
        except Exception as e:
            print(f"Error loading batting stats from {file_path}: {e}")
    
    return team_stats

def extract_surname(full_name):
    """Extract surname from a full name."""
    if pd.isna(full_name) or full_name == '':
        return ''
    
    # Handle 'No Bowler' case
    if full_name == 'No Bowler':
        return 'No Bowler'
    
    # Split the name and return the last part
    parts = full_name.split()
    return parts[-1].lower() if parts else ''

def match_bowler_to_stats(bowler_name, team_stats):
    """
    Match a bowler name from aggregated data to their stats in the bowling stats file.
    Uses fuzzy matching to handle slight naming differences.
    """
    if pd.isna(bowler_name) or bowler_name == 'No Bowler' or team_stats is None:
        return None
    
    # Extract surname for comparison
    bowler_surname = extract_surname(bowler_name)
    
    # Create a list of bowler names from team_stats
    stat_bowler_names = team_stats['player_name'].tolist()
    
    if not stat_bowler_names:
        return None
    
    # Try direct surname matching first
    stat_bowler_surnames = [extract_surname(name) for name in stat_bowler_names]
    for idx, surname in enumerate(stat_bowler_surnames):
        if bowler_surname == surname:
            return team_stats.iloc[idx]
    
    # If direct matching fails, try fuzzy matching
    # This is useful for handling slight spelling variations
    match_result = process.extractOne(bowler_name, stat_bowler_names)
    if match_result and match_result[1] >= 70:  # Match score threshold
        matched_name = match_result[0]
        return team_stats[team_stats['player_name'] == matched_name].iloc[0]
    
    return None

def match_batsman_to_stats(batsman_name, team_stats):
    """
    Match a batsman name from aggregated data to their stats in the batting stats file.
    Uses fuzzy matching to handle slight naming differences.
    """
    if pd.isna(batsman_name) or batsman_name == 'No Partner' or team_stats is None:
        return None
    
    # Extract surname for comparison
    batsman_surname = extract_surname(batsman_name)
    
    # Create a list of batsman names from team_stats
    stat_batsman_names = team_stats['player_name'].tolist()
    
    if not stat_batsman_names:
        return None
    
    # Try direct surname matching first
    stat_batsman_surnames = [extract_surname(name) for name in stat_batsman_names]
    for idx, surname in enumerate(stat_batsman_surnames):
        if batsman_surname == surname:
            return team_stats.iloc[idx]
    
    # If direct matching fails, try fuzzy matching
    # This is useful for handling slight spelling variations
    match_result = process.extractOne(batsman_name, stat_batsman_names)
    if match_result and match_result[1] >= 70:  # Match score threshold
        matched_name = match_result[0]
        return team_stats[team_stats['player_name'] == matched_name].iloc[0]
    
    return None

def correct_historical_stats(aggregated_df, base_dirs):
    """
    Correct historical stats for bowlers in the aggregated data.
    """
    # Create a copy of the aggregated data
    corrected_df = aggregated_df.copy()
    
    # Define the historical stats columns to update
    historical_cols = [
        'bowler1_historical_average', 'bowler1_historical_economy', 'bowler1_historical_strike_rate',
        'bowler2_historical_average', 'bowler2_historical_economy', 'bowler2_historical_strike_rate'
    ]
    
    # Create new columns to store the correction status
    for col in historical_cols:
        corrected_df[f"{col}_corrected"] = False
    
    # Number of stats corrected
    corrections_made = 0
    
    # Identify rows with missing stats
    problem_rows, problem_match_ids = identify_missing_stats(aggregated_df)
    print(f"Processing {len(problem_match_ids)} unique match IDs with missing bowler stats.")
    
    # Process each unique match ID with problems
    for match_id in problem_match_ids:
        # Find bowling stats files for this specific match
        match_bowling_files = find_match_specific_bowling_stats(base_dirs, match_id)
        
        if not match_bowling_files:
            print(f"No bowling stats files found for match ID {match_id}. Skipping.")
            continue
            
        # Load the bowling stats for this match
        match_bowling_stats = load_match_bowling_stats(match_bowling_files)
        
        if not match_bowling_stats:
            print(f"No valid bowling stats loaded for match ID {match_id}. Skipping.")
            continue
            
        # Filter rows for this match ID
        match_rows = problem_rows[problem_rows['match_id'] == match_id]
        
        for idx, row in match_rows.iterrows():
            # Get the batting team to determine bowling team
            batting_team = row['batting_team']
            teams = list(match_bowling_stats.keys())
            
            # Find opposing team (the one that's bowling)
            bowling_team = None
            for team in teams:
                if batting_team not in team:  # Simple heuristic - can be improved
                    bowling_team = team
                    break
            
            if not bowling_team:
                print(f"Could not identify bowling team for match {match_id}, batting team {batting_team}")
                continue
                
            team_stats = match_bowling_stats.get(bowling_team)
            
            if team_stats is None:
                print(f"No stats found for bowling team in match {match_id}")
                continue
            
            # Update stats for bowler1
            bowler1_name = row['bowler1_name']
            if not pd.isna(bowler1_name) and bowler1_name != 'No Bowler':
                bowler1_stats = match_bowler_to_stats(bowler1_name, team_stats)
                
                if bowler1_stats is not None:
                    # Convert to numeric to handle string values
                    corrected_df.at[idx, 'bowler1_historical_average'] = pd.to_numeric(bowler1_stats['average'], errors='coerce')
                    corrected_df.at[idx, 'bowler1_historical_economy'] = pd.to_numeric(bowler1_stats['economy'], errors='coerce')
                    corrected_df.at[idx, 'bowler1_historical_strike_rate'] = pd.to_numeric(bowler1_stats['strike_rate'], errors='coerce')
                    
                    corrected_df.at[idx, 'bowler1_historical_average_corrected'] = True
                    corrected_df.at[idx, 'bowler1_historical_economy_corrected'] = True
                    corrected_df.at[idx, 'bowler1_historical_strike_rate_corrected'] = True
                    
                    corrections_made += 3
                    print(f"Updated stats for bowler1 {bowler1_name} in match {match_id}")
            
            # Update stats for bowler2
            bowler2_name = row['bowler2_name']
            if not pd.isna(bowler2_name) and bowler2_name != 'No Bowler':
                bowler2_stats = match_bowler_to_stats(bowler2_name, team_stats)
                
                if bowler2_stats is not None:
                    # Convert to numeric to handle string values
                    corrected_df.at[idx, 'bowler2_historical_average'] = pd.to_numeric(bowler2_stats['average'], errors='coerce')
                    corrected_df.at[idx, 'bowler2_historical_economy'] = pd.to_numeric(bowler2_stats['economy'], errors='coerce')
                    corrected_df.at[idx, 'bowler2_historical_strike_rate'] = pd.to_numeric(bowler2_stats['strike_rate'], errors='coerce')
                    
                    corrected_df.at[idx, 'bowler2_historical_average_corrected'] = True
                    corrected_df.at[idx, 'bowler2_historical_economy_corrected'] = True
                    corrected_df.at[idx, 'bowler2_historical_strike_rate_corrected'] = True
                    
                    corrections_made += 3
                    print(f"Updated stats for bowler2 {bowler2_name} in match {match_id}")
    
    print(f"Made {corrections_made} corrections to historical bowling stats.")
    return corrected_df

def correct_batsman_historical_stats(aggregated_df, base_dirs):
    """
    Correct historical stats for batsmen in the aggregated data.
    Uses a combined approach that doesn't rely on team name matching.
    """
    # Create a copy of the aggregated data
    corrected_df = aggregated_df.copy()
    
    # Define the historical stats columns to update
    historical_cols = [
        'batsman1_historical_average', 'batsman1_historical_strike_rate',
        'batsman2_historical_average', 'batsman2_historical_strike_rate'
    ]
    
    # Create new columns to store the correction status
    for col in historical_cols:
        corrected_df[f"{col}_corrected"] = False
    
    # Number of stats corrected
    corrections_made = 0
    
    # Identify rows with missing stats
    problem_rows, problem_match_ids = identify_missing_batsman_stats(aggregated_df)
    print(f"Processing {len(problem_match_ids)} unique match IDs with missing batsman stats.")
    
    # Process each unique match ID with problems
    for match_id in problem_match_ids:
        # Find batting stats files for this specific match
        match_batting_files = find_match_specific_batting_stats(base_dirs, match_id)
        
        if not match_batting_files:
            print(f"No batting stats files found for match ID {match_id}. Skipping.")
            continue
            
        # Load the batting stats for this match
        match_batting_stats = load_match_batting_stats(match_batting_files)
        
        if not match_batting_stats:
            print(f"No valid batting stats loaded for match ID {match_id}. Skipping.")
            continue
        
        # NEW: Create a combined DataFrame with all batting stats from all teams
        combined_stats = pd.DataFrame()
        for team, stats_df in match_batting_stats.items():
            # Add team name column to keep track of which team the player belongs to
            stats_df = stats_df.copy()
            stats_df['team'] = team
            combined_stats = pd.concat([combined_stats, stats_df], ignore_index=True)
        
        print(f"Created combined batting stats with {len(combined_stats)} players for match {match_id}")
        
        # Filter rows for this match ID
        match_rows = problem_rows[problem_rows['match_id'] == match_id]
        
        for idx, row in match_rows.iterrows():
            # Update stats for batsman1
            batsman1_name = row['batsman1_name']
            if not pd.isna(batsman1_name):
                # Find batsman in combined stats using the existing match function
                # but passing the combined DataFrame instead of a team-specific one
                batsman1_stats = find_player_in_combined_stats(batsman1_name, combined_stats)
                
                if batsman1_stats is not None:
                    try:
                        # Convert to numeric to handle string values
                        corrected_df.at[idx, 'batsman1_historical_average'] = pd.to_numeric(batsman1_stats['average'], errors='coerce')
                        corrected_df.at[idx, 'batsman1_historical_strike_rate'] = pd.to_numeric(batsman1_stats['strike_rate'], errors='coerce')
                        
                        corrected_df.at[idx, 'batsman1_historical_average_corrected'] = True
                        corrected_df.at[idx, 'batsman1_historical_strike_rate_corrected'] = True
                        
                        corrections_made += 2
                        print(f"Updated stats for batsman1 {batsman1_name} in match {match_id} (team: {batsman1_stats.get('team', 'unknown')})")
                    except Exception as e:
                        print(f"Error updating stats for batsman1 {batsman1_name}: {e}")
                        print(f"Available columns in stats: {batsman1_stats.index.tolist()}")
            
            # Update stats for batsman2
            batsman2_name = row['batsman2_name']
            if not pd.isna(batsman2_name) and batsman2_name != 'No Partner':
                batsman2_stats = find_player_in_combined_stats(batsman2_name, combined_stats)
                
                if batsman2_stats is not None:
                    try:
                        # Convert to numeric to handle string values
                        corrected_df.at[idx, 'batsman2_historical_average'] = pd.to_numeric(batsman2_stats['average'], errors='coerce')
                        corrected_df.at[idx, 'batsman2_historical_strike_rate'] = pd.to_numeric(batsman2_stats['strike_rate'], errors='coerce')
                        
                        corrected_df.at[idx, 'batsman2_historical_average_corrected'] = True
                        corrected_df.at[idx, 'batsman2_historical_strike_rate_corrected'] = True
                        
                        corrections_made += 2
                        print(f"Updated stats for batsman2 {batsman2_name} in match {match_id} (team: {batsman2_stats.get('team', 'unknown')})")
                    except Exception as e:
                        print(f"Error updating stats for batsman2 {batsman2_name}: {e}")
                        print(f"Available columns in stats: {batsman2_stats.index.tolist()}")
    
    print(f"Made {corrections_made} corrections to historical batting stats.")
    return corrected_df

def find_player_in_combined_stats(player_name, combined_stats):
    """
    Find a player in the combined stats DataFrame using surname matching and fuzzy matching.
    
    Args:
        player_name: Name of the player to find
        combined_stats: DataFrame containing stats from all teams
        
    Returns:
        Series: The player's stats if found, None otherwise
    """
    if pd.isna(player_name) or player_name == 'No Partner' or combined_stats.empty:
        return None
    
    # Extract surname for comparison
    player_surname = extract_surname(player_name)
    
    # Get all player names from the combined stats
    all_player_names = combined_stats['player_name'].tolist()
    
    # Try direct surname matching first
    all_surnames = [extract_surname(name) for name in all_player_names]
    for idx, surname in enumerate(all_surnames):
        if player_surname == surname:
            return combined_stats.iloc[idx]
    
    # If direct matching fails, try fuzzy matching
    match_result = process.extractOne(player_name, all_player_names)
    if match_result and match_result[1] >= 70:  # Match score threshold
        matched_name = match_result[0]
        return combined_stats[combined_stats['player_name'] == matched_name].iloc[0]
    
    return None

def update_batsman_stats(df, idx, batsman_num, stats):
    """Update historical stats for a specific batsman."""
    try:
        # Convert stats to numeric values to handle any string formatting
        avg = pd.to_numeric(stats['average'], errors='coerce')
        sr = pd.to_numeric(stats['strike_rate'], errors='coerce')
        
        # Update the stats
        df.at[idx, f'batsman{batsman_num}_historical_average'] = avg
        df.at[idx, f'batsman{batsman_num}_historical_strike_rate'] = sr
        
        # Mark as corrected
        df.at[idx, f'batsman{batsman_num}_historical_average_corrected'] = True
        df.at[idx, f'batsman{batsman_num}_historical_strike_rate_corrected'] = True
        
        print(f"Updated stats for batsman{batsman_num} to avg={avg}, sr={sr}")
        return True
    except Exception as e:
        print(f"Error updating batsman stats: {e}")
        return False

def clean_corrected_data(corrected_df):
    """
    Clean the corrected data by removing placeholder stats and handling anomalies
    for both bowlers and batsmen.
    """
    # Remove historical stats for 'No Bowler'
    mask = corrected_df['bowler1_name'] == 'No Bowler'
    corrected_df.loc[mask, ['bowler1_historical_average', 'bowler1_historical_economy', 'bowler1_historical_strike_rate']] = None
    
    mask = corrected_df['bowler2_name'] == 'No Bowler'
    corrected_df.loc[mask, ['bowler2_historical_average', 'bowler2_historical_economy', 'bowler2_historical_strike_rate']] = None
    
    # Remove historical stats for 'No Partner'
    mask = corrected_df['batsman2_name'] == 'No Partner'
    corrected_df.loc[mask, ['batsman2_historical_average', 'batsman2_historical_strike_rate']] = None
    
    # Convert stats columns to numeric before comparison - Bowlers
    for col in ['bowler1_historical_average', 'bowler2_historical_average']:
        # Convert to numeric, coercing errors to NaN
        corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce')
        # Now safely apply comparison
        mask = (corrected_df[col] < 0) | (corrected_df[col] > 100)
        corrected_df.loc[mask, col] = None
    
    for col in ['bowler1_historical_economy', 'bowler2_historical_economy']:
        corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce')
        mask = (corrected_df[col] < 0) | (corrected_df[col] > 15)
        corrected_df.loc[mask, col] = None
    
    for col in ['bowler1_historical_strike_rate', 'bowler2_historical_strike_rate']:
        corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce')
        mask = (corrected_df[col] < 0) | (corrected_df[col] > 100)
        corrected_df.loc[mask, col] = None
    
    # Convert stats columns to numeric before comparison - Batsmen
    for col in ['batsman1_historical_average', 'batsman2_historical_average']:
        corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce')
        mask = (corrected_df[col] < 0) | (corrected_df[col] > 100)
        corrected_df.loc[mask, col] = None
    
    for col in ['batsman1_historical_strike_rate', 'batsman2_historical_strike_rate']:
        corrected_df[col] = pd.to_numeric(corrected_df[col], errors='coerce')
        mask = (corrected_df[col] < 0) | (corrected_df[col] > 250)  # Batsmen can have higher strike rates than bowlers
        corrected_df.loc[mask, col] = None
    
    return corrected_df

def main():
    # 1. Load the aggregated data
    aggregated_df = load_aggregated_data()
    if aggregated_df is None:
        return
    
    # 2. Base directories to search for match-specific stats
    base_dirs = [
        r'C:\Project\crawler_learning\ipl_prediction_model\ipl_scraper\indian-premier-league-2023-1345038',
        r'C:\Project\crawler_learning\ipl_prediction_model\ipl_scraper\indian-premier-league-2024-1410320'
    ]
    
    # 3. Correct historical stats for bowlers
    corrected_df = correct_historical_stats(aggregated_df, base_dirs)
    
    # 4. Correct historical stats for batsmen
    corrected_df = correct_batsman_historical_stats(corrected_df, base_dirs)
    
    # 5. Clean the corrected data
    corrected_df = clean_corrected_data(corrected_df)
    
    # 6. Save the corrected data to a new CSV file with error handling
    output_file = 'aggregated_match_data_batsman_corrected.csv'
    try:
        corrected_df.to_csv(output_file, index=False)
        print(f"Saved corrected data to '{output_file}'.")
    except PermissionError:
        # If permission error occurs, try with an alternative filename
        import time
        alt_output_file = f'aggregated_match_data_batsman_corrected_{int(time.time())}.csv'
        print(f"Permission denied when saving to '{output_file}'. Trying alternative filename: {alt_output_file}")
        try:
            corrected_df.to_csv(alt_output_file, index=False)
            print(f"Saved corrected data to alternative file '{alt_output_file}'.")
        except Exception as e:
            print(f"Error saving to alternative file: {e}")
            print("Could not save corrected data to any file. Please check permissions and try again.")
    except Exception as e:
        print(f"Error saving corrected data: {e}")
    
    # Print summary statistics of corrections made
    for col in [col for col in corrected_df.columns if '_corrected' in col]:
        count_corrected = corrected_df[col].sum()
        print(f"Corrected {count_corrected} values for {col.replace('_corrected', '')}")

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import fuzzywuzzy
    except ImportError:
        print("Installing required packages...")
        import pip
        pip.main(['install', 'fuzzywuzzy', 'python-Levenshtein'])
        from fuzzywuzzy import process
    
    main()
