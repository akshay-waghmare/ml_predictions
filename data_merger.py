import pandas as pd
import os
import re

def merge_match_data(match_dir, ball_by_ball_dir):
    """
    Merge the ball-by-ball data with the corrected data to create a unified dataset.
    """
    # Extract match_id from directory path - improved to handle specific folder structure
    match_id = os.path.basename(match_dir)
    if match_id == "ball_by_ball":  # Handle the case where ball_by_ball is the leaf directory
        match_id = os.path.basename(os.path.dirname(match_dir))
    
    # Extract numeric match ID if we have a path like "ipl-2020-21-1210595\1216492"
    if os.path.sep in match_id or "\\" in match_id:
        match_id = match_id.split(os.path.sep)[-1]
        match_id = match_id.split("\\")[-1]  # Handle Windows paths
    
    # Extract just the numeric part if the match_id contains other text
    numeric_match = re.search(r'(\d+)$', match_id)
    if numeric_match:
        match_id = numeric_match.group(1)
    
    print(f"Processing match ID: {match_id}")
    
    # Read the corrected data
    corrected_data = pd.read_csv(os.path.join(ball_by_ball_dir, 'corrected_data.csv'))
    
    # Read both innings data - note the swapped files to correct the mixup
    second_innings = pd.read_csv(os.path.join(ball_by_ball_dir, 'first_innings_ball_by_ball.csv'))
    first_innings = pd.read_csv(os.path.join(ball_by_ball_dir, 'second_innings_ball_by_ball.csv'))
    
    # Add innings number to the dataframes
    first_innings['innings_num'] = 1
    second_innings['innings_num'] = 2
    
    # Ensure over_number is numeric for proper sorting
    first_innings['over_number'] = pd.to_numeric(first_innings['over_number'], errors='coerce')
    second_innings['over_number'] = pd.to_numeric(second_innings['over_number'], errors='coerce')
    
    # Fill missing values for sorting consistency
    first_innings['over_number'] = first_innings['over_number'].fillna(0)
    second_innings['over_number'] = second_innings['over_number'].fillna(0)
    
    # Combine both innings data
    ball_by_ball_data = pd.concat([first_innings, second_innings])
    
    # Rename columns in corrected_data to match with ball_by_ball_data
    corrected_data = corrected_data.rename(columns={
        'innings': 'batting_team',
        'over': 'over_number',
        'fav_team': 'favored_team',
        'win_probability': 'win_percentage'
    })
    
    # Merge based on batting team and over number
    merged_data = pd.merge(
        corrected_data,
        ball_by_ball_data,
        on=['batting_team', 'over_number'],
        how='outer'
    )
    
    # Add match_id to the merged data
    merged_data['match_id'] = match_id
    
    # Handle duplicate columns and rename for clarity
    if 'total_runs_x' in merged_data.columns and 'total_runs_y' in merged_data.columns:
        merged_data['total_runs'] = merged_data['total_runs_y'].fillna(merged_data['total_runs_x'])
        merged_data = merged_data.drop(['total_runs_x', 'total_runs_y'], axis=1)
    
    # Read the match information file if it exists
    # We'll read it and include specific fields in the unified file
    match_info = {}
    match_info_file = os.path.join(ball_by_ball_dir, 'match_info.csv')
    if os.path.exists(match_info_file):
        try:
            match_info_df = pd.read_csv(match_info_file)
            if not match_info_df.empty:
                match_info = match_info_df.iloc[0].to_dict()
                print(f"Loaded match information: {match_info}")
        except Exception as e:
            print(f"Error loading match information: {e}")
    
    # Clean up and organize columns - include specific match info columns
    desired_columns = [
        'match_id', 'innings_num', 'batting_team', 'over_number', 'ball_number', 'runs_scored', 
        'boundaries', 'dot_balls', 'wickets', 'extras',
        'favored_team', 'win_percentage', 'striker_batsman',
        # Add these specific match info columns
        'winner', 'toss_winner', 'toss_decision',
        'batsman1_name', 'batsman1_runs', 'batsman1_balls_faced', 'batsman1_fours', 'batsman1_sixes',
        'batsman2_name', 'batsman2_runs', 'batsman2_balls_faced', 'batsman2_fours', 'batsman2_sixes',
        'bowler1_name', 'bowler1_overs_bowled', 'bowler1_maidens_bowled', 'bowler1_runs_conceded', 'bowler1_wickets_taken',
        'bowler2_name', 'bowler2_overs_bowled', 'bowler2_maidens_bowled', 'bowler2_runs_conceded', 'bowler2_wickets_taken',
        'venue', 'matches_played', 'average_runs_per_wicket', 'average_runs_per_over'
    ]
    
    # Add historical player stats
    merged_data = enrich_with_player_stats(merged_data, ball_by_ball_dir)
    
    # Add new columns to desired_columns
    historical_columns = [
        'batsman1_historical_average', 'batsman1_historical_strike_rate',
        'batsman2_historical_average', 'batsman2_historical_strike_rate',
        'bowler1_historical_average', 'bowler1_historical_economy', 'bowler1_historical_strike_rate',
        'bowler2_historical_average', 'bowler2_historical_economy', 'bowler2_historical_strike_rate'
    ]
    desired_columns.extend(historical_columns)
    
    # Keep only columns that exist in the merged data
    existing_columns = [col for col in desired_columns if col in merged_data.columns]
    merged_data = merged_data[existing_columns]
    
    # Add specific match information columns to merged_data
    match_info_columns = ['winner', 'toss_winner', 'toss_decision']
    for key in match_info_columns:
        if key in match_info:
            merged_data[key] = match_info[key]
    
    # Ensure all required columns exist, fill with defaults if missing
    for col in desired_columns:
        if col not in merged_data.columns:
            if col == 'favored_team':
                merged_data[col] = ''
            elif col == 'win_percentage':
                merged_data[col] = 0.0
            elif col == 'innings_num':
                merged_data[col] = 0
            elif col == 'match_id':
                merged_data[col] = match_id
            # Add this condition for match info columns we want to include
            elif col in match_info_columns and col in match_info:
                merged_data[col] = match_info[col]
            elif 'name' in col:
                merged_data[col] = ''
            elif any(x in col for x in ['runs', 'balls', 'fours', 'sixes', 'wickets', 'maidens']):
                merged_data[col] = 0
            else:
                merged_data[col] = ''
    
    # Sort columns according to desired order
    merged_data = merged_data[desired_columns]
    
    # Sort data by innings number and then by ball number for chronological order
    # Convert over_number and ball_number to numeric values first to ensure proper sorting
    merged_data['over_number'] = pd.to_numeric(merged_data['over_number'], errors='coerce')
    if 'ball_number' in merged_data.columns:
        merged_data['ball_number'] = pd.to_numeric(merged_data['ball_number'], errors='coerce')
    
    # Fill NA values to prevent sorting issues
    merged_data['over_number'] = merged_data['over_number'].fillna(0)
    if 'ball_number' in merged_data.columns:
        merged_data['ball_number'] = merged_data['ball_number'].fillna(0)
    
    # First, identify the correct innings number for each team
    # This is based on the most common innings number for each team
    team_innings_map = {}
    for team in merged_data['batting_team'].unique():
        team_data = merged_data[merged_data['batting_team'] == team]
        if not team_data.empty and 'innings_num' in team_data.columns:
            # Get the most common non-NaN innings number for this team
            innings_nums = team_data['innings_num'].dropna().astype(int)
            if not innings_nums.empty:
                most_common = innings_nums.mode()[0]
                team_innings_map[team] = most_common
    
    # Fill in missing innings numbers for all rows, especially 20th over rows
    for idx, row in merged_data.iterrows():
        team = row['batting_team']
        if pd.isna(row['innings_num']) and team in team_innings_map:
            merged_data.at[idx, 'innings_num'] = team_innings_map[team]
    
    # Handle special case for 20th over - ensure it's included with correct innings number
    for team in merged_data['batting_team'].unique():
        if team not in team_innings_map:
            continue
            
        innings_num = team_innings_map[team]
        team_innings_data = merged_data[(merged_data['batting_team'] == team) & 
                                      (merged_data['innings_num'] == innings_num)]
        
        if team_innings_data.empty:
            continue
            
        # Check if 20th over exists
        has_over_20 = any((merged_data['batting_team'] == team) & (merged_data['over_number'] == 20))
        last_over = team_innings_data['over_number'].max()
        
        if not has_over_20 and last_over < 20:
            # Create a placeholder row for the 20th over
            last_row = team_innings_data.iloc[-1].copy()
            last_row['over_number'] = 20
            last_row['ball_number'] = 20.0  # Set to a value for the 20th over
            last_row['innings_num'] = innings_num  # Explicitly set innings number
            
            merged_data = pd.concat([merged_data, pd.DataFrame([last_row])], ignore_index=True)
    
    # Sort again to make sure new rows are in right position
    merged_data = merged_data.sort_values(by=['innings_num', 'over_number', 'ball_number'])
    
    # Save the merged data
    output_file = os.path.join(ball_by_ball_dir, 'unified_match_data_enriched.csv')
    merged_data.to_csv(output_file, index=False)
    print(f"Merged data saved to: {output_file}")
    
    return merged_data

def clean_player_name(name):
    """Clean player name by removing special characters, captaincy indicators, and normalizing spaces"""
    if pd.isna(name):
        return name
    
    # Convert to string if not already
    name = str(name)
    
    # Remove all non-ASCII characters (handles Â, †, and other Unicode characters)
    name = ''.join(c for c in name if ord(c) < 128)
    
    # Remove (c), (wk), and other indicators
    name = re.sub(r'\(c\)|\(wk\)|\(.*?\)', '', name)
    
    # Remove trailing commas
    name = name.strip(',')
    
    # Keep only alphanumeric chars and spaces, handle special characters
    name = ''.join(c for c in name if c.isalnum() or c.isspace() or c in "-'.")
    
    # Normalize spaces and strip
    return ' '.join(name.split())

def load_team_stats(match_dir, team, encoding_list=['utf-8', 'latin1', 'cp1252']):
    """Load team statistics with multiple encoding attempts"""
    stats = {}
    
    for file_type in ['batting', 'bowling']:
        filename = os.path.join(match_dir, f'{team}_{file_type}_stats.csv')
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found")
            continue
            
        for encoding in encoding_list:
            try:
                df = pd.read_csv(filename, encoding=encoding)
                df['clean_name'] = df['player_name'].apply(clean_player_name)
                df['lower_name'] = df['clean_name'].str.lower()
                stats[file_type] = df
                print(f"Successfully loaded {team} {file_type} stats with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                break
                
    return stats

def find_player_match(name, stats_df):
    """Find matching player in stats dataframe using multiple matching strategies"""
    if stats_df is None or stats_df.empty:
        return None
        
    clean_name = clean_player_name(name)
    lower_name = clean_name.lower()
    
    # Try exact match
    match = stats_df[stats_df['lower_name'] == lower_name]
    if not match.empty:
        return match.iloc[0]
    
    # Try partial match
    match = stats_df[stats_df['lower_name'].str.contains(lower_name, na=False)]
    if not match.empty:
        return match.iloc[0]
    
    # Try fuzzy match (remove spaces and special chars)
    simplified_name = re.sub(r'[^a-z]', '', lower_name)
    stats_df['simplified_name'] = stats_df['lower_name'].apply(lambda x: re.sub(r'[^a-z]', '', x))
    match = stats_df[stats_df['simplified_name'].str.contains(simplified_name, na=False)]
    if not match.empty:
        return match.iloc[0]
    
    return None

def enrich_with_player_stats(merged_data, match_dir):
    """Add historical player statistics to the unified match data"""
    teams = merged_data['batting_team'].unique()
    print(f"\nTeams found in merged data: {teams}")
    
    # Initialize stats for all teams
    team_stats = {team: load_team_stats(match_dir, team) for team in teams}
    
    # Initialize historical columns if they don't exist
    historical_columns = {
        'batsman': ['historical_average', 'historical_strike_rate'],
        'bowler': ['historical_average', 'historical_economy', 'historical_strike_rate']
    }
    
    for role in ['batsman1', 'batsman2', 'bowler1', 'bowler2']:
        base = role.replace('1', '').replace('2', '')
        for stat in historical_columns[base]:
            col = f"{role}_{stat}"
            if col not in merged_data.columns:
                merged_data[col] = 0.0
    
    # Process each row
    for idx, row in merged_data.iterrows():
        batting_team = row['batting_team']
        bowling_team = next(team for team in teams if team != batting_team)
        
        # Process batsmen
        for i in [1, 2]:
            batsman_name = row[f'batsman{i}_name']
            if pd.notna(batsman_name) and 'batting' in team_stats[batting_team]:
                stats_df = team_stats[batting_team]['batting']
                match = find_player_match(batsman_name, stats_df)
                if match is not None:
                    # Update batsman name with correct full name
                    merged_data.at[idx, f'batsman{i}_name'] = match['player_name']
                    merged_data.at[idx, f'batsman{i}_historical_average'] = match['average']
                    merged_data.at[idx, f'batsman{i}_historical_strike_rate'] = match['strike_rate']
                    print(f"Found match for {batsman_name} -> {match['player_name']}")
                else:
                    print(f"No match found for {batsman_name}")
        
        # Process bowlers
        for i in [1, 2]:
            bowler_name = row[f'bowler{i}_name']
            if pd.notna(bowler_name) and 'bowling' in team_stats[bowling_team]:
                stats_df = team_stats[bowling_team]['bowling']
                match = find_player_match(bowler_name, stats_df)
                if match is not None:
                    # Update bowler name with correct full name
                    merged_data.at[idx, f'bowler{i}_name'] = match['player_name']
                    merged_data.at[idx, f'bowler{i}_historical_average'] = match['average']
                    merged_data.at[idx, f'bowler{i}_historical_economy'] = match['economy']
                    merged_data.at[idx, f'bowler{i}_historical_strike_rate'] = match['strike_rate']
                    print(f"Found match for {bowler_name} -> {match['player_name']}")
                else:
                    print(f"No match found for {bowler_name}")
    
    return merged_data

def generate_team_summary(merged_data):
    """
    Generate a summary of team performance from the merged data.
    """
    # Define available metrics to aggregate
    metrics = {}
    
    # Check which columns are available and add them to metrics dict
    if 'total_runs' in merged_data.columns:
        metrics['total_runs'] = 'sum'
    if 'boundaries' in merged_data.columns:
        metrics['boundaries'] = 'sum'
    if 'dot_balls' in merged_data.columns:
        metrics['dot_balls'] = 'sum'
    if 'wickets' in merged_data.columns:
        metrics['wickets'] = 'sum'
    if 'extras' in merged_data.columns:
        metrics['extras'] = 'sum'
    
    if not metrics:
        print("Warning: No metrics columns found in the data")
        return pd.DataFrame()
    
    team_summary = merged_data.groupby('batting_team').agg(metrics).reset_index()
    
    # Calculate additional metrics only if required columns exist
    if 'boundaries' in team_summary.columns:
        team_summary['boundary_percentage'] = (team_summary['boundaries'] * 100 / 20).round(2)
    if 'dot_balls' in team_summary.columns:
        team_summary['dot_ball_percentage'] = (team_summary['dot_balls'] * 100 / 120).round(2)
    
    return team_summary

if __name__ == "__main__":
    # Example usage
    match_dir = "/path/to/match/directory"
    ball_by_ball_dir = "/path/to/ball_by_ball/directory"
    
    merged_data = merge_match_data(match_dir, ball_by_ball_dir)
    team_summary = generate_team_summary(merged_data)
    print("\nTeam Summary:")
    print(team_summary)
