# test_merger.py
from data_merger import merge_match_data
import os
import pandas as pd

def try_read_csv(file_path):
    """Try different encodings to read the CSV file"""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding")
            continue
    raise ValueError(f"Could not read file with any encoding: {file_path}")

def examine_player_stats(batting_stats_file, player_name):
    """Examine stats for a specific player"""
    print(f"\nExamining stats for {player_name}:")
    
    if os.path.exists(batting_stats_file):
        stats_df = try_read_csv(batting_stats_file)
        
        # Clean player names and create variations for matching
        stats_df['clean_name'] = stats_df['player_name'].apply(
            lambda x: x.split('(')[0].strip() if '(' in x else x.strip()
        )
        stats_df['lower_name'] = stats_df['clean_name'].str.lower()
        
        # Try different matching approaches
        print("\n1. Exact match:")
        exact_match = stats_df[stats_df['clean_name'] == player_name]
        if not exact_match.empty:
            print(exact_match)
        else:
            print("No exact match found")
        
        print("\n2. Case-insensitive match:")
        case_insensitive = stats_df[stats_df['lower_name'] == player_name.lower()]
        if not case_insensitive.empty:
            print(case_insensitive)
        else:
            print("No case-insensitive match found")
        
        print("\n3. Partial match:")
        partial_match = stats_df[stats_df['lower_name'].str.contains(
            player_name.lower(), na=False)]
        if not partial_match.empty:
            print(partial_match)
        else:
            print("No partial match found")
        
        # Print all player names for reference
        print("\nAll available player names:")
        for idx, row in stats_df.iterrows():
            print(f"Original: {row['player_name']:<30} "
                  f"Clean: {row['clean_name']:<30} "
                  f"Lower: {row['lower_name']}")
    else:
        print(f"File not found: {batting_stats_file}")

def test_merge():
    # Set paths
    match_dir = r"C:\Project\crawler_learning\crex_scrapper_python\ipl_scraper\ipl-2020-21-1210595\1216492"
    ball_by_ball_dir = os.path.join(match_dir, "ball_by_ball")
    
    try:
        # Check stats for specific players
        batting_stats_file = os.path.join(ball_by_ball_dir, 'MI_batting_stats.csv')
        examine_player_stats(batting_stats_file, "Quinton de Kock")
        examine_player_stats(batting_stats_file, "Rohit Sharma")
        
        # Run the merge function
        print("\nRunning merge function...")
        merged_data = merge_match_data(match_dir, ball_by_ball_dir)
        
        # Analyze first few overs
        print("\nAnalyzing first 3 overs:")
        first_overs = merged_data[
            (merged_data['batting_team'] == 'MI') & 
            (merged_data['over_number'].between(1, 3))
        ]
        
        for _, row in first_overs.iterrows():
            print(f"\nOver {row['over_number']}.{row['ball_number']}:")
            print(f"Batsman 1: {row['batsman1_name']}")
            print(f"  - Historical avg: {row['batsman1_historical_average']}")
            print(f"  - Historical SR: {row['batsman1_historical_strike_rate']}")
            print(f"Batsman 2: {row['batsman2_name']}")
            print(f"  - Historical avg: {row['batsman2_historical_average']}")
            print(f"  - Historical SR: {row['batsman2_historical_strike_rate']}")
        
    except Exception as e:
        print(f"Error during merge: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_merge()