import pandas as pd
import numpy as np  # Import numpy

def handle_missing_values(df):
    """
    Handles missing values in the aggregated match data.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    # 1. Imputation with Constant Values
    df['batsman2_name'] = df['batsman2_name'].fillna('No Partner')
    df['bowler2_name'] = df['bowler2_name'].fillna('No Bowler')
    cols_to_fill_zero = ['batsman2_runs', 'batsman2_balls_faced', 'batsman2_fours',
                         'batsman2_sixes', 'bowler2_overs_bowled', 'bowler2_maidens_bowled',
                         'bowler2_runs_conceded', 'bowler2_wickets_taken']
    df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)

    # 2. Imputation with Median
    historical_cols = ['batsman1_historical_average', 'batsman1_historical_strike_rate',
                       'batsman2_historical_average', 'batsman2_historical_strike_rate',
                       'bowler1_historical_average', 'bowler1_historical_economy',
                       'bowler1_historical_strike_rate', 'bowler2_historical_average',
                       'bowler2_historical_economy', 'bowler2_historical_strike_rate']

    for col in historical_cols:
        df[col] = df[col].fillna(df[col].median())

    # 3. Forward Fill for 'favored_team' and 'win_percentage' (with caution)
    # Consider if this is appropriate for your analysis
    df['favored_team'] = df['favored_team'].ffill()
    df['win_percentage'] = df['win_percentage'].ffill()

    # Fill any remaining missing values in 'favored_team' and 'win_percentage' with a default value
    df['favored_team'] = df['favored_team'].fillna('Unknown')
    df['win_percentage'] = df['win_percentage'].fillna(0.5)  # 0.5 could represent equal chances

    # 4. Handle missing venue data (example - replace with "Unknown")
    df['venue'] = df['venue'].fillna('Unknown')
    df['matches_played'] = df['matches_played'].fillna(0)
    df['average_runs_per_wicket'] = df['average_runs_per_wicket'].fillna(0)
    df['average_runs_per_over'] = df['average_runs_per_over'].fillna(0)

    return df

def fill_missing_venue_data(df):
    """
    Fills missing venue information and stats within each match_id group.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data with match_id and venue columns.
        
    Returns:
        pd.DataFrame: DataFrame with venue information propagated within each match_id group.
    """
    # Check if match_id column exists
    if 'match_id' not in df.columns:
        print("Warning: 'match_id' column not found in DataFrame. Skipping venue fill operation.")
        return df

    # Copy the DataFrame to avoid modifying the original
    df_filled = df.copy()
    
    # Columns to fill
    cols_to_fill = ['venue', 'matches_played', 'average_runs_per_wicket', 'average_runs_per_over']
    
    # Filter columns that actually exist in the DataFrame
    cols_to_fill = [col for col in cols_to_fill if col in df_filled.columns]
    
    if not cols_to_fill:
        print("Warning: None of the venue columns found in DataFrame.")
        return df
        
    # Group by 'match_id' and apply forward fill and backward fill
    for col in cols_to_fill:
        df_filled[col] = df_filled.groupby('match_id')[col].ffill()
        df_filled[col] = df_filled.groupby('match_id')[col].bfill()
    
    print(f"Successfully filled missing values in columns: {cols_to_fill}")
    return df_filled

# The following ensures these functions are available when importing this module
__all__ = ['handle_missing_values', 'fill_missing_venue_data']

# Example usage (assuming you have loaded your data into a DataFrame called 'aggregated_data')
# from data_cleaning import handle_missing_values, fill_missing_venue_data  # Import the functions
# cleaned_data = handle_missing_values(aggregated_data)
# cleaned_data = fill_missing_venue_data(cleaned_data)
# print(cleaned_data.isnull().sum())  # Check for remaining missing values
