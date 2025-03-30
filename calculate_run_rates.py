import pandas as pd
import numpy as np
import os
import time

def load_data(file_path='aggregated_match_data_batsman_corrected.csv'):
    """
    Load the match data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded match data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_run_rates(df):
    """
    Calculate run rates for each ball in the match data.
    
    Args:
        df (pd.DataFrame): The match data
        
    Returns:
        pd.DataFrame: The data with added run rate columns
    """
    # Create a copy of the original dataframe
    df_with_rates = df.copy()
    
    print("Calculating run rates...")
    
    # Initialize columns
    df_with_rates['current_run_rate'] = np.nan
    df_with_rates['required_run_rate'] = np.nan
    df_with_rates['projected_score'] = np.nan
    df_with_rates['runs_in_over'] = df['runs_scored']  # Map runs_scored to runs_in_over
    df_with_rates['is_wicket'] = df['wickets'] > 0  # Convert wickets to boolean
    df_with_rates['is_four'] = (df['boundaries'] == 1) & (df['runs_scored'] == 4)  # Identify fours
    df_with_rates['is_six'] = (df['boundaries'] == 1) & (df['runs_scored'] == 6)   # Identify sixes
    df_with_rates['total_score'] = np.nan  # Initialize column for cumulative score
    df_with_rates['total_wickets'] = np.nan  # Initialize column for cumulative wickets
    
    # Add game phase indicators
    df_with_rates['powerplay'] = (df_with_rates['over_number'] < 6).astype(int)
    df_with_rates['middle_overs'] = ((df_with_rates['over_number'] >= 6) & (df_with_rates['over_number'] < 16)).astype(int)
    df_with_rates['death_overs'] = (df_with_rates['over_number'] >= 16).astype(int)
    
    # Process each match and innings separately
    for (match_id, innings), group in df_with_rates.groupby(['match_id', 'innings_num']):
        # Sort by over and ball to ensure chronological order
        group_sorted = group.sort_values(by=['over_number', 'ball_number'])
        
        # Calculate cumulative runs for each ball (excluding extras)
        cumulative_runs = group_sorted['runs_scored'].cumsum()
        cumulative_wickets = group_sorted['wickets'].cumsum()
        
        # Parse ball_number correctly
        ball_number_str = group_sorted['ball_number'].astype(str).str.split('.', expand=True)
        over_part = pd.to_numeric(ball_number_str[0], errors='coerce')
        ball_part = pd.to_numeric(ball_number_str[1], errors='coerce')
        
        # Validate and clean ball/over numbers
        valid_overs = (over_part >= 0) & (over_part < 20)
        valid_balls = (ball_part >= 0) & (ball_part <= 6)
        valid_mask = valid_overs & valid_balls
        
        # Calculate current overs with validation
        current_overs = np.where(valid_mask,
                               over_part + (ball_part / 6),
                               np.nan)
        
        # Calculate current run rate with bounds checking
        current_run_rate = np.where(
            (current_overs > 0) & (current_overs <= 20),
            cumulative_runs / current_overs,
            np.nan
        )
        
        # Clip run rate to reasonable bounds (0 to 36 runs per over)
        current_run_rate = np.clip(current_run_rate, 0, 36)
        
        # Update the main dataframe with calculated values
        idx = group_sorted.index
        df_with_rates.loc[idx, 'total_score'] = cumulative_runs
        df_with_rates.loc[idx, 'current_run_rate'] = current_run_rate
        df_with_rates.loc[idx, 'total_wickets'] = cumulative_wickets
        
        # Calculate projected score (current run rate * total overs)
        max_overs = 20  # T20 match
        df_with_rates.loc[idx, 'projected_score'] = np.clip(
            current_run_rate * max_overs,
            0,
            720  # Max possible score (36 runs per over * 20 overs)
        )
        
        # For 2nd innings, calculate required run rate
        if innings == 2:
            try:
                # Find the target by looking at team 1's final score
                innings1_data = df[(df['match_id'] == match_id) & (df['innings_num'] == 1)]
                if not innings1_data.empty:
                    # Use only runs_scored for target
                    target = innings1_data['runs_scored'].sum() + 1
                    
                    # Calculate runs needed
                    runs_needed = target - cumulative_runs
                    
                    # Calculate remaining overs
                    remaining_overs = max_overs - current_overs
                    
                    # Calculate required run rate with validation
                    required_run_rate = np.where(
                        (remaining_overs > 0) & (remaining_overs <= 20),
                        runs_needed / remaining_overs,
                        np.nan
                    )
                    
                    # Clip required run rate to reasonable bounds
                    required_run_rate = np.clip(required_run_rate, 0, 36)
                    df_with_rates.loc[idx, 'required_run_rate'] = required_run_rate
                    
                print(f"Calculated required run rate for match {match_id}, innings {innings}")
            except Exception as e:
                print(f"Error calculating required run rate for match {match_id}, innings {innings}: {e}")
    
    # Final validation and cleanup
    df_with_rates['current_run_rate'] = pd.to_numeric(df_with_rates['current_run_rate'], errors='coerce')
    df_with_rates['required_run_rate'] = pd.to_numeric(df_with_rates['required_run_rate'], errors='coerce')
    df_with_rates['projected_score'] = pd.to_numeric(df_with_rates['projected_score'], errors='coerce')
    
    # Remove any remaining invalid values
    df_with_rates.loc[df_with_rates['current_run_rate'] < 0, 'current_run_rate'] = np.nan
    df_with_rates.loc[df_with_rates['required_run_rate'] < 0, 'required_run_rate'] = np.nan
    df_with_rates.loc[df_with_rates['projected_score'] < 0, 'projected_score'] = np.nan
    
    # Round run rates to 2 decimal places for readability
    df_with_rates['current_run_rate'] = df_with_rates['current_run_rate'].round(2)
    df_with_rates['required_run_rate'] = df_with_rates['required_run_rate'].round(2)
    df_with_rates['projected_score'] = df_with_rates['projected_score'].round(0)
    
    return df_with_rates

def calculate_pressure_index(df):
    """
    Calculate a pressure index based on required run rate and wickets remaining.
    """
    df_with_pressure = df.copy()
    
    # Initialize pressure index column
    df_with_pressure['pressure_index'] = 0
    
    # Only calculate for 2nd innings
    second_innings_mask = df_with_pressure['innings_num'] == 2
    second_innings_data = df_with_pressure[second_innings_mask]
    
    if second_innings_data.empty:
        return df_with_pressure
    
    # Base pressure on required run rate
    df_with_pressure.loc[second_innings_mask, 'pressure_index'] = \
        df_with_pressure.loc[second_innings_mask, 'required_run_rate'].fillna(0) / 2
    
    # Process each match separately
    for match_id, group in second_innings_data.groupby(['match_id']):
        # Parse ball_number to extract over and ball for sorting
        ball_numbers = group['ball_number'].astype(str)
        
        # Extract over and ball part
        split_values = ball_numbers.str.split('.', expand=True)
        
        if len(split_values.columns) >= 2:
            over_part = split_values[0].astype(float)
            ball_part = split_values[1].astype(float)
        else:
            over_part = ball_numbers.astype(float).apply(np.floor)
            ball_part = (ball_numbers.astype(float) - over_part) * 10
        
        # Calculate the exact over position
        current_overs = over_part + (ball_part / 6)
        
        # Sort by calculated overs
        group = group.copy()
        group['calculated_overs'] = current_overs
        group_sorted = group.sort_values(by='calculated_overs')
        
        # Count wickets and calculate pressure factor
        cumulative_wickets = group_sorted['wickets'].cumsum()
        wicket_pressure = cumulative_wickets * 0.5
        
        # Update pressure index
        df_with_pressure.loc[group_sorted.index, 'pressure_index'] += wicket_pressure
    
    # Normalize pressure index to 0-10 scale
    max_pressure = df_with_pressure.loc[second_innings_mask, 'pressure_index'].max()
    if max_pressure > 0:
        df_with_pressure.loc[second_innings_mask, 'pressure_index'] = \
            (df_with_pressure.loc[second_innings_mask, 'pressure_index'] / max_pressure) * 10
    
    # Round pressure index
    df_with_pressure['pressure_index'] = df_with_pressure['pressure_index'].round(1)
    
    return df_with_pressure

def calculate_over_summaries(df):
    """
    Calculate over-by-over summaries including runs, wickets, and boundaries.
    
    Args:
        df (pd.DataFrame): Match data
        
    Returns:
        pd.DataFrame: Over summary statistics
    """
    # Create over summary by grouping by match_id, innings, and over
    over_summary = df.groupby(['match_id', 'innings_num', 'over_number']).agg({
        'runs_scored': 'sum',
        'wickets': 'sum',
        'boundaries': 'sum',
        'dot_balls': 'sum',
        'current_run_rate': 'last',  # Run rate at the end of the over
        'required_run_rate': 'last'  # Required run rate at the end of the over
    }).reset_index()
    
    # Rename columns for clarity
    over_summary = over_summary.rename(columns={
        'runs_scored': 'runs_in_over',
        'wickets': 'wickets_in_over',
        'boundaries': 'boundaries_in_over'
    })
    
    return over_summary

def main():
    start_time = time.time()
    
    # Load the data
    data_file = 'aggregated_match_data_batsman_corrected.csv'
    df = load_data(data_file)
    if df is None:
        return
    
    # Calculate run rates
    run_rate_time = time.time()
    df_with_rates = calculate_run_rates(df)
    print(f"Run rates calculated in {time.time() - run_rate_time:.2f} seconds")
    
    # Calculate pressure index
    pressure_time = time.time()
    df_with_pressure = calculate_pressure_index(df_with_rates)
    print(f"Pressure index calculated in {time.time() - pressure_time:.2f} seconds")
    
    # Calculate over summaries
    summary_time = time.time()
    over_summary = calculate_over_summaries(df_with_pressure)
    print(f"Over summaries calculated in {time.time() - summary_time:.2f} seconds")
    
    # Save the enhanced data
    try:
        output_file = 'match_data_with_run_rates.csv'
        df_with_pressure.to_csv(output_file, index=False)
        print(f"Saved data with run rates to {output_file}")
        
        summary_file = 'over_summary_stats.csv'
        over_summary.to_csv(summary_file, index=False)
        print(f"Saved over summary stats to {summary_file}")
    except Exception as e:
        print(f"Error saving files: {e}")
        
        # Try alternative location
        try:
            alt_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(alt_dir, exist_ok=True)
            
            alt_output = os.path.join(alt_dir, 'match_data_with_run_rates.csv')
            df_with_pressure.to_csv(alt_output, index=False)
            
            alt_summary = os.path.join(alt_dir, 'over_summary_stats.csv')
            over_summary.to_csv(alt_summary, index=False)
            
            print(f"Saved files to alternative location: {alt_dir}")
        except Exception as e2:
            print(f"Failed to save to alternative location: {e2}")
    
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    try:
        print(f"Total matches processed: {df_with_pressure['match_id'].nunique()}")
        print(f"Average run rate: {df_with_pressure['current_run_rate'].mean():.2f}")
        if 'innings_num' in df_with_pressure.columns:
            second_innings_data = df_with_pressure[df_with_pressure['innings_num'] == 2]
            if not second_innings_data.empty:
                print(f"Average required run rate (2nd innings): {second_innings_data['required_run_rate'].mean():.2f}")
                print(f"Average pressure index (2nd innings): {second_innings_data['pressure_index'].mean():.2f}")
    except Exception as e:
        print(f"Error displaying summary statistics: {e}")

if __name__ == "__main__":
    main()
