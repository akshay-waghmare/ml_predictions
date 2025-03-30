import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from data_cleaning import handle_missing_values, fill_missing_venue_data

def load_match_data(file_path='match_data_with_run_rates.csv'):
    """
    Load the enhanced match data with run rates.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded match data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded match data with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # Try to load the original data as fallback
        try:
            df = pd.read_csv('aggregated_match_data_batsman_corrected.csv')
            print(f"Loaded fallback data with {len(df)} rows.")
            return df
        except FileNotFoundError:
            print("No match data found. Please run calculate_run_rates.py first.")
            return None
        except Exception as e:
            print(f"Error loading fallback data: {e}")
            return None
    except Exception as e:
        print(f"Error loading match data: {e}")
        return None

def create_match_summary(df):
    """
    Create a summary of match statistics.
    
    Args:
        df (pd.DataFrame): Match data
        
    Returns:
        pd.DataFrame: Match summary statistics
    """
    # Group by match_id
    match_summary = df.groupby('match_id').agg({
        'batting_team': lambda x: x.iloc[0],
        'winner': lambda x: x.iloc[0],
        'toss_winner': lambda x: x.iloc[0],
        'toss_decision': lambda x: x.iloc[0],
        'venue': lambda x: x.iloc[0],
        'runs_scored': 'sum',
        'wickets': 'sum',
        'boundaries': 'sum',
        'dot_balls': 'sum',
        'extras': 'sum'
    }).reset_index()
    
    # Calculate total overs
    match_overs = df.groupby('match_id')['over_number'].max().reset_index()
    match_summary = match_summary.merge(match_overs, on='match_id')
    
    # Rename columns for clarity
    match_summary = match_summary.rename(columns={
        'over_number': 'total_overs',
        'runs_scored': 'total_runs',
        'wickets': 'total_wickets',
        'boundaries': 'total_boundaries',
        'dot_balls': 'total_dot_balls',
        'extras': 'total_extras'
    })
    
    # Calculate run rate
    match_summary['run_rate'] = match_summary['total_runs'] / match_summary['total_overs']
    match_summary['run_rate'] = match_summary['run_rate'].round(2)
    
    return match_summary

def analyze_player_performance(df):
    """
    Analyze and summarize player performance.
    
    Args:
        df (pd.DataFrame): Match data
        
    Returns:
        tuple: DataFrames for batsman and bowler performance
    """
    # Extract unique batsmen and their performances
    batsmen_cols = ['batsman1_name', 'batsman1_runs', 'batsman1_balls_faced', 
                   'batsman1_fours', 'batsman1_sixes', 
                   'batsman1_historical_average', 'batsman1_historical_strike_rate']
    
    # Create a dataframe for batsman1
    batsman1_df = df[batsmen_cols].copy()
    batsman1_df.columns = ['name', 'runs', 'balls_faced', 'fours', 'sixes', 
                          'historical_average', 'historical_strike_rate']
    
    # Create a dataframe for batsman2 with the same columns
    batsman2_cols = ['batsman2_name', 'batsman2_runs', 'batsman2_balls_faced', 
                    'batsman2_fours', 'batsman2_sixes', 
                    'batsman2_historical_average', 'batsman2_historical_strike_rate']
    batsman2_df = df[batsman2_cols].copy()
    batsman2_df.columns = batsman1_df.columns
    
    # Combine the two dataframes
    batsmen_df = pd.concat([batsman1_df, batsman2_df])
    
    # Remove rows with NaN or 'No Partner' as name
    batsmen_df = batsmen_df[batsmen_df['name'].notna() & (batsmen_df['name'] != 'No Partner')]
    
    # Group by name and aggregate stats
    batsman_stats = batsmen_df.groupby('name').agg({
        'runs': 'sum',
        'balls_faced': 'sum',
        'fours': 'sum',
        'sixes': 'sum',
        'historical_average': 'mean',
        'historical_strike_rate': 'mean'
    }).reset_index()
    
    # Calculate strike rate
    batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls_faced']) * 100
    batsman_stats['strike_rate'] = batsman_stats['strike_rate'].round(2)
    
    # Clean up NaN and inf values in strike rate
    batsman_stats['strike_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)
    batsman_stats['strike_rate'].fillna(0, inplace=True)
    
    # Similar process for bowlers
    bowler_cols = ['bowler1_name', 'bowler1_overs_bowled', 'bowler1_maidens_bowled', 
                   'bowler1_runs_conceded', 'bowler1_wickets_taken', 
                   'bowler1_historical_average', 'bowler1_historical_economy', 
                   'bowler1_historical_strike_rate']
    
    bowler1_df = df[bowler_cols].copy()
    bowler1_df.columns = ['name', 'overs', 'maidens', 'runs_conceded', 'wickets', 
                         'historical_average', 'historical_economy', 'historical_strike_rate']
    
    bowler2_cols = ['bowler2_name', 'bowler2_overs_bowled', 'bowler2_maidens_bowled', 
                   'bowler2_runs_conceded', 'bowler2_wickets_taken', 
                   'bowler2_historical_average', 'bowler2_historical_economy', 
                   'bowler2_historical_strike_rate']
    
    bowler2_df = df[bowler2_cols].copy()
    bowler2_df.columns = bowler1_df.columns
    
    bowlers_df = pd.concat([bowler1_df, bowler2_df])
    bowlers_df = bowlers_df[bowlers_df['name'].notna() & (bowlers_df['name'] != 'No Bowler')]
    
    bowler_stats = bowlers_df.groupby('name').agg({
        'overs': 'sum',
        'maidens': 'sum',
        'runs_conceded': 'sum',
        'wickets': 'sum',
        'historical_average': 'mean',
        'historical_economy': 'mean',
        'historical_strike_rate': 'mean'
    }).reset_index()
    
    # Calculate economy rate
    bowler_stats['economy_rate'] = bowler_stats['runs_conceded'] / bowler_stats['overs']
    bowler_stats['economy_rate'] = bowler_stats['economy_rate'].round(2)
    
    # Clean up NaN and inf values
    bowler_stats['economy_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)
    bowler_stats['economy_rate'].fillna(0, inplace=True)
    
    return batsman_stats, bowler_stats

def analyze_venue_performance(df):
    """
    Analyze performance at different venues.
    
    Args:
        df (pd.DataFrame): Match data
        
    Returns:
        pd.DataFrame: Venue performance statistics
    """
    # Group by venue to get statistics
    venue_stats = df.groupby('venue').agg({
        'match_id': 'nunique',
        'runs_scored': 'sum',
        'wickets': 'sum',
        'boundaries': 'sum',
        'dot_balls': 'sum',
        'extras': 'sum',
        'average_runs_per_wicket': 'mean',
        'average_runs_per_over': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    venue_stats = venue_stats.rename(columns={
        'match_id': 'matches_played',
        'runs_scored': 'total_runs',
        'wickets': 'total_wickets',
        'boundaries': 'total_boundaries',
        'dot_balls': 'total_dot_balls',
        'extras': 'total_extras'
    })
    
    # Calculate average runs per match
    venue_stats['avg_runs_per_match'] = venue_stats['total_runs'] / venue_stats['matches_played']
    venue_stats['avg_runs_per_match'] = venue_stats['avg_runs_per_match'].round(2)
    
    # Calculate average wickets per match
    venue_stats['avg_wickets_per_match'] = venue_stats['total_wickets'] / venue_stats['matches_played']
    venue_stats['avg_wickets_per_match'] = venue_stats['avg_wickets_per_match'].round(2)
    
    # Remove any rows with 'Unknown' venue
    venue_stats = venue_stats[venue_stats['venue'] != 'Unknown']
    
    return venue_stats

def calculate_win_probability_factors(df):
    """
    Calculate factors that may influence win probability.
    
    Args:
        df (pd.DataFrame): Match data
        
    Returns:
        pd.DataFrame: Factors affecting win probability
    """
    # Group by match_id to get match level data
    match_data = df.groupby('match_id').agg({
        'batting_team': lambda x: x.iloc[0],
        'winner': lambda x: x.iloc[0],
        'toss_winner': lambda x: x.iloc[0],
        'toss_decision': lambda x: x.iloc[0],
        'favored_team': lambda x: x.iloc[0],
        'win_percentage': lambda x: x.iloc[0],
        'runs_scored': 'sum',
        'wickets': 'sum',
        'boundaries': 'sum',
        'dot_balls': 'sum'
    }).reset_index()
    
    # Calculate whether toss winner won the match
    match_data['toss_winner_won'] = match_data['toss_winner'] == match_data['winner']
    
    # Calculate whether favored team won the match (if available)
    if 'favored_team' in match_data.columns:
        match_data['favored_team_won'] = match_data['favored_team'] == match_data['winner']
    
    # Calculate win probability factors
    factors = {
        'toss_win_match_win': match_data['toss_winner_won'].mean() * 100,
    }
    
    if 'favored_team' in match_data.columns:
        factors['favored_team_win_percentage'] = match_data['favored_team_won'].mean() * 100
    
    # Analyze toss decision impact
    toss_decision_impact = match_data.groupby('toss_decision')['toss_winner_won'].mean() * 100
    
    # Prepare results
    results_df = pd.DataFrame({
        'Factor': list(factors.keys()) + list(toss_decision_impact.index.map(lambda x: f'Toss decision: {x}')),
        'Win Percentage': list(factors.values()) + list(toss_decision_impact.values)
    })
    
    return results_df, match_data

def visualize_match_progression(df, match_id):
    """
    Visualize the progression of a specific match.
    
    Args:
        df (pd.DataFrame): Match data
        match_id: ID of the match to visualize
    
    Returns:
        tuple: Figure and axes objects
    """
    match_data = df[df['match_id'] == match_id].copy()
    
    if match_data.empty:
        print(f"No data found for match ID {match_id}")
        return None, None
    
    # Create innings data
    innings1 = match_data[match_data['innings_num'] == 1].sort_values(['over_number', 'ball_number'])
    innings2 = match_data[match_data['innings_num'] == 2].sort_values(['over_number', 'ball_number'])
    
    # Create over-by-over runs
    over_runs_1 = innings1.groupby('over_number')['runs_scored'].sum()
    over_runs_2 = innings2.groupby('over_number')['runs_scored'].sum()
    
    # Create figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot innings 1
    if not innings1.empty:
        team1 = innings1['batting_team'].iloc[0]
        cumulative_runs_1 = innings1.groupby('over_number')['runs_scored'].sum().cumsum()
        cumulative_wickets_1 = innings1.groupby('over_number')['wickets'].sum().cumsum()
        
        ax1.plot(cumulative_runs_1.index, cumulative_runs_1.values, 'b-', label=f"{team1} Runs")
        ax1.set_ylabel('Cumulative Runs')
        ax1.set_title(f"Innings 1: {team1}")
        
        # Add wickets as markers
        for over, wickets in cumulative_wickets_1.items():
            if wickets > 0 and over > 0:
                prev_over = over - 1
                if prev_over in cumulative_wickets_1:
                    wickets_in_over = wickets - cumulative_wickets_1[prev_over]
                    if wickets_in_over > 0:
                        runs = cumulative_runs_1[over]
                        ax1.scatter(over, runs, color='red', s=100, marker='x')
                        ax1.annotate(f"{wickets_in_over}W", (over, runs), 
                                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot innings 2
    if not innings2.empty:
        team2 = innings2['batting_team'].iloc[0]
        cumulative_runs_2 = innings2.groupby('over_number')['runs_scored'].sum().cumsum()
        cumulative_wickets_2 = innings2.groupby('over_number')['wickets'].sum().cumsum()
        
        ax2.plot(cumulative_runs_2.index, cumulative_runs_2.values, 'g-', label=f"{team2} Runs")
        ax2.set_ylabel('Cumulative Runs')
        ax2.set_title(f"Innings 2: {team2}")
        
        # Add target line if available
        if not innings1.empty:
            target = cumulative_runs_1.max() + 1
            ax2.axhline(y=target, color='r', linestyle='-', label=f"Target: {target}")
        
        # Add wickets as markers
        for over, wickets in cumulative_wickets_2.items():
            if wickets > 0 and over > 0:
                prev_over = over - 1
                if prev_over in cumulative_wickets_2:
                    wickets_in_over = wickets - cumulative_wickets_2[prev_over]
                    if wickets_in_over > 0:
                        runs = cumulative_runs_2[over]
                        ax2.scatter(over, runs, color='red', s=100, marker='x')
                        ax2.annotate(f"{wickets_in_over}W", (over, runs), 
                                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Set common x-axis label and limits
    ax1.set_xlim(0, 20)
    ax2.set_xlim(0, 20)
    ax2.set_xlabel('Over')
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    # Add match result
    winner = match_data['winner'].iloc[0] if not match_data.empty else "Unknown"
    fig.suptitle(f"Match {match_id} Progression (Winner: {winner})", fontsize=16)
    
    return fig, (ax1, ax2)

def calculate_player_impact_score(df):
    """
    Calculate an impact score for each player based on their performance.
    
    Args:
        df (pd.DataFrame): Match data
    
    Returns:
        tuple: DataFrames for batsmen and bowler impact scores
    """
    # Extract batsmen data
    batsman_data = pd.DataFrame()
    
    # Extract and combine batsman1 and batsman2 data
    for i in [1, 2]:
        batsman_cols = [
            f'batsman{i}_name', f'batsman{i}_runs', f'batsman{i}_balls_faced', 
            f'batsman{i}_fours', f'batsman{i}_sixes', 'match_id', 'batting_team'
        ]
        
        temp_df = df[batsman_cols].copy()
        temp_df.columns = ['name', 'runs', 'balls_faced', 'fours', 'sixes', 'match_id', 'team']
        batsman_data = pd.concat([batsman_data, temp_df])
    
    # Filter valid data
    batsman_data = batsman_data[batsman_data['name'].notna() & (batsman_data['name'] != 'No Partner')]
    
    # Group by player and match
    batsman_match_stats = batsman_data.groupby(['name', 'match_id', 'team']).agg({
        'runs': 'sum',
        'balls_faced': 'sum',
        'fours': 'sum',
        'sixes': 'sum'
    }).reset_index()
    
    # Calculate strike rate
    batsman_match_stats['strike_rate'] = (batsman_match_stats['runs'] / batsman_match_stats['balls_faced']) * 100
    
    # Calculate impact score components
    batsman_match_stats['run_impact'] = batsman_match_stats['runs'] / 30  # 30 runs = 1 point
    batsman_match_stats['sr_impact'] = (batsman_match_stats['strike_rate'] - 120) / 20  # 20 SR above 120 = 1 point
    batsman_match_stats['boundary_impact'] = (batsman_match_stats['fours'] * 0.2) + (batsman_match_stats['sixes'] * 0.5)
    
    # Calculate total impact score
    batsman_match_stats['impact_score'] = (
        batsman_match_stats['run_impact'] + 
        batsman_match_stats['sr_impact'] + 
        batsman_match_stats['boundary_impact']
    )
    
    # Handle NaN and negative values
    batsman_match_stats['impact_score'].fillna(0, inplace=True)
    batsman_match_stats['impact_score'] = batsman_match_stats['impact_score'].clip(0, None)
    
    # Aggregate player impact across all matches
    batsman_impact = batsman_match_stats.groupby('name').agg({
        'runs': 'sum',
        'balls_faced': 'sum',
        'fours': 'sum',
        'sixes': 'sum',
        'impact_score': 'mean',
        'team': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    }).reset_index()
    
    batsman_impact['overall_strike_rate'] = (batsman_impact['runs'] / batsman_impact['balls_faced']) * 100
    batsman_impact.sort_values('impact_score', ascending=False, inplace=True)
    
    # Similar approach for bowlers
    bowler_data = pd.DataFrame()
    
    for i in [1, 2]:
        bowler_cols = [
            f'bowler{i}_name', f'bowler{i}_overs_bowled', f'bowler{i}_maidens_bowled', 
            f'bowler{i}_runs_conceded', f'bowler{i}_wickets_taken', 'match_id', 'batting_team'
        ]
        
        temp_df = df[bowler_cols].copy()
        temp_df.columns = ['name', 'overs', 'maidens', 'runs_conceded', 'wickets', 'match_id', 'opposition']
        bowler_data = pd.concat([bowler_data, temp_df])
    
    # Filter valid data
    bowler_data = bowler_data[bowler_data['name'].notna() & (bowler_data['name'] != 'No Bowler')]
    
    # Group by player and match
    bowler_match_stats = bowler_data.groupby(['name', 'match_id']).agg({
        'overs': 'sum',
        'maidens': 'sum',
        'runs_conceded': 'sum',
        'wickets': 'sum',
        'opposition': 'first'  # Keep track of opposition team
    }).reset_index()
    
    # Calculate economy rate
    bowler_match_stats['economy_rate'] = bowler_match_stats['runs_conceded'] / bowler_match_stats['overs']
    
    # Calculate impact score components
    bowler_match_stats['wicket_impact'] = bowler_match_stats['wickets'] * 1.0  # 1 wicket = 1 point
    bowler_match_stats['economy_impact'] = (8.0 - bowler_match_stats['economy_rate']) * 0.5  # Economy below 8 = positive impact
    bowler_match_stats['maiden_impact'] = bowler_match_stats['maidens'] * 0.5  # 1 maiden = 0.5 points
    
    # Calculate total impact score
    bowler_match_stats['impact_score'] = (
        bowler_match_stats['wicket_impact'] + 
        bowler_match_stats['economy_impact'] + 
        bowler_match_stats['maiden_impact']
    )
    
    # Handle NaN and cap the minimum impact score
    bowler_match_stats['impact_score'].fillna(0, inplace=True)
    bowler_match_stats['impact_score'] = bowler_match_stats['impact_score'].clip(-2, None)
    
    # Aggregate player impact across all matches
    bowler_impact = bowler_match_stats.groupby('name').agg({
        'overs': 'sum',
        'wickets': 'sum',
        'runs_conceded': 'sum',
        'maidens': 'sum',
        'impact_score': 'mean'
    }).reset_index()
    
    bowler_impact['overall_economy'] = bowler_impact['runs_conceded'] / bowler_impact['overs']
    bowler_impact.sort_values('impact_score', ascending=False, inplace=True)
    
    return batsman_impact, bowler_impact

def export_analysis_results(match_summary, batsman_stats, bowler_stats, venue_stats, win_factors, batsman_impact, bowler_impact):
    """
    Export analysis results to CSV files.
    
    Args:
        Various DataFrames containing analysis results
    """
    output_dir = os.path.join(os.getcwd(), "analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        match_summary.to_csv(os.path.join(output_dir, 'match_summary.csv'), index=False)
        batsman_stats.to_csv(os.path.join(output_dir, 'batsman_stats.csv'), index=False)
        bowler_stats.to_csv(os.path.join(output_dir, 'bowler_stats.csv'), index=False)
        venue_stats.to_csv(os.path.join(output_dir, 'venue_stats.csv'), index=False)
        win_factors.to_csv(os.path.join(output_dir, 'win_probability_factors.csv'), index=False)
        batsman_impact.to_csv(os.path.join(output_dir, 'batsman_impact.csv'), index=False)
        bowler_impact.to_csv(os.path.join(output_dir, 'bowler_impact.csv'), index=False)
        
        print(f"Analysis results exported to {output_dir}")
    except Exception as e:
        print(f"Error exporting results: {e}")

def main():
    # Load the match data
    df = load_match_data()
    if df is None:
        return
    
    # Clean the data if needed
    if df.isnull().sum().sum() > 0:
        print("Cleaning missing data...")
        df = handle_missing_values(df)
        df = fill_missing_venue_data(df)
    
    # Create match summary
    print("Creating match summary...")
    match_summary = create_match_summary(df)
    
    # Analyze player performance
    print("Analyzing player performance...")
    batsman_stats, bowler_stats = analyze_player_performance(df)
    
    # Analyze venue performance
    print("Analyzing venue performance...")
    venue_stats = analyze_venue_performance(df)
    
    # Calculate win probability factors
    print("Calculating win probability factors...")
    win_factors, match_data = calculate_win_probability_factors(df)
    
    # Calculate player impact scores
    print("Calculating player impact scores...")
    batsman_impact, bowler_impact = calculate_player_impact_score(df)
    
    # Visualize a few sample matches
    sample_matches = df['match_id'].sample(min(3, df['match_id'].nunique())).unique()
    for match_id in sample_matches:
        print(f"Visualizing match {match_id}...")
        fig, _ = visualize_match_progression(df, match_id)
        if fig:
            output_dir = os.path.join(os.getcwd(), "analysis_output", "match_visualizations")
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f'match_{match_id}_progression.png'))
            plt.close(fig)
    
    # Export results
    print("Exporting analysis results...")
    export_analysis_results(match_summary, batsman_stats, bowler_stats, venue_stats, win_factors, batsman_impact, bowler_impact)
    
    print("Match analysis complete!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(os.path.join(os.getcwd(), "analysis_output", "match_visualizations"), exist_ok=True)
    
    # Run the main function
    main()
