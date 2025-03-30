import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_simplified_match_data():
    print("Creating simplified match data CSV by combining X_test and prediction results...")
    
    # 1. Load both datasets
    try:
        # Load the X_test data
        X_test = pd.read_csv('X_test.csv')
        print(f"Loaded X_test with {len(X_test)} rows")
        
        # Load the prediction results
        predictions = pd.read_csv('prediction_results_with_teams.csv')
        print(f"Loaded prediction results with {len(predictions)} rows")
        
        # Make sure they have the same number of rows
        if len(X_test) != len(predictions):
            print(f"Warning: X_test ({len(X_test)} rows) and predictions ({len(predictions)} rows) have different lengths")
            # Use the smaller length
            min_len = min(len(X_test), len(predictions))
            X_test = X_test.iloc[:min_len]
            predictions = predictions.iloc[:min_len]
            print(f"Using first {min_len} rows from both datasets")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Print available columns for debugging
    print(f"X_test columns: {X_test.columns[:10]}...")
    print(f"Predictions columns: {predictions.columns[:10]}...")
    
    # 2. Function to determine team name based on boolean columns
    def get_team_name(row, prefix):
        teams = ['CSK', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH']
        for team in teams:
            col_name = f'{prefix}_{team}'
            if col_name in row and row[col_name] == 1:
                return team
        return "Unknown"
    
    # 3. Create new combined dataframe
    combined_df = pd.DataFrame()
    
    # 4. Add match phase information if possible
    if 'over_number' in X_test.columns:
        # Define cricket phases
        combined_df['Phase'] = 'Unknown'
        combined_df.loc[X_test['over_number'] <= 6, 'Phase'] = 'Powerplay'
        combined_df.loc[(X_test['over_number'] > 6) & (X_test['over_number'] <= 15), 'Phase'] = 'Middle'
        combined_df.loc[X_test['over_number'] > 15, 'Phase'] = 'Death'
    
    # 5. Add innings number
    if 'innings_num' in X_test.columns:
        combined_df['Innings'] = X_test['innings_num'].astype(int)
    
    # 6. Add over and ball information
    if all(col in X_test.columns for col in ['over_number', 'ball_number']):
        combined_df['Over'] = X_test['over_number'].astype(str) + "." + X_test['ball_number'].astype(str)
    
    # 7. Add batting team in a clear format
    combined_df['Batting_Team'] = X_test.apply(lambda row: get_team_name(row, 'batting_team'), axis=1)
    
    # 8. Add favored team (by market) in a clear format
    combined_df['Market_Favored_Team'] = X_test.apply(lambda row: get_team_name(row, 'favored_team'), axis=1)
    
    # 9. Add match information from X_test
    match_cols = [
        ('total_score', 'Current_Score'),
        ('projected_score', 'Projected_Score'),
        ('current_run_rate', 'Current_Run_Rate'),
        ('required_run_rate', 'Required_Run_Rate'),
        ('target', 'Target'),
        ('balls_remaining', 'Balls_Remaining'),
        ('wickets_fallen', 'Wickets'),
        ('win_percentage', 'Market_Win_Probability')
    ]
    
    for source_col, target_col in match_cols:
        if source_col in X_test.columns:
            combined_df[target_col] = X_test[source_col].fillna("N/A")
    
    # 10. Add model prediction information
    if 'probability_class_0' in predictions.columns:
        combined_df['Model_Prob_Loss'] = predictions['probability_class_0'].round(3)
        
    if 'probability_class_1' in predictions.columns:
        combined_df['Model_Prob_Win'] = predictions['probability_class_1'].round(3)
    
    if 'actual' in predictions.columns:
        combined_df['Actual_Result'] = predictions['actual']
        combined_df['Actual_Outcome'] = np.where(predictions['actual'] == 1, 'Win', 'Loss')
    
    if 'predicted' in predictions.columns:
        combined_df['Predicted_Result'] = predictions['predicted']
        combined_df['Predicted_Outcome'] = np.where(predictions['predicted'] == 1, 'Win', 'Loss')
    
    if 'correct' in predictions.columns:
        combined_df['Prediction_Correct'] = predictions['correct']
    
    # 11. Calculate the model's favored team
    if 'Model_Prob_Win' in combined_df.columns and 'Batting_Team' in combined_df.columns:
        # Create a Bowling_Team column - safer method without direct column access
        # First check if bowling team columns exist
        bowling_teams = []
        teams = ['CSK', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH']
        
        # Check which bowling team columns actually exist
        available_bowling_cols = [f'bowling_team_{team}' for team in teams 
                                if f'bowling_team_{team}' in X_test.columns]
        
        if available_bowling_cols:
            print(f"Found {len(available_bowling_cols)} bowling team columns")
            combined_df['Bowling_Team'] = X_test.apply(lambda row: get_team_name(row, 'bowling_team'), axis=1)
        else:
            # No direct bowling team columns, infer from batting team
            print("No bowling team columns found, inferring from opponent data")
            # Try to derive bowling team from other data
            if 'bowling_team' in predictions.columns:
                combined_df['Bowling_Team'] = predictions['bowling_team']
            else:
                # As a last resort, try to infer bowling team from match data
                print("Attempting to infer bowling team from match data...")
                
                # If we have match_id, we can use that to determine opponents
                if 'match_id' in X_test.columns:
                    # Group by match_id and get unique teams
                    match_teams = {}
                    for i, row in X_test.iterrows():
                        match_id = row.get('match_id', i // 100)  # Use index as fallback
                        batting_team = get_team_name(row, 'batting_team')
                        
                        if match_id not in match_teams:
                            match_teams[match_id] = set()
                        
                        if batting_team != "Unknown":
                            match_teams[match_id].add(batting_team)
                    
                    # Now assign bowling team as the other team in the match
                    combined_df['Bowling_Team'] = "Unknown"
                    for i, row in combined_df.iterrows():
                        match_id = X_test.at[i, 'match_id'] if 'match_id' in X_test.columns else i // 100
                        batting_team = row['Batting_Team']
                        
                        if match_id in match_teams and len(match_teams[match_id]) == 2:
                            other_teams = [t for t in match_teams[match_id] if t != batting_team]
                            if other_teams:
                                combined_df.at[i, 'Bowling_Team'] = other_teams[0]
                                
                    print(f"Inferred bowling team for {(combined_df['Bowling_Team'] != 'Unknown').sum()} rows")
                else:
                    # As a last resort, just leave it unknown
                    combined_df['Bowling_Team'] = "Unknown"
                    print("Warning: Could not determine bowling team")
        
        # Now determine model favored team
        combined_df['Model_Favored_Team'] = np.where(
            combined_df['Model_Prob_Win'] >= 0.5,
            combined_df['Batting_Team'],
            combined_df['Bowling_Team']
        )
    
    # 12. Calculate model vs market difference
    if all(col in combined_df.columns for col in ['Model_Prob_Win', 'Market_Win_Probability']):
        try:
            # First ensure both columns are numeric
            combined_df['Model_Prob_Win'] = pd.to_numeric(combined_df['Model_Prob_Win'], errors='coerce')
            combined_df['Market_Win_Probability'] = pd.to_numeric(combined_df['Market_Win_Probability'], errors='coerce')
            
            # Check if market probability is in percentage format (>1)
            if combined_df['Market_Win_Probability'].mean() > 1:
                print("Converting Market Win Probability from percentage to decimal format")
                combined_df['Market_Win_Probability'] = combined_df['Market_Win_Probability'] / 100
                
            # Then calculate the difference
            combined_df['Model_Market_Diff'] = (
                combined_df['Model_Prob_Win'] - 
                combined_df['Market_Win_Probability']
            ).round(3)
            
            # Calculate absolute difference for analysis
            combined_df['Abs_Model_Market_Diff'] = combined_df['Model_Market_Diff'].abs().round(3)
            
            # Define value bet (model probability > market probability)
            combined_df['Value_Bet'] = np.where(
                combined_df['Model_Market_Diff'] > 0.05,  # At least 5% difference
                'Yes', 'No'
            )
            
            # Add percentage format columns for easier reading
            combined_df['Model_Win_Pct'] = (combined_df['Model_Prob_Win'] * 100).round(1)
            combined_df['Market_Win_Pct'] = (combined_df['Market_Win_Probability'] * 100).round(1)
            combined_df['Diff_Pct'] = (combined_df['Model_Market_Diff'] * 100).round(1)
            
            # Add more specific value bet categories for analysis - FIXED ORDER
            combined_df['Value_Category'] = 'Neutral'
            combined_df.loc[combined_df['Diff_Pct'] > 10, 'Value_Category'] = 'Strong Value'
            combined_df.loc[(combined_df['Diff_Pct'] <= 10) & 
                        (combined_df['Diff_Pct'] > 5), 'Value_Category'] = 'Slight Value'
            combined_df.loc[(combined_df['Diff_Pct'] < -5) &
                        (combined_df['Diff_Pct'] >= -10), 'Value_Category'] = 'Slight Overvalued'
            combined_df.loc[combined_df['Diff_Pct'] < -10, 'Value_Category'] = 'Strong Overvalued'
            
            # Print summary of value bets
            value_count = (combined_df['Value_Bet'] == 'Yes').sum()
            print(f"Found {value_count} value betting opportunities ({value_count/len(combined_df)*100:.1f}%)")
            
            # Print value category distribution
            value_dist = combined_df['Value_Category'].value_counts()
            print("\nValue category distribution:")
            for category, count in value_dist.items():
                print(f"- {category}: {count} ({count/len(combined_df)*100:.1f}%)")
                
            # Adjust Market_Win_Pct based on team perspective
            if all(col in combined_df.columns for col in ['Batting_Team', 'Market_Favored_Team', 'Market_Win_Pct']):
                print("\nAdjusting market win percentage based on team perspective...")
                
                # Create a new column for team perspective win percentage
                combined_df['Market_Favored_Win_Pct'] = combined_df['Market_Win_Pct']
                
                # If market favored team is different from batting team, invert the probability
                # (i.e., if batting team has 30% chance, then bowling team has 70% chance)
                mask = combined_df['Market_Favored_Team'] != combined_df['Batting_Team']
                combined_df.loc[mask, 'Market_Favored_Win_Pct'] = 100 - combined_df.loc[mask, 'Market_Win_Pct']
                
                # Also adjust the decimal version for calculations
                combined_df['Market_Favored_Probability'] = combined_df['Market_Favored_Win_Pct'] / 100
                
                # Count adjustments made
                adj_count = mask.sum()
                print(f"Adjusted {adj_count} rows ({adj_count/len(combined_df)*100:.1f}%) where market favored team != batting team")
                
                # Create a clearer column showing which team the market probability refers to
                combined_df['Market_Prob_Team'] = combined_df['Batting_Team']
                combined_df.loc[mask, 'Market_Prob_Team'] = combined_df.loc[mask, 'Bowling_Team']
                
                # Add a column to indicate if market favors batting team
                combined_df['Market_Favors_Batting'] = combined_df['Market_Favored_Team'] == combined_df['Batting_Team']
                
                # MODIFIED: Recalculate Diff_Pct based on batting team perspective
                # If batting team is market favored, use Model_Win_Pct - Market_Win_Pct
                # If not, use Model_Win_Pct - Market_Favored_Win_Pct
                combined_df['Diff_Pct'] = np.where(
                    combined_df['Batting_Team'] == combined_df['Market_Favored_Team'],
                    combined_df['Model_Win_Pct'] - combined_df['Market_Win_Pct'],
                    combined_df['Model_Win_Pct'] - combined_df['Market_Favored_Win_Pct']
                ).round(1)
                
                print("Recalculated Diff_Pct to always compare from batting team's perspective")
                
            # Create some additional analytical columns
            if 'Model_Favored_Team' in combined_df.columns and 'Market_Favored_Team' in combined_df.columns:
                # We need to check if model and market agree on whether the *batting team* will win
                # Rather than just if they favor the same team
                combined_df['Model_Favors_Batting'] = combined_df['Model_Favored_Team'] == combined_df['Batting_Team']
                combined_df['Market_Favors_Batting'] = combined_df['Market_Favored_Team'] == combined_df['Batting_Team']
                
                # Does model agree with market on whether batting team will win?
                combined_df['Model_Market_Team_Agreement'] = combined_df['Model_Favors_Batting'] == combined_df['Market_Favors_Batting']
                
                # Percentage of agreement
                agreement_pct = (combined_df['Model_Market_Team_Agreement'].mean() * 100).round(1)
                print(f"\nModel agrees with market on batting team's chances in {agreement_pct}% of cases")
                
                # Success rate when model agrees/disagrees with market
                if 'Actual_Result' in combined_df.columns:
                    # When model agrees with market
                    agree_df = combined_df[combined_df['Model_Market_Team_Agreement']]
                    if len(agree_df) > 0:
                        # Check if the agreed prediction was correct
                        agree_success = np.mean(
                            (agree_df['Model_Favors_Batting'] == (agree_df['Actual_Result'] == 1))
                        ) * 100
                        print(f"When model agrees with market: {agree_success:.1f}% correct")
                    
                    # When model disagrees with market
                    disagree_df = combined_df[~combined_df['Model_Market_Team_Agreement']]
                    if len(disagree_df) > 0:
                        # Check if the model's prediction was correct when disagreeing
                        disagree_success = np.mean(
                            (disagree_df['Model_Favors_Batting'] == (disagree_df['Actual_Result'] == 1))
                        ) * 100
                        print(f"When model disagrees with market: {disagree_success:.1f}% correct")
            
        except Exception as e:
            print(f"Error calculating model vs market difference: {e}")
            import traceback
            traceback.print_exc()
    
    # 14. Create visualizations
    try:
        print("\nCreating visualizations...")
        
        # Team performance visualization
        if 'Batting_Team' in combined_df.columns and 'Model_Win_Pct' in combined_df.columns:
            team_analysis = combined_df.groupby('Batting_Team').agg({
                'Model_Win_Pct': 'mean',
                'Market_Win_Pct': 'mean',
                'Actual_Result': ['mean', 'count']
            })
            
            team_analysis.columns = ['Model_Win_Pct', 'Market_Win_Pct', 'Actual_Win_Pct', 'Count']
            team_analysis = team_analysis.sort_values('Actual_Win_Pct', ascending=False)
            
            # Only include teams with sufficient data
            team_analysis = team_analysis[team_analysis['Count'] >= 10]
            
            plt.figure(figsize=(12, 6))
            team_analysis[['Actual_Win_Pct', 'Model_Win_Pct', 'Market_Win_Pct']].plot(
                kind='bar',
                figsize=(12, 6)
            )
            plt.title('Team Performance: Actual vs Model vs Market')
            plt.ylabel('Win Percentage')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('team_performance.png')
            print("Saved team performance visualization")
            
        # Value bet by phase visualization
        if 'Phase' in combined_df.columns and 'Value_Bet' in combined_df.columns:
            phase_value = pd.crosstab(
                combined_df['Phase'], 
                combined_df['Value_Bet'],
                normalize='index'
            ) * 100
            
            plt.figure(figsize=(10, 6))
            phase_value['Yes'].plot(kind='bar', figsize=(10, 6))
            plt.title('Value Betting Opportunities by Match Phase')
            plt.ylabel('Percentage of Value Bets')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('value_bets_by_phase.png')
            print("Saved value betting phase analysis")
            
        # Win probability comparison scatter plot
        if 'Model_Win_Pct' in combined_df.columns and 'Market_Win_Pct' in combined_df.columns:
            plt.figure(figsize=(10, 8))
            plt.scatter(combined_df['Market_Win_Pct'], combined_df['Model_Win_Pct'], 
                       alpha=0.5, s=30, c=combined_df['Actual_Result'], cmap='coolwarm')
            
            # Add diagonal line
            plt.plot([0, 100], [0, 100], 'k--', alpha=0.5)
            
            plt.xlabel('Market Win Percentage')
            plt.ylabel('Model Win Percentage')
            plt.title('Model vs Market Win Probability')
            plt.grid(alpha=0.3)
            plt.colorbar(label='Actual Outcome (0=Loss, 1=Win)')
            plt.tight_layout()
            plt.savefig('model_vs_market.png')
            print("Saved model vs market probability comparison")
            
        # Calibration curve
        if 'Model_Win_Pct' in combined_df.columns and 'Actual_Result' in combined_df.columns:
            # Function to create calibration curve
            def get_calibration_curve(probs, actuals, bins=10):
                bin_size = 100 // bins
                bin_edges = list(range(0, 101, bin_size))
                bin_centers = [i + bin_size/2 for i in range(0, 100, bin_size)]
                
                # Initialize results
                observed_probs = []
                counts = []
                
                # Calculate observed probability for each bin
                for i in range(len(bin_edges) - 1):
                    mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
                    bin_count = mask.sum()
                    
                    if bin_count > 0:
                        observed = actuals[mask].mean() * 100
                        observed_probs.append(observed)
                        counts.append(bin_count)
                    else:
                        observed_probs.append(None)
                        counts.append(0)
                        
                return bin_centers, observed_probs, counts
            
            # Get calibration curves for model and market
            model_bins, model_obs, model_counts = get_calibration_curve(
                combined_df['Model_Win_Pct'], combined_df['Actual_Result']
            )
            
            market_bins, market_obs, market_counts = get_calibration_curve(
                combined_df['Market_Win_Pct'], combined_df['Actual_Result']
            )
            
            # Plot calibration curves
            plt.figure(figsize=(10, 8))
            
            # Plot perfect calibration line
            plt.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Calibration')
            
            # Plot model calibration
            valid_model = [i for i, x in enumerate(model_obs) if x is not None]
            plt.plot([model_bins[i] for i in valid_model], 
                    [model_obs[i] for i in valid_model], 
                    'bo-', label='Model')
            
            # Plot market calibration
            valid_market = [i for i, x in enumerate(market_obs) if x is not None]
            plt.plot([market_bins[i] for i in valid_market], 
                    [market_obs[i] for i in valid_market], 
                    'ro-', label='Market')
            
            # Add bin counts as annotations
            for i in valid_model:
                plt.annotate(str(model_counts[i]), 
                           (model_bins[i], model_obs[i]),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center', 
                           fontsize=8)
            
            plt.xlabel('Predicted Win Probability (%)')
            plt.ylabel('Observed Win Rate (%)')
            plt.title('Calibration Curve: Model vs Market')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('calibration_curve.png')
            print("Saved calibration curve")
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # 15. Export the combined data
    output_file = 'combined_match_data.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\nCreated {output_file} with {len(combined_df)} rows and {len(combined_df.columns)} columns")
    
    # Also create a simplified version with just the most important columns
    try:
        key_columns = [
            'Phase', 'Innings', 'Over', 'Batting_Team', 'Bowling_Team', 
            'Market_Favored_Team', 'Model_Favored_Team',
            'Current_Score', 'Market_Win_Pct', 'Market_Favored_Win_Pct', 
            'Model_Win_Pct', 'Diff_Pct', 'Value_Category',
            'Actual_Outcome', 'Predicted_Outcome', 'Prediction_Correct'
        ]
        
        # Filter to only columns that actually exist
        available_keys = [col for col in key_columns if col in combined_df.columns]
        simplified_df = combined_df[available_keys].copy()
        simplified_df.to_csv('simplified_match_results.csv', index=False)
        print(f"Created simplified_match_results.csv with key columns for easy viewing")
    except Exception as e:
        print(f"Error creating simplified version: {e}")
    
    print("\nColumns in combined data:")
    for col in combined_df.columns:
        print(f"- {col}")
    
    return combined_df

if __name__ == "__main__":
    combined_df = create_simplified_match_data()
    
    # Advanced analysis if data is available
    if combined_df is not None and len(combined_df) > 0:
        print("\nGenerating additional team insights...")
        
        # Team-specific analysis if we have team columns and actual results
        if all(col in combined_df.columns for col in ['Batting_Team', 'Actual_Result', 'Model_Market_Team_Agreement']):
            # Group by team
            team_stats = combined_df.groupby('Batting_Team').agg({
                'Model_Win_Pct': 'mean',
                'Market_Win_Pct': 'mean',
                'Actual_Result': 'mean',
                'Model_Market_Team_Agreement': 'mean',
                'Batting_Team': 'count'
            }).rename(columns={'Batting_Team': 'Count'})
            
            # Add model-market difference
            team_stats['Model_Market_Diff'] = (team_stats['Model_Win_Pct'] - team_stats['Market_Win_Pct']).round(1)
            
            # Add accuracy metrics
            team_stats['Model_Calibration_Error'] = abs(team_stats['Model_Win_Pct'] - team_stats['Actual_Result']*100).round(1)
            team_stats['Market_Calibration_Error'] = abs(team_stats['Market_Win_Pct'] - team_stats['Actual_Result']*100).round(1)
            
            # Format percentage columns
            percentage_cols = ['Model_Win_Pct', 'Market_Win_Pct', 'Actual_Result', 'Model_Market_Team_Agreement']
            for col in percentage_cols:
                if col == 'Actual_Result':
                    team_stats[col] = (team_stats[col] * 100).round(1)
                else:
                    team_stats[col] = team_stats[col].round(1)
            
            # Show which teams are most undervalued by the market
            undervalued = team_stats.sort_values('Model_Market_Diff', ascending=False)
            print("\nTeams most undervalued by the market (model probability > market probability):")
            print(undervalued[['Model_Win_Pct', 'Market_Win_Pct', 'Model_Market_Diff', 'Count']].head(3))
            
            # Show which teams are most overvalued by the market
            overvalued = team_stats.sort_values('Model_Market_Diff')
            print("\nTeams most overvalued by the market (market probability > model probability):")
            print(overvalued[['Model_Win_Pct', 'Market_Win_Pct', 'Model_Market_Diff', 'Count']].head(3))
            
            # Show teams where model outperforms market in calibration
            better_calibrated = team_stats[team_stats['Model_Calibration_Error'] < team_stats['Market_Calibration_Error']]
            better_calibrated = better_calibrated.sort_values('Model_Calibration_Error')
            print("\nTeams where model is better calibrated than market:")
            print(better_calibrated[['Model_Calibration_Error', 'Market_Calibration_Error', 'Count']].head(3))
            
            # Save team stats to CSV
            team_stats.sort_values('Actual_Result', ascending=False).to_csv('team_performance_stats.csv')
            print("\nTeam performance statistics saved to team_performance_stats.csv")