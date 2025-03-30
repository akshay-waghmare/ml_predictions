import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_first_balls():
    """
    Analyze model-market agreement on first balls of overs, which are key betting points
    during live cricket matches.
    """
    print("\n===== FIRST BALL OF OVER ANALYSIS =====")
    
    # Load the combined data
    try:
        # Try to use existing combined data first
        df = pd.read_csv('combined_match_data.csv')
        print(f"Loaded combined match data with {len(df)} rows")
    except FileNotFoundError:
        # If not found, try to run the full analysis
        print("Combined data not found, please run the full analysis first")
        return
    
    # Filter to only the first ball of each over
    if 'Over' in df.columns:
        # Extract over and ball numbers
        df['over_num'] = df['Over'].apply(lambda x: float(x.split('.')[0]) if isinstance(x, str) else np.nan)
        df['ball_num'] = df['Over'].apply(lambda x: float(x.split('.')[1]) if isinstance(x, str) else np.nan)
        
        # Filter to first balls only (ball number 1)
        first_balls = df[df['ball_num'] == 1.0].copy()
        print(f"Found {len(first_balls)} first balls of overs")
        
        if len(first_balls) == 0:
            print("No first balls found in the data. Check the Over column format.")
            return
    else:
        print("No 'Over' column found in data")
        return
    
    # Check model-market agreement on first balls
    if 'Model_Market_Team_Agreement' in first_balls.columns:
        agreement_rate = first_balls['Model_Market_Team_Agreement'].mean() * 100
        print(f"\nOn first balls of overs, model agrees with market {agreement_rate:.1f}% of the time")
        
        # Compare to overall agreement rate
        overall_agreement = df['Model_Market_Team_Agreement'].mean() * 100
        print(f"Overall model-market agreement rate: {overall_agreement:.1f}%")
        print(f"Difference: {agreement_rate - overall_agreement:.1f} percentage points")
        
        # Check accuracy on first balls
        if 'Actual_Result' in first_balls.columns and 'Model_Favored_Team' in first_balls.columns and 'Batting_Team' in first_balls.columns:
            # Calculate model accuracy on first balls
            first_ball_model_correct = np.mean(
                (first_balls['Model_Favored_Team'] == first_balls['Batting_Team']) == 
                (first_balls['Actual_Result'] == 1)
            ) * 100
            
            # Calculate market accuracy on first balls
            first_ball_market_correct = np.mean(
                (first_balls['Market_Favored_Team'] == first_balls['Batting_Team']) == 
                (first_balls['Actual_Result'] == 1)
            ) * 100
            
            print(f"\nOn first balls, model prediction accuracy: {first_ball_model_correct:.1f}%")
            print(f"On first balls, market prediction accuracy: {first_ball_market_correct:.1f}%")
            
            # Split by agreement/disagreement
            agree_first_balls = first_balls[first_balls['Model_Market_Team_Agreement']]
            if len(agree_first_balls) > 0:
                agree_accuracy = np.mean(
                    (agree_first_balls['Model_Favored_Team'] == agree_first_balls['Batting_Team']) == 
                    (agree_first_balls['Actual_Result'] == 1)
                ) * 100
                print(f"When model agrees with market on first balls: {agree_accuracy:.1f}% correct ({len(agree_first_balls)} cases)")
            
            disagree_first_balls = first_balls[~first_balls['Model_Market_Team_Agreement']]
            if len(disagree_first_balls) > 0:
                disagree_accuracy = np.mean(
                    (disagree_first_balls['Model_Favored_Team'] == disagree_first_balls['Batting_Team']) == 
                    (disagree_first_balls['Actual_Result'] == 1)
                ) * 100
                print(f"When model disagrees with market on first balls: {disagree_accuracy:.1f}% correct ({len(disagree_first_balls)} cases)")
        
        # Analyze by match phase
        if 'Phase' in first_balls.columns:
            # Check if Phase column has valid values
            valid_phases = first_balls['Phase'].dropna().unique()
            if len(valid_phases) > 0:
                phase_agreement = first_balls.groupby('Phase')['Model_Market_Team_Agreement'].mean() * 100
                
                print("\nModel-market agreement rate by phase (first balls only):")
                for phase, rate in phase_agreement.items():
                    print(f"- {phase}: {rate:.1f}%")
                
                # Only create visualization if we have data
                if len(phase_agreement) > 0:
                    try:
                        plt.figure(figsize=(10, 6))
                        # Use a safer method to create the bar plot
                        plt.bar(phase_agreement.index, phase_agreement.values)
                        plt.title('Model-Market Agreement on First Balls by Match Phase')
                        plt.ylabel('Agreement Rate (%)')
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig('first_ball_agreement_by_phase.png')
                        print("Saved first ball agreement visualization by phase")
                    except Exception as e:
                        print(f"Error creating phase agreement plot: {e}")
                else:
                    print("No phase agreement data to visualize")
            else:
                print("No valid phase values found in the data")
        
        # Analyze by over number
        try:
            # Make sure we have valid over numbers
            valid_overs = first_balls['over_num'].dropna().unique()
            if len(valid_overs) > 0:
                over_agreement = first_balls.groupby('over_num')['Model_Market_Team_Agreement'].mean() * 100
                
                # Only create visualization if we have data
                if len(over_agreement) > 0:
                    plt.figure(figsize=(12, 6))
                    # Plot directly with matplotlib for safer operation
                    plt.plot(over_agreement.index, over_agreement.values, 'o-')
                    plt.title('Model-Market Agreement on First Balls by Over Number')
                    plt.xlabel('Over')
                    plt.ylabel('Agreement Rate (%)')
                    plt.grid(alpha=0.3)
                    
                    # Set more reasonable tick marks based on available data
                    max_over = int(first_balls['over_num'].max()) + 1
                    plt.xticks(range(1, min(max_over, 21)))
                    plt.tight_layout()
                    plt.savefig('first_ball_agreement_by_over.png')
                    print("Saved first ball agreement visualization by over number")
                else:
                    print("No over agreement data to visualize")
            else:
                print("No valid over numbers found in the data")
        except Exception as e:
            print(f"Error analyzing by over number: {e}")
        
        # Value bet analysis for first balls
        if 'Value_Category' in first_balls.columns:
            # Check if we have valid value categories
            valid_categories = first_balls['Value_Category'].dropna().unique()
            if len(valid_categories) > 0:
                value_dist = first_balls['Value_Category'].value_counts(normalize=True) * 100
                print("\nValue category distribution for first balls:")
                for category, pct in value_dist.items():
                    print(f"- {category}: {pct:.1f}%")
                
                # Compare to overall value category distribution
                if 'Value_Category' in df.columns:
                    overall_value_dist = df['Value_Category'].value_counts(normalize=True) * 100
                    print("\nDifference in value category distribution (first balls vs overall):")
                    for category in value_dist.index:
                        if category in overall_value_dist:
                            diff = value_dist[category] - overall_value_dist[category]
                            print(f"- {category}: {diff:+.1f} percentage points")
            else:
                print("No valid value categories found in the data")
                
        # Add a new analysis: model performance by over
        try:
            # Calculate model accuracy by over
            if 'Actual_Result' in first_balls.columns and 'Model_Favored_Team' in first_balls.columns and 'Batting_Team' in first_balls.columns:
                first_balls['Model_Correct'] = (
                    (first_balls['Model_Favored_Team'] == first_balls['Batting_Team']) == 
                    (first_balls['Actual_Result'] == 1)
                )
                
                model_accuracy_by_over = first_balls.groupby('over_num')['Model_Correct'].mean() * 100
                
                # Create visualization
                if len(model_accuracy_by_over) > 0:
                    plt.figure(figsize=(12, 6))
                    plt.plot(model_accuracy_by_over.index, model_accuracy_by_over.values, 'go-', linewidth=2)
                    plt.title('Model Prediction Accuracy by Over (First Balls Only)')
                    plt.xlabel('Over')
                    plt.ylabel('Accuracy (%)')
                    plt.grid(alpha=0.3)
                    
                    # Add 50% baseline
                    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Baseline')
                    
                    # Add overall accuracy
                    overall_accuracy = first_balls['Model_Correct'].mean() * 100
                    plt.axhline(y=overall_accuracy, color='b', linestyle='--', alpha=0.5, 
                               label=f'Overall Accuracy: {overall_accuracy:.1f}%')
                    
                    # Set reasonable tick marks
                    max_over = int(first_balls['over_num'].max()) + 1
                    plt.xticks(range(1, min(max_over, 21)))
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('first_ball_accuracy_by_over.png')
                    print("Saved model accuracy by over visualization")
                    
                    # Print highest accuracy overs
                    top_overs = model_accuracy_by_over.sort_values(ascending=False).head(5)
                    print("\nOvers with highest model accuracy (first balls):")
                    for over, acc in top_overs.items():
                        count = len(first_balls[first_balls['over_num'] == over])
                        print(f"- Over {int(over)}: {acc:.1f}% ({count} observations)")
        except Exception as e:
            print(f"Error analyzing model performance by over: {e}")
            
        # Export first ball data
        first_balls.to_csv('first_ball_analysis.csv', index=False)
        print("\nExported first ball data to first_ball_analysis.csv")
        
    else:
        print("No 'Model_Market_Team_Agreement' column found")

if __name__ == "__main__":
    try:
        analyze_first_balls()
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()