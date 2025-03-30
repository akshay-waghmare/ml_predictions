def investigate_run_rate_outliers(df):
    """
    Investigate outliers in Current Run Rate (CRR), Required Run Rate (RRR), 
    and Projected Score
    """
    print("INVESTIGATING RUN RATE AND SCORE OUTLIERS")
    print("-" * 50)
    
    # Check extreme CRR cases
    high_crr = df[df['current_run_rate'] > 20].copy()
    print("\nHigh Current Run Rate cases (>20):")
    if not high_crr.empty:
        print(f"Found {len(high_crr)} cases")
        high_crr_summary = high_crr[['match_id', 'innings_num', 'over_number', 'ball_number', 
                                   'current_run_rate', 'total_score']].sort_values('current_run_rate', ascending=False)
        print(high_crr_summary.head())
    else:
        print("No cases found")
        
    # Check extreme RRR cases
    high_rrr = df[df['required_run_rate'] > 20].copy()
    print("\nHigh Required Run Rate cases (>20):")
    if not high_rrr.empty:
        print(f"Found {len(high_rrr)} cases")
        high_rrr_summary = high_rrr[['match_id', 'innings_num', 'over_number', 'ball_number', 
                                   'required_run_rate', 'total_score']].sort_values('required_run_rate', ascending=False)
        print(high_rrr_summary.head())
    else:
        print("No cases found")
        
    # Check extreme projected scores
    high_projected = df[df['projected_score'] > 300].copy()
    print("\nHigh Projected Score cases (>300):")
    if not high_projected.empty:
        print(f"Found {len(high_projected)} cases")
        high_proj_summary = high_projected[['match_id', 'innings_num', 'over_number', 
                                          'current_run_rate', 'projected_score', 'total_score']]\
                                         .sort_values('projected_score', ascending=False)
        print(high_proj_summary.head())
    else:
        print("No cases found")
        
    # Analyze first over run rates
    first_over = df[df['over_number'] == 0].copy()
    print("\nFirst Over Run Rate Statistics:")
    if not first_over.empty:
        print(first_over['current_run_rate'].describe())
    else:
        print("No first over data found")
        
    # Analyze last over run rates
    last_over = df[df['over_number'] == 19].copy()
    print("\nLast Over Run Rate Statistics:")
    if not last_over.empty:
        print(last_over['current_run_rate'].describe())
    else:
        print("No last over data found")
        
    # Check for potential data quality issues
    print("\nPotential Data Quality Issues:")
    print(f"Negative CRR values: {len(df[df['current_run_rate'] < 0])}")
    print(f"Negative RRR values: {len(df[df['required_run_rate'] < 0])}")
    print(f"Negative projected scores: {len(df[df['projected_score'] < 0])}")
    
    # Return summary of findings
    return {
        'high_crr_cases': len(high_crr) if 'high_crr' in locals() else 0,
        'high_rrr_cases': len(high_rrr) if 'high_rrr' in locals() else 0,
        'high_projected_cases': len(high_projected) if 'high_projected' in locals() else 0
    }

# Example usage in notebook:
# outlier_results = investigate_run_rate_outliers(df)
