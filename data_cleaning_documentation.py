"""
IPL Match Data Cleaning Documentation

This file documents the data cleaning process for the IPL match data.

Data Structure:
--------------
The dataset contains ball-by-ball information for IPL matches, including:
- Basic match information (match_id, innings_num, batting_team, etc.)
- Ball-by-ball events (runs_scored, wickets, boundaries, dot_balls, extras)
- Win probabilities and team information
- Batsman and bowler statistics
- Venue information
- Historical player statistics

Key Columns:
-----------
1. Match Identifiers:
   - match_id: Unique identifier for each match
   - innings_num: 1.0 or 2.0, representing first or second innings
   - batting_team: Team currently batting
   - winner: Team that won the match

2. Ball-by-Ball Information:
   - over_number: Current over
   - ball_number: Ball within the over
   - runs_scored: Runs scored on this ball
   - boundaries: Whether the ball was hit for a boundary (4 or 6)
   - dot_balls: Whether it was a dot ball (no runs)
   - wickets: Whether a wicket fell on this ball
   - extras: Extra runs scored

3. Player Information:
   - striker_batsman: Batsman on strike
   - batsman1_name, batsman2_name: Names of the two batsmen
   - bowler1_name, bowler2_name: Names of the two bowlers
   - Various statistics for each player (runs, balls, overs, etc.)

4. Historical Stats:
   - Various historical statistics for batsmen and bowlers
   - These columns have been corrected and now have "_corrected" flag columns

Cleaning Process:
---------------
1. Missing Value Handling:
   - Filled missing batsman2 with 'No Partner'
   - Filled missing bowler2 with 'No Bowler'
   - Filled numeric stats with appropriate defaults (0 for counts, median for rates)

2. Historical Stats Correction:
   - Added new columns with "_corrected" suffix to track corrections
   - Corrected historical bowling stats using match-specific data
   - Corrected historical batting stats using match-specific data
   - Removed invalid statistical values (negative or unreasonably high)

3. Data Enrichment:
   - Added run rates calculation
   - Added pressure index during chase
   - Added projected scores
   - Created over-by-over summaries

Data Quality Notes:
-----------------
- Some matches may have incomplete data
- Historical player statistics may still have gaps for some players
- Pressure index is a synthetic metric based on required run rate and wickets

Usage Guidelines:
---------------
1. Always check for missing values before analysis
2. Prefer columns with "_corrected" suffix when available
3. Make sure to handle innings separately for most analyses
4. Use the over-by-over summaries for high-level trends
"""

def get_data_schema_info():
    """
    Returns a description of key columns in the dataset.
    """
    schema = {
        "Match Identifiers": [
            "match_id - Unique identifier for each match",
            "innings_num - 1.0 or 2.0, representing first or second innings",
            "batting_team - Team currently batting",
            "winner - Team that won the match",
            "toss_winner - Team that won the toss",
            "toss_decision - Decision made by toss winner"
        ],
        "Ball-by-Ball Information": [
            "over_number - Current over",
            "ball_number - Ball within the over",
            "runs_scored - Runs scored on this ball",
            "boundaries - Whether the ball was hit for a boundary",
            "dot_balls - Whether it was a dot ball",
            "wickets - Whether a wicket fell on this ball",
            "extras - Extra runs scored"
        ],
        "Player Information": [
            "striker_batsman - Batsman on strike",
            "batsman1_name, batsman2_name - Names of the two batsmen",
            "bowler1_name, bowler2_name - Names of the two bowlers",
            "batsman1_runs, batsman2_runs - Runs scored by each batsman",
            "bowler1_wickets_taken, bowler2_wickets_taken - Wickets taken"
        ],
        "Historical Stats": [
            "batsman1_historical_average - Historical batting average",
            "batsman1_historical_strike_rate - Historical strike rate",
            "bowler1_historical_average - Historical bowling average",
            "bowler1_historical_economy - Historical economy rate",
            "bowler1_historical_strike_rate - Historical bowling strike rate",
            "(and similar columns for batsman2 and bowler2)",
            "Various _corrected columns indicating if stats were fixed"
        ],
        "Added Metrics": [
            "current_run_rate - Run rate at this point in the innings",
            "required_run_rate - Required run rate for 2nd innings",
            "projected_score - Projected final score based on current rate",
            "pressure_index - Index of pressure (0-10) during chase"
        ]
    }
    
    return schema

def print_data_schema():
    """Prints the data schema in a readable format"""
    schema = get_data_schema_info()
    print("IPL Match Data Schema:\n")
    
    for category, columns in schema.items():
        print(f"{category}:")
        for col in columns:
            print(f"  - {col}")
        print()

def get_cleaning_recommendations():
    """
    Returns recommendations for additional data cleaning
    """
    recommendations = [
        "1. Check for and handle any remaining missing values in key columns",
        "2. Consider interpolating missing historical player statistics for better analysis",
        "3. Validate and possibly adjust venue statistics that have limited match data",
        "4. Create consistent team name formats across all columns",
        "5. Add a mapping of player names to resolve variations in spelling",
        "6. Consider adding seasonal context (which IPL season each match belongs to)",
        "7. Calculate player form metrics (recent performance vs historical averages)",
        "8. Add team-specific stats (e.g., head-to-head records)"
    ]
    
    return recommendations

# Example usage
if __name__ == "__main__":
    print("IPL Data Cleaning Documentation")
    print("================================\n")
    print_data_schema()
    
    print("Recommendations for Further Cleaning:")
    for i, rec in enumerate(get_cleaning_recommendations()):
        print(rec)
