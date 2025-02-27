# IPL Data Format Documentation

## Ball-by-Ball CSV Files

The ball-by-ball CSV files contain detailed information about each delivery in an innings. 

### Column Descriptions

- `batting_team`: The team currently batting
- `over_number`, `ball_number`: The current over and ball within that over
- `runs_scored`: Runs scored on this delivery
- `boundaries`: 1 if the ball was hit for a boundary (4 or 6), 0 otherwise
- `dot_balls`: 1 if no runs scored off the bat, 0 otherwise
- `wickets`: 1 if a wicket fell, 0 otherwise
- `extras`: Extras (wides, no balls, etc.) scored on this delivery

#### Batsmen Information
- `batsman1_name`, `batsman2_name`: Names of the two batsmen at the crease
- Associated stats: runs, balls faced, fours hit, sixes hit

#### Bowlers Information
- `bowler1_name`: The **current bowler** (bowling the current delivery)
- `bowler1_overs_bowled`, `bowler1_maidens_bowled`, `bowler1_runs_conceded`, `bowler1_wickets_taken`: Statistics for the current bowler
- `bowler2_name`: The bowler who delivered the **last over** (previous over's bowler)
- `bowler2_overs_bowled`, `bowler2_maidens_bowled`, `bowler2_runs_conceded`, `bowler2_wickets_taken`: Statistics for the previous over's bowler

### Example
In the ball-by-ball data and innings summary data, for each delivery or over summary:
- `bowler1` represents the active bowler delivering the current ball/over
- `bowler2` represents the bowler who bowled the previous over
