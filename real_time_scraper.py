# real_time_scraper.py
import csv
import os
import re
import sys
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from playwright.sync_api import sync_playwright
from enum import Enum
from fuzzywuzzy import process
import joblib 
import pickle
import os
from feature_processor import FeatureProcessor

# Import functions from your existing scripts
from espnscraper_ballbyball import parse_ball_details, get_current_batting_team
from espnscraper import ensure_directory_exists, get_match_info_from_url, get_venue_stats, fetch_player_batting_summary, process_csv_data, scrape_all_overs_data
from batting_stats import BattingHistoryDownloader
from feature_processor import FeatureProcessor

class IPLTeam(Enum):
    MI = "Mumbai Indians"
    DC = "Delhi Capitals"
    SRH = "Sunrisers Hyderabad"
    RCB = "Royal Challengers Bangalore"
    KKR = "Kolkata Knight Riders"
    PBKS = "Punjab Kings"
    CSK = "Chennai Super Kings"
    RR = "Rajasthan Royals"
    GT = "Gujarat Titans"
    LSG = "Lucknow Super Giants"

def ensure_proper_dtype(df, feature, desired_dtype):
    """
    Ensures a feature has the proper data type, with special handling for booleans
    
    Args:
        df: DataFrame containing the feature
        feature: Name of the feature to convert
        desired_dtype: Target data type ('bool', 'float64', 'int64', etc.)
    
    Returns:
        None (modifies df in place)
    """
    try:
        if desired_dtype == 'bool':
            # For boolean columns, we need special handling
            # Convert numeric values to boolean (0 -> False, non-zero -> True)
            if pd.api.types.is_numeric_dtype(df[feature]):
                df[feature] = df[feature].astype(bool)
            # Convert string values to boolean
            elif pd.api.types.is_string_dtype(df[feature]) or df[feature].dtype == 'object':
                # For string columns, normalize the strings first
                true_values = ['true', 't', 'yes', 'y', '1', 'on']
                df[feature] = df[feature].astype(str).str.lower().isin(true_values)
            else:
                # For other types (like NoneType), convert to bool directly
                df[feature] = df[feature].astype(bool)
        else:
            # For numeric types, use standard conversion
            df[feature] = df[feature].astype(desired_dtype)
    except Exception as e:
        print(f"Warning: Could not convert {feature} to {desired_dtype}: {e}")
        # Provide sensible defaults for failed conversions
        if desired_dtype == 'bool':
            df[feature] = False
        elif desired_dtype == 'float64' or 'float' in desired_dtype:
            df[feature] = 0.0
        elif 'int' in desired_dtype:
            df[feature] = 0
            
def clean_column_name(col_name):
    """Standardize column names to match training data format"""
    # Replace spaces with underscores
    col_name = str(col_name).replace(' ', '_')
    # Replace commas with underscores
    col_name = col_name.replace(',', '_')
    # Remove any other special characters
    col_name = re.sub(r'[^\w_]', '', col_name)
    return col_name

def launch_prediction_display(url=None):
    import subprocess
    import sys
    import os
    
    # Get the URL and extract match info
    if not url:
        url = "https://www.espncricinfo.com/series/ipl-2025-1449924/gujarat-titans-vs-mumbai-indians-9th-match-1473446/live-cricket-score"
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get season folder and match ID from URL
    season_folder, match_id = get_match_info_from_url(url)
    
    match_dir = os.path.join(current_dir, season_folder, match_id, "ball_by_ball")
    # Ensure the directory exists
    os.makedirs(match_dir, exist_ok=True)
    
    # Construct path to CSV file
    csv_path = os.path.join(current_dir, season_folder, match_id, "ball_by_ball", f"{match_id}_ball_feeders.csv")
    
    # Path to the prediction_display.py script
    display_script = os.path.join(os.path.dirname(__file__), 'prediction_display.py')
    
    # Launch in a new process with CSV path as argument 
    subprocess.Popen([sys.executable, display_script, csv_path])
    print(f"Launched prediction display window with CSV path: {csv_path}")
    
# Replace your model loading code with this
def load_model(model_paths):
    """Try to load model using both joblib and pickle"""
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Try loading with joblib first
                model = joblib.load(path)
                print(f"✅ Successfully loaded model from {path} using joblib: {type(model).__name__}")
                return model
            except Exception as joblib_error:
                print(f"Joblib loading error for {path}: {joblib_error}")
                try:
                    # Fall back to pickle if joblib fails
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"✅ Successfully loaded model from {path} using pickle: {type(model).__name__}")
                    return model
                except Exception as pickle_error:
                    print(f"Pickle loading error for {path}: {pickle_error}")
    
    print("❌ Could not load model from any of the specified paths")
    return None

def calculate_pressure_index(enhanced_ball_data, match_state):
    """
    Calculate a pressure index based on required run rate and wickets remaining.
    
    Args:
        enhanced_ball_data (dict): Current ball data
        match_state (dict): Current match state tracking wickets and other metrics
    
    Returns:
        float: Pressure index on a scale of 0-10
    """
    pressure_index = 0
    
    # Only calculate for 2nd innings
    if not enhanced_ball_data.get('is_second_innings', False):
        return pressure_index
    
    # Base pressure from required run rate
    if 'required_run_rate' in enhanced_ball_data:
        pressure_index = enhanced_ball_data['required_run_rate'] / 2
    
    # Add pressure from wickets lost
    wickets_lost = match_state.get('wickets_lost', 0)
    pressure_index += wickets_lost * 0.5
    
    # Normalize to 0-10 scale (assuming max possible pressure is ~20)
    max_possible_pressure = 10
    pressure_index = min((pressure_index / max_possible_pressure) * 10, 10)
    
    # Round to 1 decimal place
    pressure_index = round(pressure_index, 1)
    
    return pressure_index

def fuzzy_match_team(team_name):
    """
    Fuzzy matches a given team name to the IPLTeam enum.
    Returns: The enum value if a match is found, otherwise None.
    """
    try:
        # Extract team names from the enum
        team_choices = [team.value for team in IPLTeam]
        
        # Find the best match using fuzzy matching
        best_match, score = process.extractOne(team_name, team_choices)
        
        # Only return if the score is above a certain threshold (e.g., 80)
        if score >= 80:
            # Find the enum that matches the best match
            for team in IPLTeam:
                if team.value == best_match:
                    return team.name  # Return the enum name (e.g., "PBKS")
        
        print(f"Fuzzy match failed for team name: {team_name}")
        return None
    except Exception as e:
        print(f"Error during fuzzy matching: {e}")
        return None

class IPLLiveDataCollector:
    def __init__(self, model_path="best_model_Random Forest.pkl"):
        self.model = self.load_model(model_path)
        self.historical_stats = {}  # Cache for player historical stats
        self.venue_stats = {}       # Cache for venue stats
        self.last_ball_data = None  # To track what we've already processed
        self.match_info = {}        # Store match info (toss, etc.)
        
    def load_model(self, model_path):
        """Load the trained model"""
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def find_live_matches(self):
        """Find ongoing IPL matches"""
        # Implementation to check ESPN Cricinfo for live matches
        pass
        
    def initialize_match(self, match_url):
        """Set up data collection for a specific match"""
        # 1. Fetch match info (teams, venue, toss)
        # 2. Pre-load player historical stats
        # 3. Get venue statistics
        pass
    
    def collect_ball_data(self, match_url):
        """Collect data for the latest ball"""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(match_url)
            
            # Get latest ball commentary
            latest_ball = page.locator("xpath=//div[contains(@class, 'match-comment-over')][1]")
            ball_text = latest_ball.inner_text()
            
            # Check if this is a new ball
            if ball_text != self.last_ball_data:
                self.last_ball_data = ball_text
                ball_info = self.parse_ball_commentary(ball_text)
                return ball_info
            
            return None
    
    def build_model_input(self, ball_info):
        """Transform ball data into model input format"""
        # Combine ball info with historical stats
        model_input = {
            # Match info
            'match_id': self.match_info['match_id'],
            'innings_num': ball_info['innings_num'],
            'batting_team': ball_info['batting_team'],
            
            # Ball info
            'over_number': ball_info['over_number'],
            'ball_number': ball_info['ball_number'],
            'runs_scored': ball_info['runs_scored'],
            'boundaries': 1 if ball_info['runs_scored'] in [4, 6] else 0,
            'dot_balls': 1 if ball_info['runs_scored'] == 0 else 0,
            'wickets': ball_info['is_wicket'],
            'extras': ball_info['is_extra'],
            
            # Match state
            'favored_team': ball_info['favored_team'],
            'win_percentage': ball_info['win_percentage'],
            
            # Players
            'striker_batsman': ball_info['striker_batsman'],
            # Include other batsmen and bowler stats
            
            # Pre-loaded historical stats for players
            # ...
        }
        return pd.DataFrame([model_input])
    
    def predict_outcome(self, model_input):
        """Make prediction using the model"""
        prediction = self.model.predict_proba(model_input)[0]
        return {
            'win_probability': prediction[1],
            'loss_probability': prediction[0]
        }
    
    def start_monitoring(self, match_url, polling_interval=15):
        """Start monitoring a match"""
        self.initialize_match(match_url)
        
        print(f"Starting to monitor match: {match_url}")
        while True:
            try:
                ball_data = self.collect_ball_data(match_url)
                if (ball_data):
                    print(f"New ball: {ball_data['over_number']}.{ball_data['ball_in_over']}")
                    model_input = self.build_model_input(ball_data)
                    prediction = self.predict_outcome(model_input)
                    print(f"Win probability: {prediction['win_probability']:.2%}")
                    
                    # Save the prediction to a file or database
                    self.save_prediction(ball_data, prediction)
                
                # Sleep until next poll
                time.sleep(polling_interval)
                
            except KeyboardInterrupt:
                print("Monitoring stopped by user")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(polling_interval * 2)  # Wait longer after an error

import time

def wait_for_new_ball_update(page, match_state):
    """Continuously monitors the commentary divs and waits for a new ball update."""
    print("⏳ Waiting for the next ball to be bowled...")

    # Locate all commentary divs 
    commentary_divs = page.locator(".lg\\:hover\\:ds-bg-ui-fill-translucent.ds-hover-parent.ds-relative > .ds-text-tight-m.ds-font-regular.ds-flex.ds-px-3.ds-py-2.lg\\:ds-px-4.lg\\:ds-py-\\[10px\\].ds-items-start.ds-select-none.lg\\:ds-select-auto")
    
    # Get initial information from the first div
    initial_count = commentary_divs.count()
    initial_text = None
    if initial_count > 0:
        initial_text = commentary_divs.nth(0).inner_text().strip()
    
    # Keep track of processed ball numbers
    processed_balls = set()
    if 'last_ball' in match_state:
        processed_balls.add(match_state['last_ball'])
        
    # Store previous commentary text to detect updates that don't change ball number
    previous_commentary = initial_text
    
    while True:
        time.sleep(1)  # Poll every second
        
        if commentary_divs.count() == 0:
            continue
            
        # Get the first div which should contain the latest ball
        first_div = commentary_divs.nth(0)
        current_text = first_div.inner_text().strip()
        
        # Skip if text hasn't changed at all
        #if current_text == previous_commentary:
        if current_text == previous_commentary:
            continue
            
        # Update previous commentary regardless of whether this is a new ball
        previous_commentary = current_text
        
        # Extract ball number
        ball_number_element = first_div.locator(
            "xpath=.//span[contains(@class, 'ds-text-tight-s') and contains(@class, 'ds-font-regular') and contains(@class, 'ds-text-typo-mid1')]"
        )
        
        if ball_number_element.count() == 0:
            continue
            
        ball_number = ball_number_element.inner_text().strip()
        
        # Skip if this is not a ball (e.g., a between-overs comment)
        import re
        if not re.match(r'^\d+\.\d+$', ball_number):
            continue
            
        # Skip if we've already processed this ball
        if ball_number in processed_balls:
            print(f"Already processed ball {ball_number}, waiting for next...")
            continue
            
        # We have a new ball!
        print(f"New ball detected: {ball_number}")
        processed_balls.add(ball_number)
        
        # Store the current ball in match state
        match_state['last_ball'] = ball_number
        
        # Extract all required information
        runs_or_event = first_div.locator(
            "xpath=.//div[contains(@class, 'ds-text-tight-m') and contains(@class, 'ds-font-bold')]/span"
        ).inner_text().strip()
        
        short_commentary = first_div.locator(
            "xpath=.//div[contains(@class, 'ds-leading-')]"
        ).evaluate_all("nodes => nodes.map(node => node.textContent.trim()).join(' ')")
        
        #detailed_commentary = first_div.locator("xpath=.//p[contains(@class, 'ci-html-content')]").inner_text().strip()
        
        # Extract over info
        over_info = extract_over_info(ball_number)
        
        # Check if we need to extract end-of-over data for the previous over
        if over_info and 'over_number' in over_info:
            current_over = over_info['over_number']
            
            # Initialize over_summaries if not exists
            if 'over_summaries' not in match_state:
                match_state['over_summaries'] = {}
            
            # Loop over the previous 3 overs (n, n-1, n-2)
            for prev_over in range(max(1, current_over-2), current_over+1):
                # Only extract if we haven't already processed this over
                if prev_over not in match_state['over_summaries']:
                    print(f"Extracting end-of-over data for over {prev_over}...")
                    end_of_over_data = get_end_of_over_data(page, prev_over)
                    
                    if end_of_over_data:
                        # Cache the data
                        match_state['over_summaries'][prev_over] = end_of_over_data
                        print(f"📊 Cached End of Over {prev_over} Summary: {end_of_over_data}")
                    else:
                        # Mark as attempted even if no data found
                        match_state['over_summaries'][prev_over] = None
                        print(f"⚠️ No end-of-over data found for over {prev_over}")
        
        # Check if we need to extract end-of-over data for previous overs
        if over_info and 'over_number' in over_info:
            current_over = over_info['over_number']
            
            # Initialize over_summaries if not exists
            if 'over_summaries' not in match_state:
                match_state['over_summaries'] = {}
            
            # Loop over the previous 3 overs (n, n-1, n-2)
            current_over = over_info['over_number']
            for prev_over in range(max(1, current_over-2), current_over+1):
                # Only extract if we haven't already processed this over
                if prev_over not in match_state['over_summaries']:
                    print(f"Extracting end-of-over data for over {prev_over}...")
                    end_of_over_data = get_end_of_over_data(page, prev_over)
                    
                    if end_of_over_data:
                        # Cache the data
                        match_state['over_summaries'][prev_over] = end_of_over_data
                        print(f"📊 Cached End of Over {prev_over} Summary: {end_of_over_data}")
                    else:
                        # Mark as attempted even if no data found
                        match_state['over_summaries'][prev_over] = None
                        print(f"⚠️ No end-of-over data found for over {prev_over}")
        
        # Extract win probability information
        favored_team, win_percentage = extract_win_probability(page)
        print(f"Favored team: {favored_team} with {win_percentage}% win probability")
        
        # Extract current batting team information
        batting_info = extract_current_batting_team(page)
        if batting_info:
            print(f"Current batting team: {batting_info['batting_team']}, Score: {batting_info['runs']}/{batting_info['wickets']}, Target: {batting_info['target']}")
            
        run_rates = extract_run_rates(page)
        # Return all the extracted data
        return {
            'ball_number': ball_number,
            'runs_or_event': runs_or_event,
            'short_commentary': short_commentary,
            'over_info': over_info,
            'favored_team': favored_team,
            'win_percentage': win_percentage,
            'forecast_score':batting_info.get('forecast_score') if batting_info else None,
            'batting_team': batting_info.get('batting_team') if batting_info else None,
            'bowling_team': batting_info.get('bowling_team') if batting_info else None,
            'batting_team_score': f"{batting_info.get('runs', 0)}/{batting_info.get('wickets', 0)}" if batting_info else None,
            'is_second_innings': batting_info.get('is_second_innings', False) if batting_info else False,
            'target': batting_info.get('target') if batting_info else None,
            'current_over': batting_info.get('current_over', 0) if batting_info else 0,
            'overs_info': batting_info.get('overs_info') if batting_info else None,
            'current_over': batting_info.get('current_over', 0) if batting_info else 0,
            'overs_info': batting_info.get('overs_info') if batting_info else None,
            'current_run_rate': run_rates.get('current_run_rate'),
            'required_run_rate': run_rates.get('required_run_rate'),
            'last_5_overs_runs': run_rates.get('last_5_overs_runs'),
            'last_5_overs_wickets': run_rates.get('last_5_overs_wickets'),
            'last_5_overs_run_rate': run_rates.get('last_5_overs_run_rate')
        }

def extract_over_info(ball_number):
    """Extract over and ball numbers from ball number string"""
    import re
    # Parse format like "41.3"
    match = re.match(r'^(\d+)\.(\d+)$', ball_number)
    
    if match:
        return {
            'over_number': int(match.group(1)),
            'ball_number': int(match.group(2))
        }
    return None

def get_end_of_over_data(page, over_number):
    """
    Find and extract data from the "END OF OVER" div for the specified over number
    """
    try:
        # Use JavaScript execution to find the END OF OVER div
        script = f"""
        () => {{
            const endOfOverDivs = Array.from(document.querySelectorAll('div > .ds-border-b.ds-border-line'))
            .filter(div => div.parentElement && 
                   div.parentElement.innerText.trim().startsWith('END OF OVER {over_number}'))
            .map(div => div.parentElement);
            
            if (endOfOverDivs.length > 0) {{
            return endOfOverDivs[0].outerHTML;
            }}
            return null;
        }}
        """
        
        # Execute the JavaScript in the page context
        result = page.evaluate(script)
        
        if (result):
            # Process the end-of-over summary text
            return parse_end_of_over_summary(result)
        
        return None
        
    except Exception as e:
        print(f"Error getting end of over data: {e}")
        return None

def parse_end_of_over_summary(text):
    """
    Parse the END OF OVER summary text to extract key statistics including
    batsmen and bowlers information.
    
    Args:
        text (str): HTML or text content of the end-of-over panel
        
    Returns:
        dict: Comprehensive summary with runs, wickets, batsmen and bowlers details
    """
    import re
    from bs4 import BeautifulSoup
    import traceback

    # Initialize the summary dictionary with default values
    summary = {}
    
    try:
        # Check if the input is HTML (common when using evaluate in Playwright)
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            
            # Extract over number from the uppercase heading
            over_header = soup.select_one('.ds-uppercase')
            if over_header:
                over_match = re.search(r'END OF OVER (\d+)', over_header.text.strip().upper())
                if over_match:
                    summary['over_number'] = int(over_match.group(1))
            
            # Extract runs and wickets in the over
            runs_wickets_text = soup.select_one('.ds-block.ds-mt-px')
            if runs_wickets_text:
                runs_match = re.search(r'(\d+) runs?', runs_wickets_text.text)
                if runs_match:
                    summary['over_runs'] = int(runs_match.group(1))
                
                wickets_match = re.search(r'(\d+) wickets?', runs_wickets_text.text)
                if wickets_match:
                    summary['over_wickets'] = int(wickets_match.group(1))
            
            # Extract team score
            score_element = soup.select_one('.ds-text-tight-m.ds-font-bold:not(.ds-uppercase)')
            if score_element:
                score_match = re.search(r'([A-Z]+)\s*:\s*(\d+)/(\d+)', score_element.text)
                if score_match:
                    summary['batting_team'] = score_match.group(1)
                    summary['total_score'] = int(score_match.group(2))
                    summary['total_wickets'] = int(score_match.group(3))
            
            # Extract run rates
            rates_element = soup.select_one('span:contains("CRR")')
            if rates_element:
                crr_match = re.search(r'CRR\s*:\s*(\d+\.\d+)', rates_element.text)
                if crr_match:
                    summary['current_run_rate'] = float(crr_match.group(1))
                
                rrr_match = re.search(r'RRR\s*:\s*(\d+\.\d+)', rates_element.text)
                if rrr_match:
                    summary['required_run_rate'] = float(rrr_match.group(1))
                
                # Extract target information
                target_match = re.search(r'Need (\d+) from (\d+)b', rates_element.text)
                if target_match:
                    summary['runs_needed'] = int(target_match.group(1))
                    summary['balls_remaining'] = int(target_match.group(2))
            
            # Extract batsmen and bowlers using more robust selectors
            # Use attribute selectors instead of class selectors with slashes
            batsmen_container = soup.find('div', attrs={'class': lambda x: x and 'ds-w-1/2' in x and not 'ds-border-l' in x})
            bowlers_container = soup.find('div', attrs={'class': lambda x: x and 'ds-w-1/2' in x and 'ds-border-l' in x})
            
            # Process batsmen
            if batsmen_container:
                batsmen_divs = batsmen_container.find_all('div', attrs={'class': lambda x: x and 'ds-justify-between' in x})
                
                if len(batsmen_divs) >= 1:
                    # First batsman
                    spans = batsmen_divs[0].find_all('span')
                    if len(spans) >= 2:
                        batsman1_name = spans[0].text.strip()
                        batsman1_info = spans[1].text.strip()
                        batsman1_stats = re.search(
                            r'(\d+)\s*\((\d+)b(?:\s*(\d+)x4)?(?:\s*(\d+)x6)?\)', 
                            batsman1_info
                        )
                        
                        if batsman1_stats:
                            summary['batsman1_name'] = batsman1_name
                            summary['batsman1_runs'] = int(batsman1_stats.group(1))
                            summary['batsman1_balls'] = int(batsman1_stats.group(2))
                            summary['batsman1_fours'] = int(batsman1_stats.group(3)) if batsman1_stats.group(3) else 0
                            summary['batsman1_sixes'] = int(batsman1_stats.group(4)) if batsman1_stats.group(4) else 0
                
                if len(batsmen_divs) >= 2:
                    # Second batsman
                    spans = batsmen_divs[1].find_all('span')
                    if len(spans) >= 2:
                        batsman2_name = spans[0].text.strip()
                        batsman2_info = spans[1].text.strip()
                        batsman2_stats = re.search(
                            r'(\d+)\s*\((\d+)b(?:\s*(\d+)x4)?(?:\s*(\d+)x6)?\)', 
                            batsman2_info
                        )
                        
                        if batsman2_stats:
                            summary['batsman2_name'] = batsman2_name
                            summary['batsman2_runs'] = int(batsman2_stats.group(1))
                            summary['batsman2_balls'] = int(batsman2_stats.group(2))
                            summary['batsman2_fours'] = int(batsman2_stats.group(3)) if batsman2_stats.group(3) else 0
                            summary['batsman2_sixes'] = int(batsman2_stats.group(4)) if batsman2_stats.group(4) else 0
            
            # Process bowlers
            if bowlers_container:
                bowler_divs = bowlers_container.find_all('div', attrs={'class': lambda x: x and 'ds-justify-between' in x})
                
                if len(bowler_divs) >= 1:
                    # First bowler
                    bowler1_spans = bowler_divs[0].find_all('span')
                    if len(bowler1_spans) >= 2:
                        bowler1_name_span = bowler1_spans[0].find('span', class_='ds-mt-px')
                        bowler1_name = bowler1_name_span.text.strip() if bowler1_name_span else bowler1_spans[0].text.strip()
                        bowler1_info = bowler1_spans[-1].text.strip()
                        bowler1_stats = re.search(r'(\d+)-(\d+)-(\d+)-(\d+)', bowler1_info)
                        
                        if bowler1_stats:
                            summary['bowler1_name'] = bowler1_name
                            summary['bowler1_overs'] = int(bowler1_stats.group(1))
                            summary['bowler1_maidens'] = int(bowler1_stats.group(2))
                            summary['bowler1_runs'] = int(bowler1_stats.group(3))
                            summary['bowler1_wickets'] = int(bowler1_stats.group(4))
                
                if len(bowler_divs) >= 2:
                    # Second bowler
                    bowler2_spans = bowler_divs[1].find_all('span')
                    if len(bowler2_spans) >= 2:
                        bowler2_name_span = bowler2_spans[0].find('span', class_='ds-mt-px')
                        bowler2_name = bowler2_name_span.text.strip() if bowler2_name_span else bowler2_spans[0].text.strip()
                        bowler2_info = bowler2_spans[-1].text.strip()
                        bowler2_stats = re.search(r'(\d+)-(\d+)-(\d+)-(\d+)', bowler2_info)
                        
                        if bowler2_stats:
                            summary['bowler2_name'] = bowler2_name
                            summary['bowler2_overs'] = int(bowler2_stats.group(1))
                            summary['bowler2_maidens'] = int(bowler2_stats.group(2))
                            summary['bowler2_runs'] = int(bowler2_stats.group(3))
                            summary['bowler2_wickets'] = int(bowler2_stats.group(4))
        else:
            # Fallback to text-based parsing if not HTML
            # ... existing text-based parsing code ...
            pass

    except Exception as e:
        print(f"Error parsing end of over summary: {e}")
        traceback.print_exc()
    
    return summary


def extract_scorecard_data(page, player_stats_cache=None):
    """Extract detailed batsman and bowler information from the live scorecard"""
    try:
        # Find the scorecard table
        scorecard_data = {}
        
        # Extract batsmen data - add more robust error handling
        batsmen_elements = page.query_selector_all(".ds-w-full .ds-table tbody:first-of-type tr")
        
        if not batsmen_elements or len(batsmen_elements) < 2:
            print("Warning: Could not find batsmen rows in scorecard")
            return None
            
        # Take only first 2 elements which should be batsmen
        batsmen = batsmen_elements[:2]
        
        # Initialize batsmen list
        scorecard_data['batsmen'] = []
        
        for batsman_row in batsmen:
            name_element = batsman_row.query_selector("td:first-child a")
            
            # Skip if no valid name element
            if not name_element:
                continue
                
            # Extract stats cells safely
            stats_cells = batsman_row.query_selector_all("td")
            if len(stats_cells) < 6:  # We need at least 6 cells for complete data
                print(f"Warning: Incomplete batsman data row, found {len(stats_cells)} cells")
                continue
                
            try:
                # Extract name and check if on strike
                full_name = name_element.inner_text().strip()
                is_on_strike = "*" in full_name
                name = full_name.replace("*", "").strip()
                
                # Extract player URL for historical stats
                player_url = name_element.get_attribute("href")
                player_id = extract_player_id_from_profile_url(player_url)
                
                # Extract batting style - with safety check
                batting_style_element = batsman_row.query_selector("td:first-child .ds-text-typo-mid3")
                batting_style = batting_style_element.inner_text().strip() if batting_style_element else "Unknown"
                
                # Extract stats safely with conversion error handling
                runs = stats_cells[1].inner_text().strip()
                balls = stats_cells[2].inner_text().strip() 
                fours = stats_cells[3].inner_text().strip()
                sixes = stats_cells[4].inner_text().strip()
                strike_rate = stats_cells[5].inner_text().strip()
                
                # Get historical stats if cache is available
                historical_stats = {}
                if player_stats_cache and player_id:
                    batting_stats, _ = player_stats_cache.get_player_stats(player_id, name)
                    if batting_stats:
                        historical_stats.update({
                            'historical_average': batting_stats['average'],
                            'historical_strike_rate': batting_stats['strike_rate'],
                            'historical_matches': batting_stats['matches'],
                            'historical_runs': batting_stats['runs'],
                            'historical_fifties': batting_stats['fifties'],
                            'historical_hundreds': batting_stats['hundreds']
                        })
                
                # Add to batsmen list with safe conversions
                batsman_data = {
                    'name': name,
                    'is_on_strike': is_on_strike,
                    'player_id': player_id,
                    'batting_style': batting_style,
                    'runs': int(runs) if runs.isdigit() else 0,
                    'balls': int(balls) if balls.isdigit() else 0,
                    'fours': int(fours) if fours.isdigit() else 0, 
                    'sixes': int(sixes) if sixes.isdigit() else 0,
                    'strike_rate': float(strike_rate) if strike_rate.replace('.','',1).isdigit() else 0.0
                }
                
                # Add historical stats if available
                batsman_data.update(historical_stats)
                
                scorecard_data['batsmen'].append(batsman_data)
                
            except Exception as inner_e:
                print(f"Error processing batsman row: {inner_e}")
                continue
        
        # Similarly process bowler data with added historical stats
        bowlers = page.query_selector_all(".ds-w-full .ds-table tbody:nth-of-type(2) tr")
        scorecard_data['bowlers'] = []
        
        for bowler_row in bowlers:
            try:
                name_element = bowler_row.query_selector("td:first-child a")
                if not name_element:
                    continue
                    
                name = name_element.inner_text().strip()
                player_url = name_element.get_attribute("href")
                player_id = extract_player_id_from_profile_url(player_url)
                
                stats_cells = bowler_row.query_selector_all("td")
                if len(stats_cells) < 6:
                    continue
                    
                overs = stats_cells[1].inner_text().strip()
                maidens = stats_cells[2].inner_text().strip()
                runs = stats_cells[3].inner_text().strip()
                wickets = stats_cells[4].inner_text().strip()
                economy = stats_cells[5].inner_text().strip()
                
                # Get historical stats if cache is available
                historical_stats = {}
                if player_stats_cache and player_id:
                    _, bowling_stats = player_stats_cache.get_player_stats(player_id, name)
                    if bowling_stats:
                        historical_stats.update({
                            'historical_average': bowling_stats['average'],
                            'historical_economy': bowling_stats['economy'],
                            'historical_strike_rate': bowling_stats['strike_rate'],
                            'historical_wickets': bowling_stats['wickets'],
                            'historical_matches': bowling_stats['matches'],
                            'historical_best_bowling': bowling_stats['best_bowling']
                        })
                
                # Create bowler data dictionary
                bowler_data = {
                    'name': name,
                    'player_id': player_id,
                    'overs': float(overs) if overs.replace('.','',1).isdigit() else 0.0,
                    'maidens': int(maidens) if maidens.isdigit() else 0,
                    'runs': int(runs) if runs.isdigit() else 0,
                    'wickets': int(wickets) if wickets.isdigit() else 0,
                    'economy': float(economy) if economy.replace('.','',1).isdigit() else 0.0
                }
                
                # Add historical stats if available
                bowler_data.update(historical_stats)
                
                scorecard_data['bowlers'].append(bowler_data)
                
            except Exception as inner_e:
                print(f"Error processing bowler row: {inner_e}")
                continue
        
        return scorecard_data
        
    except Exception as e:
        print(f"Error extracting scorecard data: {e}")
        traceback.print_exc()
        return None

def extract_match_info(page):
    """Extract venue, toss and other match information using the scorecard table rows"""
    match_info = {}
    
    try:
        # Get all rows from the scorecard table
        info_rows = page.locator(".ds-w-full .ds-table tbody:first-of-type tr")
        row_count = info_rows.count()
        
        # Loop through all rows to find the toss information by content
        toss_row_index = -1
        for i in range(row_count):
            row_text = info_rows.nth(i).inner_text().strip()
            if "Toss" in row_text:
                toss_row_index = i
                break
                
        # If we found the toss row, extract toss info
        if toss_row_index >= 0:
            toss_row = info_rows.nth(toss_row_index)
            
            toss_text = toss_row.inner_text().strip()
            match_info['toss_full'] = toss_text
            
            # Parse "Toss Tribhuwan Army Club, elected to field first"
            if "elected to" in toss_text:
                match_info['toss_winner'] = toss_text.split(',')[0].replace('Toss', '').strip()
                match_info['toss_decision'] = 'field' if 'field' in toss_text else 'bat'
        
            # Venue is usually one row before toss
            if toss_row_index > 0:
                venue_row = info_rows.nth(toss_row_index - 1)
                venue_element = venue_row.locator("td")
                
                if venue_element.count() > 0:
                    venue_text = venue_element.inner_text().strip()
                    match_info['venue'] = venue_text
                    
                    # Try to get venue link if available
                    venue_link_element = venue_row.locator("a")
                    if venue_link_element.count() > 0:
                        venue_link = venue_link_element.get_attribute("href")
                        match_info['venue_link'] = venue_link
        
        # If we didn't find the toss row using the above method, fall back to alternative approaches
        if 'toss_full' not in match_info:
            # Try to find the toss info using a more specific selector
            toss_element = page.locator("td:has-text('Toss')").first
            if toss_element.count() > 0:
                parent_row = toss_element.evaluate("el => el.closest('tr')")
                if parent_row:
                    toss_text = parent_row.inner_text().strip()
                    match_info['toss_full'] = toss_text
                    
                    # Parse toss information
                    if "elected to" in toss_text:
                        match_info['toss_winner'] = toss_text.split(',')[0].replace('Toss', '').strip()
                        match_info['toss_decision'] = 'field' if 'field' in toss_text else 'bat'
        
        # Extract teams playing
        """ team_elements = page.locator(".ds-flex.ds-items-center.ds-min-w-0.ds-mr-1 p")
        if team_elements.count() >= 2:
            match_info['team1'] = team_elements.nth(0).inner_text().strip()
            match_info['team2'] = team_elements.nth(1).inner_text().strip() """
        
        return match_info
    
    except Exception as e:
        print(f"Error extracting match info: {e}")
        traceback.print_exc()
        return match_info

def extract_venue_stats(context, venue_id, timeout=30000):
    """
    Extract venue statistics with improved error handling and timeout protection
    
    Args:
        context: Playwright browser context
        venue_id: ID of the venue
        timeout: Page load timeout in milliseconds (default: 30s)
    """
    # Initialize default stats
    stats = {
        "venue_name": "Unknown",
        "matches_played": "N/A",
        "total_runs": "N/A",
        "total_wickets": "N/A",
        "balls_bowled": "N/A",
        "average_runs_per_wicket": "N/A",
        "average_runs_per_over": "N/A",
        "highest_total": "N/A",
        "lowest_total": "N/A",
        "highest_successful_chase": "N/A"
    }
    
    venue_page = None
    
    try:
        # Construct venue URL
        venue_url = f"https://stats.espncricinfo.com/ci/engine/ground/58827.html?class=6;template=results;type=aggregate"
        print(f"Fetching venue statistics from: {venue_url}")
        
        # Open new page with timeout protection
        venue_page = context.new_page()
        
        # Set a shorter timeout and handle the exception if it occurs
        try:
            venue_page.goto(venue_url, timeout=timeout)
        except Exception as timeout_error:
            print(f"Warning: Timeout when loading venue stats page ({timeout_error})")
            print("Continuing with default venue stats values")
            if venue_page:
                venue_page.close()
            return stats
            
        # Check if we got a valid page
        if "Page not found" in venue_page.title() or venue_page.url.endswith("error404.html"):
            print(f"Warning: Venue page not found for ID {venue_id}")
            venue_page.close()
            return stats
            
        # Extract basic venue name from title
        stats["venue_name"] = venue_page.title().replace(" Statistics", "").strip()
        
        # Try a simplified extraction approach
        try:
            # Look for matches played
            matches_element = venue_page.locator("td:has-text('Matches')").first
            if matches_element.count() > 0:
                parent_row = matches_element.locator("xpath=./parent::tr")
                if parent_row.count() > 0:
                    cells = parent_row.locator("td")
                    if cells.count() >= 2:
                        stats["matches_played"] = cells.nth(1).inner_text().strip()
            
            # Look for runs and wickets
            runs_element = venue_page.locator("td:has-text('Runs')").first
            if runs_element.count() > 0:
                parent_row = runs_element.locator("xpath=./parent::tr")
                if parent_row.count() > 0:
                    cells = parent_row.locator("td")
                    if cells.count() >= 2:
                        stats["total_runs"] = cells.nth(1).inner_text().strip()
                        
            wickets_element = venue_page.locator("td:has-text('Wickets')").first
            if wickets_element.count() > 0:
                parent_row = wickets_element.locator("xpath=./parent::tr")
                if parent_row.count() > 0:
                    cells = parent_row.locator("td")
                    if cells.count() >= 2:
                        stats["total_wickets"] = cells.nth(1).inner_text().strip()
                        
            # Extract highest and lowest scores
            highest_element = venue_page.locator("text=Highest total").first
            if highest_element.count() > 0:
                parent_row = highest_element.locator("xpath=./ancestor::tr")
                if parent_row.count() > 0:
                    cells = parent_row.locator("td")
                    if cells.count() >= 3:
                        stats["highest_total"] = cells.nth(2).inner_text().strip()
                        
            lowest_element = venue_page.locator("text=Lowest total").first 
            if lowest_element.count() > 0:
                parent_row = lowest_element.locator("xpath=./ancestor::tr")
                if parent_row.count() > 0:
                    cells = parent_row.locator("td")
                    if cells.count() >= 3:
                        stats["lowest_total"] = cells.nth(2).inner_text().strip()
                        
        except Exception as inner_error:
            print(f"Warning: Error extracting venue details: {inner_error}")
            # Continue with partial data
            
    except Exception as e:
        print(f"Error extracting venue stats: {e}")
        traceback.print_exc()
    finally:
        # Make sure we always close the page
        if venue_page:
            try:
                venue_page.close()
            except:
                pass
                
    return stats

class PlayerStatsCache:
    """
    Cache for player historical statistics to avoid redundant API calls
    """
    def __init__(self, browser_context=None):
        self.batting_stats = {}  # Player ID -> batting stats
        self.bowling_stats = {}  # Player ID -> bowling stats
        self.downloader = BattingHistoryDownloader()
        self.browser_context = browser_context  # Store the browser context
        
    def get_player_stats(self, player_id, player_name):
        """
        Get historical stats for a player, using cache if available
        """
        if not player_id:
            print(f"Warning: No player ID for {player_name}, cannot fetch stats")
            return None, None
            
        # Check cache first
        if player_id in self.batting_stats and player_id in self.bowling_stats:
            return self.batting_stats[player_id], self.bowling_stats[player_id]
            
        try:
            print(f"Fetching historical stats for {player_name} (ID: {player_id})")
            
            # Use Playwright if we have a browser context
            if self.browser_context:
                #batting_df, bowling_df = self.downloader.get_ipl_batting_stats_with_playwright(player_id, self.browser_context)
                batting_df, bowling_df = self.downloader.get_ipl_stats_from_csv(player_id=player_id, player_name=player_name)
            else:
                # Fallback to original method if no browser context available
                batting_df, bowling_df = self.downloader.get_ipl_batting_stats(player_id)
            
            # Process stats to get averages
            batting_stats = self.process_batting_stats(batting_df) if batting_df is not None else None
            bowling_stats = self.process_bowling_stats(bowling_df) if bowling_df is not None else None
            
            # Cache the results
            self.batting_stats[player_id] = batting_stats
            self.bowling_stats[player_id] = bowling_stats
            
            return batting_stats, bowling_stats
        except Exception as e:
            print(f"Error fetching stats for {player_name}: {e}")
            traceback.print_exc()
            return None, None
    
    def process_batting_stats(self, batting_df):
        """
        Process the batting dataframe to extract key metrics with improved error handling
        """
        try:
            if batting_df is None or batting_df.empty:
                return None
                
            # Try to get IPL stats if Format column exists
            if 'Format' in batting_df.columns:
                ipl_row = batting_df[batting_df['Format'] == 'IPL']
                if not ipl_row.empty:
                    row = ipl_row.iloc[0]
                else:
                    # If no IPL stats, use the first row
                    row = batting_df.iloc[0]
            else:
                # If no Format column, use the first row
                row = batting_df.iloc[0]
                
            # Safely get values with fallbacks
            return {
                'matches': row.get('Matches', 0) if isinstance(row, pd.Series) else 0,
                'innings': row.get('Innings', 0) if isinstance(row, pd.Series) else 0,
                'runs': row.get('Runs', 0) if isinstance(row, pd.Series) else 0,
                'average': row.get('Average', 0) if isinstance(row, pd.Series) else 0,
                'strike_rate': row.get('Strike Rate', 0) if isinstance(row, pd.Series) else 0,
                'fifties': row.get('50s', 0) if isinstance(row, pd.Series) else 0,
                'hundreds': row.get('100s', 0) if isinstance(row, pd.Series) else 0,
                'fours': row.get('Fours', 0) if isinstance(row, pd.Series) else 0,
                'sixes': row.get('Sixes', 0) if isinstance(row, pd.Series) else 0
            }
        except Exception as e:
            print(f"Error processing batting stats: {e}")
            return None

    def process_bowling_stats(self, bowling_df):
        """
        Process the bowling dataframe to extract key metrics with improved error handling
        """
        try:
            if bowling_df is None or bowling_df.empty:
                return None
                
            # Try to get IPL stats if Format column exists
            if 'Format' in bowling_df.columns:
                ipl_row = bowling_df[bowling_df['Format'] == 'IPL']
                if not ipl_row.empty:
                    row = ipl_row.iloc[0]
                else:
                    # If no IPL stats, use the first row
                    row = bowling_df.iloc[0]
            else:
                # If no Format column, use the first row
                row = bowling_df.iloc[0]
                
            # Safely get values with fallbacks
            return {
                'matches': row.get('Matches', 0) if isinstance(row, pd.Series) else 0,
                'innings': row.get('Innings', 0) if isinstance(row, pd.Series) else 0,
                'wickets': row.get('Wickets', 0) if isinstance(row, pd.Series) else 0,
                'average': row.get('Average', 0) if isinstance(row, pd.Series) else 0,
                'economy': row.get('Economy', 0) if isinstance(row, pd.Series) else 0,
                'strike_rate': row.get('Strike Rate', 0) if isinstance(row, pd.Series) else 0,
                'best_bowling': row.get('BBI', '0/0') if isinstance(row, pd.Series) else '0/0'
            }
        except Exception as e:
            print(f"Error processing bowling stats: {e}")
            return None

def enhance_ball_data_with_historical_stats(ball_data, player_stats_cache):
    """
    Add historical statistics to ball data for batsmen and bowlers
    Returns: Enhanced ball data with historical player statistics
    """
    enhanced_data = ball_data.copy()
    
    # Process batsmen stats
    batsman_fields = {
        'batsman1': {'name': 'batsman1_name', 'id': 'batsman1_player_id'},
        'batsman2': {'name': 'batsman2_name', 'id': 'batsman2_player_id'}
    }
    
    for batsman_key, fields in batsman_fields.items():
        if fields['name'] in ball_data and fields['id'] in ball_data:
            player_name = ball_data[fields['name']]
            player_id = ball_data[fields['id']]
            
            if player_id and player_stats_cache:
                batting_stats, _ = player_stats_cache.get_player_stats(player_id, player_name)
                if batting_stats:
                    prefix = f"{batsman_key}_historical_"
                    enhanced_data.update({
                        f"{prefix}matches": batting_stats.get('matches', 0),
                        f"{prefix}innings": batting_stats.get('innings', 0),
                        f"{prefix}runs": batting_stats.get('runs', 0),
                        f"{prefix}average": batting_stats.get('average', 0),
                        f"{prefix}strike_rate": batting_stats.get('strike_rate', 0),
                        f"{prefix}fifties": batting_stats.get('fifties', 0),
                        f"{prefix}hundreds": batting_stats.get('hundreds', 0),
                        f"{prefix}fours": batting_stats.get('fours', 0),
                        f"{prefix}sixes": batting_stats.get('sixes', 0)
                    })
    
    # Process bowlers stats
    bowler_fields = {
        'bowler1': {'name': 'bowler1_name', 'id': 'bowler1_player_id'},
        'bowler2': {'name': 'bowler2_name', 'id': 'bowler2_player_id'}
    }
    
    for bowler_key, fields in bowler_fields.items():
        if fields['name'] in ball_data and fields['id'] in ball_data:  # Check for both name and ID
            player_name = ball_data[fields['name']]
            player_id = ball_data[fields['id']]  # Get player ID from ball_data
            
            if player_id and player_stats_cache:
                _, bowling_stats = player_stats_cache.get_player_stats(player_id, player_name)
                if bowling_stats:
                    prefix = f"{bowler_key}_historical_"
                    enhanced_data.update({
                        f"{prefix}matches": bowling_stats.get('matches', 0),
                        f"{prefix}innings": bowling_stats.get('innings', 0),
                        f"{prefix}wickets": bowling_stats.get('wickets', 0),
                        f"{prefix}average": bowling_stats.get('average', 0),
                        f"{prefix}economy": bowling_stats.get('economy', 0),
                        f"{prefix}strike_rate": bowling_stats.get('strike_rate', 0),
                        f"{prefix}best_bowling": bowling_stats.get('best_bowling', '0/0')
                    })
    
    return enhanced_data

def extract_player_id_from_profile_url(url):
    """
    Extract player ID and slug from profile URL
    Example URL: "/cricketers/virat-kohli-253802"
    Returns: The full slug (e.g., 'virat-kohli-253802')
    """
    if not url:
        return None
        
    # Try to match the full slug pattern after /cricketers/
    match = re.search(r'/cricketers/([^/]+)', url)
    if match:
        return match.group(1)  # Returns 'virat-kohli-253802'
    return None

def extract_win_probability(page):
    """
    Extract win probability information from the page with robust fallback methods.
    Returns: tuple of (favored_team, win_percentage)
    """
    try:
        # Method 1: Try the original selector
        win_prob_element = page.locator(".ds-flex.ds-items-center.ds-bg-fill-content-alternate.lg\\:ds-p-0")
        
        if win_prob_element.count() > 0:
            # Get the inner text
            win_prob_text = win_prob_element.inner_text().strip()
            print(f"Method 1 - Raw win probability text: {win_prob_text}")
            
            # Example text: 'Win Probability: PBKS 3.66% • MI 96.34%'
            if "Win Probability" in win_prob_text:
                # Extract team names and probabilities
                parts = win_prob_text.replace("Win Probability:", "").strip().split("•")
                
                if len(parts) == 2:
                    # Process first team
                    team1_parts = parts[0].strip().split()
                    team1 = team1_parts[0].strip()
                    team1_prob = float(team1_parts[1].replace("%", "").strip())
                    
                    # Process second team
                    team2_parts = parts[1].strip().split()
                    team2 = team2_parts[0].strip()
                    team2_prob = float(team2_parts[1].replace("%", "").strip())
                    
                    print(f"Extracted probabilities: {team1}: {team1_prob}%, {team2}: {team2_prob}%")
                    
                    # Determine favored team (>50% probability)
                    if team1_prob > team2_prob:
                        return team1, team1_prob
                    else:
                        return team2, team2_prob
        
        # Method 2: Try alternative selector for newer HTML format
        print("Trying alternative win probability selector...")
        win_prob_title = page.locator("span.ds-text-title-xs.ds-font-bold.ds-uppercase:has-text('Win Probability')")
        
        if win_prob_title.count() > 0:
            # Find the parent container
            parent_div = win_prob_title.evaluate("el => el.closest('.ds-flex')")
            
            if parent_div:
                # Execute JavaScript to extract the win probability info
                result = page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('.ds-text-tight-s.ds-font-bold.ds-ml-1');
                    for (const el of elements) {
                        if (el.textContent.includes('%')) {
                            return el.textContent.trim();
                        }
                    }
                    return null;
                }
                """)
                
                if result:
                    print(f"Method 2 - Raw win probability text: {result}")
                    # Extract team and percentage (e.g., "RR 76.95%")
                    import re
                    match = re.search(r'([A-Z]+).*?(\d+\.\d+)%', result)
                    if match:
                        team = match.group(1).strip()
                        percentage = float(match.group(2))
                        print(f"Extracted probability: {team}: {percentage}%")
                        return team, percentage
        
        # Method 3: Use a broader selector and look for percentage pattern
        print("Trying broad selector for win probability...")
        percentage_elements = page.locator(".ds-text-tight-s:has-text('%')")
        
        for i in range(percentage_elements.count()):
            element_text = percentage_elements.nth(i).inner_text().strip()
            # Look for text with team code and percentage
            import re
            match = re.search(r'([A-Z]{2,4})[^\d]*(\d+\.\d+)%', element_text)
            if match:
                team = match.group(1).strip()
                percentage = float(match.group(2))
                print(f"Method 3 - Extracted probability: {team}: {percentage}%")
                return team, percentage
        
        print("Win probability not found using any method")
        return None, None
            
    except Exception as e:
        print(f"Error extracting win probability: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_current_batting_team(page):
    """
    Extract the current batting team by finding the team with the 'icon-dot_circular' indicator
    Returns: dictionary with batting team information, handling first innings scenarios
    """
    try:
        # Find all team containers
        team_containers = page.locator(".ci-team-score.ds-flex.ds-justify-between.ds-items-center.ds-text-typo.ds-mb-1")
        
        batting_team_name = None
        batting_team_score = None
        target_score = None
        is_second_innings = False
        overs_info = None
        bowling_team_name = None
        
        # Check both team containers
        for i in range(team_containers.count()):
            team_container = team_containers.nth(i)
            
            # Check if this team has the batting indicator
            batting_indicator = team_container.locator("i.icon-dot_circular")
            
            # Get team name
            team_name_element = team_container.locator(".ds-text-tight-l")
            if team_name_element.count() == 0:
                continue
                
            team_name = team_name_element.inner_text().strip()
            
            # Get team score
            score_element = team_container.locator("strong")
            
            # Get overs info if present
            overs_element = team_container.locator(".ds-text-compact-s")
            
            # If this team has the batting indicator, it's the batting team
            if batting_indicator.count() > 0:
                batting_team_name = team_name
                if score_element.count() > 0:
                    batting_team_score = score_element.inner_text().strip()
                
                # Get over information
                if overs_element.count() > 0:
                    overs_text = overs_element.inner_text().strip()
                    overs_info = overs_text
                    
                    # Check if this is second innings with target
                    if "T:" in overs_text:
                        is_second_innings = True
                        target_match = re.search(r'T:(\d+)', overs_text)
                        if target_match:
                            target_score = int(target_match.group(1))
            else:
                # This is the bowling team in first innings (or waiting team in second)
                bowling_team_name = team_name
                
                # In a first innings, the other team won't have a score yet
                # In a second innings, the other team's score is the first innings total
                if score_element.count() > 0:
                    other_team_score = score_element.inner_text().strip()
                    
                    # If we see a score for the other team but no "T:" in overs,
                    # this is likely a second innings without explicit target formatting
                    if not is_second_innings and other_team_score:
                        is_second_innings = False  # Still first innings
        
        # Extract numeric score and wickets from batting_team_score (e.g. "68/1" -> 68 runs, 1 wicket)
        runs = 0
        wickets = 0
        if batting_team_score and '/' in batting_team_score:
            score_parts = batting_team_score.split('/')
            if len(score_parts) == 2:
                runs = int(score_parts[0])
                wickets = int(score_parts[1])
        
        # Parse overs from overs info (e.g., "(6.6/20 ov)" -> 6.6)
        current_over = 0
        if overs_info:
            over_match = re.search(r'\((\d+\.\d+)/\d+', overs_info)
            if over_match:
                current_over = float(over_match.group(1))
        
        # First innings has no target
        if not is_second_innings:
            target_score = None
        
        forecast_score = None
        forecast_team = None    
        print(f"Current batting team: {batting_team_name} ({batting_team_score}), over: {current_over}")
        print(f"Bowling team: {bowling_team_name}")
        if is_second_innings:
            print(f"Second innings - Target: {target_score}")     
        else:
            print("First innings - Setting target")
            # Extract projected score from live forecast element
            try:
                projected_score_element = page.locator("div.ds-flex.ds-items-center.ds-bg-fill-content-alternate:has(span.ds-font-medium:has-text('Live Forecast'))")
                if projected_score_element.count() > 0:
                    forecast_text = projected_score_element.inner_text().strip()
                    print(f"Raw forecast text: {forecast_text}")
                    
                    # Extract team abbreviation and projected score
                    # Format: "Live Forecast: MI 146"
                    match = re.search(r'Live Forecast:?\s*([A-Z]+)\s*(\d+)', forecast_text)
                    if match:
                        forecast_team = match.group(1)
                        forecast_score = int(match.group(2))
                        print(f"📊 Extracted live forecast: {forecast_team} projected to score {forecast_score}")
                    else:
                        print("Could not parse forecast score from text")
            except Exception as e:
                print(f"Error extracting forecast score: {e}")
                traceback.print_exc()
            
            
        return {
            'batting_team': batting_team_name,
            'bowling_team': bowling_team_name,
            'runs': runs,
            'wickets': wickets,
            'target': target_score,
            'is_second_innings': is_second_innings,
            'overs_info': overs_info,
            'current_over': current_over,
            'forecast_score': forecast_score,
            'forcast_team': forecast_team,
        }
        
    except Exception as e:
        print(f"Error extracting current batting team: {e}")
        traceback.print_exc()
        return {
            'batting_team': None,
            'bowling_team': None,
            'runs': 0,
            'wickets': 0,
            'target': None,
            'is_second_innings': False,
            'overs_info': None,
            'current_over': 0
        }

def extract_run_rates(page):
    """
    Extract current run rate (CRR) and required run rate (RRR) from the dedicated stats element
    Returns: Dictionary with run rate information
    """
    try:
        # Target the run rate container - the element with CRR and RRR information
        run_rate_element = page.locator(".ds-text-tight-s.ds-font-regular.ds-overflow-x-auto.ds-scrollbar-hide.ds-whitespace-nowrap.ds-mt-1")
        
        if (run_rate_element.count() == 0):
            print("Run rate element not found")
            return {}
        
        # Get the inner text
        run_rate_text = run_rate_element.inner_text().strip()
        print(f"Raw run rate text: {run_rate_text}")
        
        # Initialize results
        run_rates = {}
        
        # Extract current run rate using regex
        crr_match = re.search(r'Current RR:?\s*(\d+\.\d+)', run_rate_text)
        if crr_match:
            run_rates['current_run_rate'] = float(crr_match.group(1))
        
        # Extract required run rate if available (second innings only)
        rrr_match = re.search(r'Required RR:?\s*(\d+\.\d+)', run_rate_text)
        if rrr_match:
            run_rates['required_run_rate'] = float(rrr_match.group(1))
        
        # Extract last 5 overs stats
        last_5_match = re.search(r'Last 5 ov.*?(\d+)/(\d+)\s*\(\s*(\d+\.\d+)\s*\)', run_rate_text)
        if last_5_match:
            run_rates['last_5_overs_runs'] = int(last_5_match.group(1))
            run_rates['last_5_overs_wickets'] = int(last_5_match.group(2))
            run_rates['last_5_overs_run_rate'] = float(last_5_match.group(3))
            
        print(f"Extracted run rates: {run_rates}")
        return run_rates
            
    except Exception as e:
        print(f"Error extracting run rates: {e}")
        traceback.print_exc()
        return {}

def calculate_rolling_averages(final_data, match_state):
    """
    Calculate rolling averages based on over_summaries in match_state.
    Uses direct over_runs and over_wickets values for team stats.
    For batsmen, uses cumulative total differences between overs.
    
    Args:
        final_data: The current ball data dictionary
        match_state: Match state containing over_summaries
        
    Returns:
        Dictionary with rolling average statistics
    """
    from fuzzywuzzy import fuzz
    
    rolling_stats = {}
    
    # Get current over number and add 1 for current over 
    current_over = final_data.get('over_number') + 1
    if not current_over:
        return rolling_stats
    
    # Check if we have over_summaries
    if 'over_summaries' not in match_state or not match_state['over_summaries']:
        return rolling_stats
    
    # Get available overs (as integers) for which we have data, skipping None entries
    #sort in reverse order 
    available_overs = sorted([int(over) for over in match_state['over_summaries'].keys()
                              if match_state['over_summaries'][over] is not None], reverse=True)
    
    # Get the last 3 completed overs (i.e. overs less than the current over)
    recent_overs = [over for over in available_overs if over < current_over][-3:]
    
    if not recent_overs:
        return rolling_stats
    
    print(f"Calculating rolling averages using overs: {recent_overs}")
    
    # Lists to store values for averaging team stats
    runs_per_over = []
    wickets_per_over = []
    
    # Process team stats - continue using direct over_runs and over_wickets
    for over in recent_overs:
        over_str = str(over)
        summary = match_state['over_summaries'].get(over_str) or match_state['over_summaries'].get(over)
        
        if summary:
            # Directly use over_runs and over_wickets
            if 'over_runs' in summary:
                runs_per_over.append(summary['over_runs'])
                
            if 'over_wickets' in summary:
                wickets_per_over.append(summary['over_wickets'])
    
    # Calculate team averages
    if runs_per_over:
        rolling_stats['runs_scored_over_rolling_avg'] = round(sum(runs_per_over) / len(runs_per_over), 2)
    else:
        rolling_stats['runs_scored_over_rolling_avg'] = 0
    
    if wickets_per_over:
        rolling_stats['wickets_over_rolling_avg'] = round(sum(wickets_per_over) / len(wickets_per_over), 2)
    else:
        rolling_stats['wickets_over_rolling_avg'] = 0
    
    # Get current batsmen names
    current_batsman1 = final_data.get('batsman1_name', '')
    current_batsman2 = final_data.get('batsman2_name', '')
    
    # For each batsman, get their cumulative runs at the earliest and latest overs
    """ if current_batsman1:
        batsman1_runs_in_period = calculate_batsman_runs_in_period(current_batsman1, recent_overs, match_state)
        if batsman1_runs_in_period is not None:
            rolling_stats['batsman1_runs_over_rolling_avg'] = round(batsman1_runs_in_period / len(recent_overs), 2)
    
    if current_batsman2:
        batsman2_runs_in_period = calculate_batsman_runs_in_period(current_batsman2, recent_overs, match_state)
        if batsman2_runs_in_period is not None:
            rolling_stats['batsman2_runs_over_rolling_avg'] = round(batsman2_runs_in_period / len(recent_overs), 2) """
    
    print(f"Team runs per over: {runs_per_over}, wickets per over: {wickets_per_over}")
    print(f"Calculated rolling stats: {rolling_stats}")
    # update the final data with rolling stats
    final_data.update(rolling_stats)
    return rolling_stats

def calculate_batsman_runs_in_period(batsman_name, overs, match_state):
    """
    Calculate how many runs a batsman scored across the given overs
    by using cumulative totals at the end of each over.
    
    Args:
        batsman_name: Name of the batsman
        overs: List of over numbers to consider (sorted)
        match_state: Match state containing over summaries
        
    Returns:
        Total runs scored in the period or None if data not available
    """
    from fuzzywuzzy import fuzz

    
    # Find the batsman in each over's summary and track their cumulative scores
    batsman_runs_by_over = {}
    
    for over in overs:
        over_str = str(over)
        summary = match_state['over_summaries'].get(over_str) or match_state['over_summaries'].get(over)
        if not summary:
            continue
            
        # Check both batsmen positions
        for position in ['batsman1', 'batsman2']:
            position_name = summary.get(f'{position}_name', '')
            if position_name and fuzz.ratio(batsman_name.lower(), position_name.lower()) >= 80:
                batsman_runs_by_over[over] = summary.get(f'{position}_runs', 0)
                break
    
    # If we don't have data for the batsman in any over, return None
    if not batsman_runs_by_over:
        return None
    
    # If we only have data for one over, just use that as the average
    if len(batsman_runs_by_over) == 1:
        over = list(batsman_runs_by_over.keys())[0]
        return batsman_runs_by_over[over] / len(overs)  # Divide by total overs for the average
    
    # Find the earliest and latest overs where we have data for this batsman
    overs_with_data = sorted(batsman_runs_by_over.keys())
    earliest_over = overs_with_data[0]
    latest_over = overs_with_data[-1]
    
    # Calculate the total runs scored between these overs
    runs_in_period = batsman_runs_by_over[latest_over] - batsman_runs_by_over[earliest_over]
    
    # Adjust if we don't have data for the first over (the batsman might have just come in)
    #equls means he has ust come in 
    if earliest_over == overs[0]:
        # This is a new batsman, so all their runs are in this period
        runs_in_period = batsman_runs_by_over[latest_over]
    
    return max(0, runs_in_period)  # Ensure we don't return negative runs

def main(url=None):
    """Main function to run the scraper and process the data."""
    
    import os
    import re
    if not url:
        # Default URL if none is provided
        url = "https://www.espncricinfo.com/series/ipl-2025-1449924/gujarat-titans-vs-mumbai-indians-9th-match-1473446/live-cricket-score"
        
    
    # Convert scorecard URL to ball-by-ball URL if needed
    if "full-scorecard" in url:
        url = url.replace("full-scorecard", "live-cricket-score")
    
        
    # Get match info and create directory structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    season_folder, match_id = get_match_info_from_url(url)
    match_dir = os.path.join(current_dir, season_folder, match_id, "ball_by_ball")
    ensure_directory_exists(match_dir)
    
    try:
        # First, import and run the ball-by-ball scraper to create innings summary files
        print("Running ball-by-ball scraper first...")
        # from espnscraper_ballbyball import main as run_ballbyball
        # run_ballbyball(url)
        
        # Initialize player stats cache
        
        #wait for new ball update
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                timezone_id="Asia/Kolkata",  # India timezone
                accept_downloads=True
            )
            
            context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
            """)
            page = context.new_page()

            player_stats_cache = PlayerStatsCache(browser_context=context)
            # Enable console logs from the browser
            #page.on("console", lambda msg: print(f"BROWSER LOG: {msg.text}"))

            # Navigate to the commentary page
            use_local = False  # Set to True if using local MHTML file
            #use_local = False  # Set to True if using local MHTML file
            if use_local:
                # Load local MHTML file
                mhtml_path = "file:///C:/Users/admn/Documents/GT70_0.mhtml"
                page.goto(mhtml_path, timeout=60000)
            else:
                # Use live URL
                page.goto(url, timeout=60000)
            time.sleep(5)  # Increased wait time after page load
            
            # Track match state with over summary cache
            match_state = {
                'current_innings': 1,
                'current_over': 0,
                'total_score': 0,
                'total_wickets': 0,
                'over_summaries': {},  # Cache for end-of-over summaries
                'player_stats': {}     # Cache for player historical stats
            }
            
            match_info = extract_match_info(page)
            print(f"Match Info: {match_info}")
            
            # Now extract venue stats if venue link is available
            if 'venue_link' in match_info and match_info['venue_link']:
                # Extract venue ID from the link - handle different formats
                venue_id = None
                venue_id_match = re.search(r'cricket-grounds/([^/]+)(?:-(\d+))?', match_info['venue_link'])
                
                if venue_id_match:
                    # Some links have ID at the end, others use a name-ID format
                    if venue_id_match.group(2):  # If we have a second group with just the ID
                        venue_id = venue_id_match.group(2)
                    elif venue_id_match.group(1) and venue_id_match.group(1).isdigit():  # If first group is numeric
                        venue_id = venue_id_match.group(1)
                        
                if not venue_id:
                    # Alternative pattern for direct numeric IDs
                    alt_match = re.search(r'(\d+)$', match_info['venue_link'])
                    venue_id = alt_match.group(1) if alt_match else None
                    
                print(f"Venue: {match_info['venue_link']}, Extracted ID: {venue_id}")
                
                if venue_id:
                    try:
                        # Use lower timeout and custom extraction with robust error handling
                        venue_stats = get_venue_stats(context, venue_id)
                        match_state['venue_stats'] = venue_stats
                        print(f"Venue statistics extracted: {venue_stats['matches_played']} matches played")
                    except Exception as venue_error:
                        print(f"Error getting venue stats: {venue_error}")
                        match_state['venue_stats'] = {}
                else:
                    print("Could not extract venue ID from link")
                    match_state['venue_stats'] = {}
            else:
                print("No venue link found in match info, skipping venue stats")
                match_state['venue_stats'] = {}
            
            while True:
                try:
                    # Get the next ball update
                    ball_update = wait_for_new_ball_update(page, match_state)
                    
                    # Extract the latest scorecard data immediately after a new ball
                    print("Fetching current scorecard data...")
                    scorecard_data = extract_scorecard_data(page, player_stats_cache)
                    
                    # Parse the ball details with the structured data
                    ball_data = parse_ball_details(
                        ball_update['ball_number'],
                        ball_update['runs_or_event'], 
                        ball_update['short_commentary']
                    )

            
                    enhanced_ball_data = {}  # Initialize here, before any access attempts
                    enhanced_ball_data.update(ball_data)

                    # Process end-of-over data if available
                    if 'over_info' in ball_update and ball_update['over_info']:
                        pass # Placeholder for existing over data processing code
                    
                    # Enhance ball_data with current player information from scorecard
                    if scorecard_data:
                        # Find batsmen on strike
                        for batsman in scorecard_data.get('batsmen', []):
                            if batsman.get('is_on_strike'):
                                ball_data['batsman1_name'] = batsman['name']
                                ball_data['batsman1_runs'] = batsman['runs']
                                ball_data['batsman1_balls_faced'] = batsman['balls']
                                ball_data['batsman1_fours'] = batsman['fours']
                                ball_data['batsman1_sixes'] = batsman['sixes']
                                # Calculate strike rate on the fly if needed
                                ball_data['batsman1_strike_rate'] = batsman['strike_rate']
                                
                                # Store player ID for historical stats lookup
                                ball_data['batsman1_player_id'] = batsman.get('player_id', '')
                            else:
                                # Second batsman at crease
                                ball_data['batsman2_name'] = batsman['name']
                                ball_data['batsman2_runs'] = batsman['runs']
                                ball_data['batsman2_balls_faced'] = batsman['balls']
                                ball_data['batsman2_fours'] = batsman['fours']
                                ball_data['batsman2_sixes'] = batsman['sixes']
                                ball_data['batsman2_strike_rate'] = batsman['strike_rate']
                                ball_data['batsman2_player_id'] = batsman.get('player_id', '')
                        
                        # Get current bowler info - usually the first one in the list
                        if scorecard_data.get('bowlers') and len(scorecard_data['bowlers']) > 0:
                            current_bowler = scorecard_data['bowlers'][0]
                            ball_data['bowler1_name'] = current_bowler['name']
                            ball_data['bowler1_overs_bowled'] = current_bowler['overs']
                            ball_data['bowler1_maidens_bowled'] = current_bowler['maidens']
                            ball_data['bowler1_runs_conceded'] = current_bowler['runs']
                            ball_data['bowler1_wickets_taken'] = current_bowler['wickets']
                            ball_data['bowler1_economy'] = current_bowler['economy']
                            
                            # Store player ID for historical stats
                            ball_data['bowler1_player_id'] = current_bowler.get('player_id', '')
                            
                            # If there's a second bowler, add their information too
                            if len(scorecard_data['bowlers']) > 1:
                                second_bowler = scorecard_data['bowlers'][1]
                                ball_data['bowler2_name'] = second_bowler['name']
                                ball_data['bowler2_overs_bowled'] = second_bowler['overs']
                                ball_data['bowler2_maidens_bowled'] = second_bowler['maidens']
                                ball_data['bowler2_runs_conceded'] = second_bowler['runs']
                                ball_data['bowler2_wickets_taken'] = second_bowler['wickets']
                                ball_data['bowler2_economy'] = second_bowler['economy']
                                ball_data['bowler2_player_id'] = second_bowler.get('player_id', '')
                    else:
                        print("Warning: Could not extract scorecard data")
                    
                    if 'current_run_rate' in ball_update and ball_update['current_run_rate'] is not None:
                        enhanced_ball_data['current_run_rate'] = ball_update['current_run_rate']

                    if 'required_run_rate' in ball_update and ball_update['required_run_rate'] is not None:
                        enhanced_ball_data['required_run_rate'] = ball_update['required_run_rate']

                    # Add last 5 overs information
                    if 'last_5_overs_runs' in ball_update and ball_update['last_5_overs_runs'] is not None:
                        enhanced_ball_data['last_5_overs_runs'] = ball_update['last_5_overs_runs']
                        enhanced_ball_data['last_5_overs_wickets'] = ball_update['last_5_overs_wickets']
                        enhanced_ball_data['last_5_overs_run_rate'] = ball_update['last_5_overs_run_rate']
                        
    
                    # Now enhance ball data with historical player statistics
                    enhanced_ball_data = enhance_ball_data_with_historical_stats(ball_data, player_stats_cache)
                        
                    # Add match info to ball data
                    enhanced_ball_data['venue'] = match_info.get('venue', '')
                    enhanced_ball_data['venue_link'] = match_info.get('venue_link', '')
                    enhanced_ball_data['toss_winner'] = match_info.get('toss_winner', '')
                    enhanced_ball_data['toss_decision'] = match_info.get('toss_decision', '')
                    
                    # Add venue information to ball_data
                    enhanced_ball_data['venue'] = match_info.get('venue', '')
                    enhanced_ball_data['venue_link'] = match_info.get('venue_link', '')

                    # Add detailed venue statistics
                    if 'venue_stats' in match_state and match_state['venue_stats']:
                        venue_stats = match_state['venue_stats']
                        enhanced_ball_data['venue_matches_played'] = venue_stats.get('matches_played', 'N/A')
                        enhanced_ball_data['venue_total_runs'] = venue_stats.get('total_runs', 'N/A')
                        enhanced_ball_data['venue_total_wickets'] = venue_stats.get('total_wickets', 'N/A') 
                        enhanced_ball_data['venue_avg_runs_per_wicket'] = venue_stats.get('average_runs_per_wicket', 'N/A')
                        enhanced_ball_data['venue_avg_runs_per_over'] = venue_stats.get('average_runs_per_over', 'N/A')
                        enhanced_ball_data['venue_highest_total'] = venue_stats.get('highest_total', 'N/A')
                    
                    # Now ball_data contains both ball commentary, current player stats, and historical player stats
                    print(f"Enhanced Ball Data with Historical Stats: {enhanced_ball_data}")

                    # Add win probability information
                    if 'favored_team' in ball_update and ball_update['favored_team']:
                        enhanced_ball_data['favored_team'] = ball_update['favored_team']
                        enhanced_ball_data['win_percentage'] = ball_update['win_percentage']
                    
                    # Add batting team information - refactored to handle first innings
                    if 'batting_team' in ball_update and ball_update['batting_team']:
                        enhanced_ball_data['batting_team'] = ball_update['batting_team']
                        enhanced_ball_data['bowling_team'] = ball_update['bowling_team']
                        enhanced_ball_data['batting_team_score'] = ball_update['batting_team_score']
                        enhanced_ball_data['is_second_innings'] = ball_update['is_second_innings']
                        enhanced_ball_data['overs_info'] = ball_update['overs_info']
                        enhanced_ball_data['current_over'] = ball_update['current_over']
                        
                        # Add first innings specific metrics
                        if not ball_update['is_second_innings']:
                            # Calculate projected score for first innings
                            if 'current_over' in ball_update and ball_update['current_over'] > 0:
                                try:
                                    ball_update['runs'] = int(enhanced_ball_data['batting_team_score'].split('/')[0])
                                except (ValueError, IndexError, AttributeError):
                                    ball_update['runs'] = 0  # Default to 0 if parsing fails
                                current_run_rate = ball_update['runs'] / ball_update['current_over']
                                remaining_overs = 20 - ball_update['current_over']
                                # Extract runs from batting_team_score (e.g. "204/5" -> 204)
                                projected_score = ball_update['runs'] + (current_run_rate * remaining_overs)
                                enhanced_ball_data['projected_score'] = round(projected_score)
                                enhanced_ball_data['current_run_rate'] = round(current_run_rate, 2)
                        
                        # Add second innings specific metrics
                        else:
                            # Add target and required runs only for second innings
                            target = ball_update.get('target')
                            if target:
                                enhanced_ball_data['target_score'] = target
                                current_runs = ball_update.get('batting_team_score', '0/0').split('/')[0]
                                try:
                                    current_runs = int(current_runs)
                                    enhanced_ball_data['runs_needed'] = target - current_runs
                                    
                                    # Calculate remaining balls
                                    if 'over_info' in ball_update and ball_update['over_info']:
                                        over_num = ball_update['over_info']['over_number']
                                        ball_num = ball_update['over_info']['ball_number']
                                        total_overs = 20  # Assuming T20 match
                                        balls_completed = over_num * 6 + ball_num
                                        balls_remaining = total_overs * 6 - balls_completed
                                        enhanced_ball_data['balls_remaining'] = balls_remaining
                                        
                                        # Calculate required run rate
                                        if balls_remaining > 0:
                                            enhanced_ball_data['required_run_rate'] = round((enhanced_ball_data['runs_needed'] * 6) / balls_remaining, 2)
                                except (ValueError, KeyError) as e:
                                    print(f"Error calculating runs needed: {e}")
                                    enhanced_ball_data['runs_needed'] = None
                                    enhanced_ball_data['required_run_rate'] = None
                    
                    # Add batting team information with safety checks
                    enhanced_ball_data['batting_team'] = ball_update.get('batting_team', 'Unknown')
                    enhanced_ball_data['bowling_team'] = ball_update.get('bowling_team', 'Unknown')
                    enhanced_ball_data['batting_team_score'] = ball_update.get('batting_team_score', '0/0')
                    enhanced_ball_data['is_second_innings'] = ball_update.get('is_second_innings', False)
                    enhanced_ball_data['overs_info'] = ball_update.get('overs_info', '')
                    enhanced_ball_data['current_over'] = ball_update.get('current_over', 0)

                    # Add first innings specific metrics 
                    if not ball_update.get('is_second_innings', False):
                       
                        enhanced_ball_data['current_run_rate'] = ball_update.get('current_run_rate', 0)

                    # Add second innings specific metrics
                    else:
                        enhanced_ball_data['current_run_rate'] = ball_update.get('current_run_rate', 0)
                        enhanced_ball_data['required_run_rate_scraped'] = ball_update.get('required_run_rate', 0)
                        # Add last 5 overs stats
                        enhanced_ball_data['last_5_overs_runs'] = ball_update.get('last_5_overs_runs', 0)
                        enhanced_ball_data['last_5_overs_wickets'] = ball_update.get('last_5_overs_wickets', 0)
                        enhanced_ball_data['last_5_overs_run_rate'] = ball_update.get('last_5_overs_run_rate', 0)
                        # Add target and required runs only for second innings
                        target = ball_update.get('target')
                        if target:
                            enhanced_ball_data['target_score'] = target
                            #'batting_team_score': '40/4', get batting team score
                            current_runs = int(ball_update.get('batting_team_score', '0/0').split('/')[0])
                            enhanced_ball_data['runs_needed'] = target - current_runs
                            
                            # Calculate remaining balls
                            if 'over_info' in ball_update and ball_update['over_info']:
                                over_num = ball_update['over_info']['over_number']
                                ball_num = ball_update['over_info']['ball_number']
                                total_overs = 20  # Assuming T20 match
                                balls_completed = over_num * 6 + ball_num
                                balls_remaining = total_overs * 6 - balls_completed
                                enhanced_ball_data['balls_remaining'] = balls_remaining
                                
                                # Calculate required run rate
                                if balls_remaining > 0:
                                    enhanced_ball_data['required_run_rate'] = round((enhanced_ball_data.get('runs_needed', 0) * 6) / balls_remaining, 2)
                    
                    print(f"Ball Data: {enhanced_ball_data}")    
                    
                    final_data = {
                        'innings_num': 2 if ball_update['is_second_innings'] else 1,
                        'batting_team': fuzzy_match_team(enhanced_ball_data['batting_team']),
                        'over_number': enhanced_ball_data['over_number'],
                        'ball_number': enhanced_ball_data['ball_in_over'],
                        'total_score': int(enhanced_ball_data['batting_team_score'].split('/')[0]),
                        'total_wickets': int(enhanced_ball_data['batting_team_score'].split('/')[1]),
                        'boundaries': 1 if enhanced_ball_data.get('is_boundary', False) else 0,
                        'dot_balls': 1 if enhanced_ball_data.get('is_dot', False) else 0,
                        'wickets': 1 if enhanced_ball_data.get('is_wicket', False) else 0,
                        'extras': 1 if enhanced_ball_data.get('is_extra', False) else 0,
                        'favored_team': enhanced_ball_data.get('favored_team', ''),
                        'win_percentage': enhanced_ball_data.get('win_percentage', 0),
                        'current_run_rate': enhanced_ball_data.get('current_run_rate', 0),
                        'required_run_rate': enhanced_ball_data.get('required_run_rate', 0) if ball_update['is_second_innings'] else 0,
                        'striker_batsman': enhanced_ball_data.get('batsman1_name', ''),
                        'toss_winner': fuzzy_match_team(enhanced_ball_data.get('toss_winner', '')),
                        'toss_decision': enhanced_ball_data.get('toss_decision', '') + ' first',
                        
                        # Use .get() with defaults for all player fields
                        'batsman1_name': enhanced_ball_data.get('batsman1_name', 'Unknown'),
                        'batsman1_runs': enhanced_ball_data.get('batsman1_runs', 0),
                        'batsman1_balls_faced': enhanced_ball_data.get('batsman1_balls_faced', 0),
                        'batsman1_fours': enhanced_ball_data.get('batsman1_fours', 0),
                        'batsman1_sixes': enhanced_ball_data.get('batsman1_sixes', 0),
                        
                        # Add defaults for missing batsman2 fields
                        'batsman2_name': enhanced_ball_data.get('batsman2_name', 'Unknown'),
                        'batsman2_runs': enhanced_ball_data.get('batsman2_runs', 0),
                        'batsman2_balls_faced': enhanced_ball_data.get('batsman2_balls_faced', 0),
                        'batsman2_fours': enhanced_ball_data.get('batsman2_fours', 0),
                        'batsman2_sixes': enhanced_ball_data.get('batsman2_sixes', 0),
                        
                        # Add defaults for missing bowler fields
                        'bowler1_name': enhanced_ball_data.get('bowler1_name', 'Unknown'),
                        'bowler1_overs_bowled': enhanced_ball_data.get('bowler1_overs_bowled', 0),
                        'bowler1_maidens_bowled': enhanced_ball_data.get('bowler1_maidens_bowled', 0),
                        'bowler1_runs_conceded': enhanced_ball_data.get('bowler1_runs_conceded', 0),
                        'bowler1_wickets_taken': enhanced_ball_data.get('bowler1_wickets_taken', 0),
                        'bowler2_name': enhanced_ball_data.get('bowler2_name', 'Unknown'),
                        'bowler2_overs_bowled': enhanced_ball_data.get('bowler2_overs_bowled', 0),
                        'bowler2_maidens_bowled': enhanced_ball_data.get('bowler2_maidens_bowled', 0),
                        'bowler2_runs_conceded': enhanced_ball_data.get('bowler2_runs_conceded', 0),
                        'bowler2_wickets_taken': enhanced_ball_data.get('bowler2_wickets_taken', 0),
                        
                        # Venue and other stats with defaults
                        'venue': enhanced_ball_data.get('venue', 'Unknown'),
                        'matches_played': enhanced_ball_data.get('venue_matches_played', 'N/A'),
                        'average_runs_per_wicket': enhanced_ball_data.get('venue_avg_runs_per_wicket', 'N/A'),
                        'average_runs_per_over': enhanced_ball_data.get('venue_avg_runs_per_over', 'N/A'),
                        
                        # Historical player stats with defaults
                        'batsman1_historical_average': enhanced_ball_data.get('batsman1_historical_average', 0),
                        'batsman1_historical_strike_rate': enhanced_ball_data.get('batsman1_historical_strike_rate', 0),
                        'batsman2_historical_average': enhanced_ball_data.get('batsman2_historical_average', 0),
                        'batsman2_historical_strike_rate': enhanced_ball_data.get('batsman2_historical_strike_rate', 0),
                        'bowler1_historical_average': enhanced_ball_data.get('bowler1_historical_average', 0),
                        'bowler1_historical_economy': enhanced_ball_data.get('bowler1_historical_economy', 0),
                        'bowler1_historical_strike_rate': enhanced_ball_data.get('bowler1_historical_strike_rate', 0),
                        'bowler2_historical_average': enhanced_ball_data.get('bowler2_historical_average', 0),
                        'bowler2_historical_economy': enhanced_ball_data.get('bowler2_historical_economy', 0),
                        'bowler2_historical_strike_rate': enhanced_ball_data.get('bowler2_historical_strike_rate', 0)
                    }
                    
                    
                    rolling_stats = calculate_rolling_averages(enhanced_ball_data, match_state)
                    #update the final data with rolling stats
                    final_data.update(rolling_stats)
                    print(f"Rolling averages calculated: {rolling_stats}")
                    print(f"Final Data: {final_data}")
                    
                    # Add game phase indicators based on over number
                    over_number = enhanced_ball_data['over_number']
                    final_data['powerplay'] = int(over_number < 6)
                    final_data['middle_overs'] = int(over_number >= 6 and over_number < 16)
                    final_data['death_overs'] = int(over_number >= 16)
                    
                    if ball_update.get('forecast_score') is not None and not ball_update['is_second_innings']:
                        final_data['projected_score'] = ball_update['forecast_score']
                        #print(f"Using scraped forecast score: {final_data['projected_score']} from {ball_update['forcast_team']}")
                    else:
                        # In the part where you calculate projected score
                        if 'current_run_rate' in enhanced_ball_data and enhanced_ball_data['current_run_rate']:
                            try:
                                crr = float(enhanced_ball_data.get('current_run_rate', 0))
                                remaining_overs = 20.0 - float(enhanced_ball_data.get('over_number', 0)) - (float(enhanced_ball_data.get('ball_in_over', 0))/6)
                                current_score = 0
                                
                                # Extract current score from batting_team_score (e.g. "120/4")
                                if 'batting_team_score' in enhanced_ball_data and enhanced_ball_data['batting_team_score']:
                                    try:
                                        current_score = int(enhanced_ball_data['batting_team_score'].split('/')[0])
                                    except (ValueError, IndexError, TypeError):
                                        # If we can't parse the score, use alternative methods
                                        if 'total_score' in ball_update:
                                            current_score = ball_update.get('total_score', 0)
                                        else:
                                            # Fallback to runs from batting_team_score if available
                                            batting_score = enhanced_ball_data.get('batting_team_score', '0/0')
                                            try:
                                                current_score = int(batting_score.split('/')[0])
                                            except (ValueError, IndexError, TypeError, AttributeError):
                                                current_score = 0
                                
                                # Calculate projected score
                                projected_score = round(current_score + (crr * remaining_overs))
                                final_data['projected_score'] = projected_score
                                print(f"Projected score: {projected_score} (CRR: {crr})")
                            except Exception as e:
                                print(f"Error calculating projected score: {e}")
                                final_data['projected_score'] = 0  # Default value
                        
                    # Add pressure index for second innings
                    if ball_update['is_second_innings']:
                        # Make sure match_state tracks wickets_lost
                        if 'wickets_lost' not in match_state:
                            match_state['wickets_lost'] = 0
                        
                        # Update wickets_lost if a wicket fell on this ball
                        if enhanced_ball_data.get('is_wicket', False):
                            match_state['wickets_lost'] += 1
                        
                        # Calculate and add pressure index
                        final_data['pressure_index'] = calculate_pressure_index(enhanced_ball_data, match_state)
                        print(f"Pressure index: {final_data['pressure_index']}")
                    else:
                        final_data['pressure_index'] = 0
                    
                    print(f"Final data: {final_data}")
                    
                    # Import feature processor and process data
                    try:
                        # Only initialize once if not already initialized
                        if 'feature_processor' not in locals() or feature_processor is None:
                            from feature_processor import FeatureProcessor
                            feature_processor = FeatureProcessor(model_dir='model_artifacts')
                        
                        import joblib
                        import os
                        import re
                        
                        # Load the trained model
                        model = None
                        name_mapping = None
                        
                        # Try to load feature name mapping first
                        mapping_paths = ['feature_name_mapping.pkl', 'model_artifacts/feature_name_mapping.pkl']
                        for path in mapping_paths:
                            if os.path.exists(path):
                                try:
                                    name_mapping = joblib.load(path)
                                    print(f"✅ Loaded feature name mapping: {len(name_mapping)} features")
                                    break
                                except Exception as e:
                                    print(f"Error loading feature mapping from {path}: {e}")
                        
                        # Function to clean feature names exactly as during training
                        def clean_feature_name(name):
                            clean_name = re.sub(r'[^\w]+', '_', str(name))
                            if clean_name[0].isdigit():
                                clean_name = 'f_' + clean_name
                            return clean_name
                        
                        # If mapping not loaded but clean_feature_names.txt exists, create a mapping
                        if name_mapping is None:
                            feature_list_paths = ['clean_feature_names.txt', 'model_artifacts/clean_feature_names.txt']
                            for path in feature_list_paths:
                                if os.path.exists(path):
                                    try:
                                        with open(path, 'r') as f:
                                            expected_features = [line.strip() for line in f.readlines()]
                                        print(f"✅ Loaded {len(expected_features)} expected feature names from {path}")
                                        break
                                    except Exception as e:
                                        print(f"Error loading feature list from {path}: {e}")
                        
                        # Try to load model with joblib
                        model_paths = [
                            'best_model_Random Forest.pkl',
                            
                        ]
                        
                        for path in model_paths:
                            if os.path.exists(path):
                                try:
                                    model = joblib.load(path)
                                    print(f"✅ Successfully loaded model from {path}: {type(model).__name__}")
                                    break
                                except Exception as e:
                                    print(f"Error loading model from {path}: {e}")
                        
                        if model is not None:
                            # Process the row for prediction
                            # Reconstruct final_data from sample row for testing/debugging
                            if False:
                                # Sample data provided for testing
                                final_data = {
                                    'innings_num': 2,
                                    'batting_team': '',
                                    'over_number': 15,
                                    'ball_number': 5, 
                                    'total_score': 189,
                                    'total_wickets': 5,
                                    'boundaries': 0,
                                    'dot_balls': 0,
                                    'wickets': 0,
                                    'extras': 0,
                                    'favored_team': 'LSG',
                                    'win_percentage': 99.42,
                                    'current_run_rate': 11.93,
                                    'required_run_rate': 0.48,
                                    'striker_batsman': 'David Miller',
                                    'toss_winner': 'LSG',
                                    'toss_decision': 'field first',
                                    'batsman1_name': 'David Miller',
                                    'batsman1_runs': 9,
                                    'batsman1_balls_faced': 6,
                                    'batsman1_fours': 1,
                                    'batsman1_sixes': 0,
                                    'batsman2_name': 'Abdul Samad',
                                    'batsman2_runs': 22,
                                    'batsman2_balls_faced': 7,
                                    'batsman2_fours': 2,
                                    'batsman2_sixes': 2,
                                    'bowler1_name': 'Adam Zampa',
                                    'bowler1_overs_bowled': 3.5,
                                    'bowler1_maidens_bowled': 0,
                                    'bowler1_runs_conceded': 46,
                                    'bowler1_wickets_taken': 1,
                                    'bowler2_name': 'Harshal Patel',
                                    'bowler2_overs_bowled': 2.0,
                                    'bowler2_maidens_bowled': 0,
                                    'bowler2_runs_conceded': 28,
                                    'bowler2_wickets_taken': 1,
                                    'venue': 'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
                                    'matches_played': 139,
                                    'average_runs_per_wicket': 26,
                                    'average_runs_per_over': 8.2,
                                    'batsman1_historical_average': 37.04,
                                    'batsman1_historical_strike_rate': 139.64,
                                    'batsman2_historical_average': 19.23,
                                    'batsman2_historical_strike_rate': 146.07,
                                    'bowler1_historical_average': 18.35,
                                    'bowler1_historical_economy': 8.18,
                                    'bowler1_historical_strike_rate': 13.4,
                                    'bowler2_historical_average': 29.76,
                                    'bowler2_historical_economy': 7.22,
                                    'bowler2_historical_strike_rate': 24.7,
                                    'runs_scored_over_rolling_avg': 8.67,
                                    'wickets_over_rolling_avg': 1.5,
                                    'powerplay': 0,
                                    'middle_overs': 0,
                                    'death_overs': 1,
                                    'pressure_index': 0.6,
                                    'target_score': 239
                                }
                                print("✅ Using test data for feature processing")
                            
                            # Process the features using the processor
                            processed_features = feature_processor.process_row(final_data)
                            
                            if processed_features is not None:
                                # Print diagnostic info about features before alignment
                                print(f"Generated {len(processed_features.columns)} features")
                                
                                # Create a DataFrame with EXACTLY the features the model expects
                                #aligned_df = align_features_for_model(processed_features, model)
                                lgb_aligned_df = align_features_for_model(processed_features, model)

                                if hasattr(model, 'feature_names_in_'):
                                    required_features = model.feature_names_in_
                                    print(f"Model requires exactly {len(required_features)} specific features")

                                    # Print some examples of required features
                                    print("Example required features:")
                                    for feat in required_features[:5]:
                                        print(f"  - {feat}")
                                else:
                                    print("Model doesn't have feature_names_in_ attribute")
                                
                                if os.path.exists('X_test.csv'):
                                    X_test_raw = pd.read_csv('X_test.csv')
                                    print(f"Loaded sample test data with {X_test_raw.shape[1]} features")

                                    # Try loading the feature importance file which might have correct feature names
                                    if os.path.exists('feature_importance.csv'):
                                        feature_info = pd.read_csv('feature_importance.csv')
                                        print(f"Loaded feature importance with {len(feature_info)} features")

                                        # Check if features match what model requires
                                        if 'Feature' in feature_info.columns:
                                            original_features = feature_info['Feature'].tolist()
                                            if set(original_features[:len(required_features)]) == set(required_features):
                                                print("Feature names from feature_importance.csv match model requirements")

                                    # Try loading selected features directly if available
                                    if os.path.exists('selected_features.csv'):
                                        selected_df = pd.read_csv('selected_features.csv')
                                        if 'selected_features' in selected_df.columns:
                                            original_features = selected_df['selected_features'].tolist()
                                            print(f"Loaded {len(original_features)} selected features from file")

                                    # Try loading clean feature names if available
                                    if os.path.exists('clean_feature_names.txt'):
                                        with open('clean_feature_names.txt', 'r') as f:
                                            clean_features = [line.strip() for line in f.readlines()]
                                            print(f"Loaded {len(clean_features)} clean feature names")

                                    # Build feature name mapping
                                    print("\nBuilding feature name mapping...")
                                    feature_map = {}
                        
                                    # Clean test data column names
                                    X_test = X_test_raw.copy()
                                    X_test.columns = [clean_column_name(col) for col in X_test.columns]

                                    # Direct match - exact matches first
                                    for req_feat in required_features:
                                        if req_feat in X_test.columns:
                                            feature_map[req_feat] = req_feat

                                    # Try to find matches for missing features
                                    missing_features = [f for f in required_features if f not in feature_map]
                                    test_columns = [c for c in X_test.columns if c not in feature_map.values()]

                                    if missing_features:
                                        print(f"Still missing {len(missing_features)} features - attempting fuzzy mapping")

                                        # Function to normalize feature names for comparison
                                        def normalize_name(name):
                                            return name.lower().replace('_', '').replace(' ', '')

                                        # Try simple matching
                                        for missing in missing_features:
                                            norm_missing = normalize_name(missing)
                                            found = False

                                            # Try exact normalized matching
                                            for col in test_columns:
                                                if normalize_name(col) == norm_missing:
                                                    feature_map[missing] = col
                                                    test_columns.remove(col)
                                                    found = True
                                                    break

                                            # Try prefix/suffix matching if exact match wasn't found
                                            if not found:
                                                for col in test_columns:
                                                    norm_col = normalize_name(col)
                                                    if (norm_missing in norm_col or norm_col in norm_missing) and (
                                                        len(norm_missing) >= 0.8 * len(norm_col) or len(norm_col) >= 0.8 * len(norm_missing)
                                                    ):
                                                        feature_map[missing] = col
                                                        test_columns.remove(col)
                                                        found = True
                                                        break

                                        # Report mapping status
                                        from fuzzywuzzy import process, fuzz
                                        
                                        mapped_features = set(feature_map.keys())
                                        still_missing = set(required_features) - mapped_features
                                        print(f"Successfully mapped {len(mapped_features)} out of {len(required_features)} required features")

                                        if still_missing:
                                            print(f"Still missing {len(still_missing)} features - will fill with zeros")
                                            for feat in list(still_missing)[:5]:  # Show first 5
                                                print(f"  - Missing: {feat}")
                                        
                                        # 6. Create a new test set with exactly the right features in the right order
                                        print("\nPreparing test data with matching features...")
                                        X_test_prepared = pd.DataFrame(index=X_test.index)
                                        
                                        for feat in required_features:
                                            if feat in feature_map:
                                                X_test_prepared[feat] = X_test[feature_map[feat]]
                                            else:
                                                # Fill missing features with 0
                                                X_test_prepared[feat] = 0
                                                print(f"Filled '{feat}' with zeros (missing feature)")
                                        
                                        print(f"Final test data shape: {X_test_prepared.shape}")
                                        if list(X_test_prepared.columns) == list(required_features):
                                            print("✓ Feature names match exactly - ready for prediction")
                                        else:
                                            print("⚠ Feature names don't match exactly - prediction may fail")
                                            return
                                        # Now that we have the whole DataFrame in the right order in X_test_prepared
                                        # but we've set the processed row in lgb_aligned_df
                                        # We need to get that row into X_test_prepared with the correct feature mapping

                                        # First, ensure X_test_prepared has at least one row
                                        if X_test_prepared.shape[0] == 0:
                                            X_test_prepared = pd.DataFrame(columns=required_features, index=[0])
                                            X_test_prepared.loc[0] = 0.0  # Initialize with zeros

                                        # For each feature in the required features list
                                        for feature in required_features:
                                            # If the feature was mapped (exists in feature_map)
                                            if feature in feature_map:
                                                # Get the corresponding column name from the processed features
                                                source_col = feature_map[feature]
                                                # Copy the value from processed_features to X_test_prepared
                                                if source_col in lgb_aligned_df.columns:
                                                    # Check if this is a boolean feature that needs special handling
                                                    is_bool_feature = (
                                                        'corrected' in feature or 
                                                        feature.startswith(('toss_winner_', 'batting_team_', 'favored_team_'))
                                                    )
                                                    
                                                    # Get the raw value from lgb_aligned_df
                                                    raw_value = lgb_aligned_df[source_col].iloc[0]
                                                    
                                                    # For boolean features, explicitly convert to bool before assignment
                                                    if is_bool_feature:
                                                        if isinstance(raw_value, (int, float)):
                                                            X_test_prepared.loc[0, feature] = bool(raw_value)
                                                        elif isinstance(raw_value, str):
                                                            X_test_prepared.loc[0, feature] = raw_value.lower() in ['true', '1', 't', 'yes']
                                                        else:
                                                            X_test_prepared.loc[0, feature] = bool(raw_value)
                                                    else:
                                                        X_test_prepared.loc[0, feature] = raw_value
                                                        
                                                    print(f"Matched feature: '{feature}' using source column '{source_col}'")
                                                else:
                                                    # Special handling for venue-related features using fuzzy matching
                                                    if source_col.startswith('venue_'):
                                                        # Extract the venue name from the feature
                                                        venue_name = source_col[len('venue_'):]
                                                        
                                                        # Check if 'venue' column exists in our processed features
                                                        if 'venue' in processed_features.columns:
                                                            current_venue = str(processed_features['venue'].iloc[0])
                                                            # Use fuzzy matching to see if this is our venue
                                                            match_score = fuzz.partial_ratio(venue_name.lower().replace('_', ' '), 
                                                                                            current_venue.lower())
                                                            
                                                            if match_score >= 80:  # High match score threshold
                                                                X_test_prepared.loc[0, feature] = True
                                                                print(f"Fuzzy matched venue feature: '{source_col}' to '{current_venue}' (score: {match_score})")
                                                            else:
                                                                X_test_prepared.loc[0, feature] = False
                                                                print(f"Venue '{venue_name}' doesn't match current venue '{current_venue}' (score: {match_score})")
                                                        else:
                                                            # No venue information available
                                                            X_test_prepared.loc[0, feature] = False
                                                            print(f"No venue information available - setting '{source_col}' to False")
                                                    else:
                                                        # For other missing columns, use default 0.0
                                                        X_test_prepared.loc[0, feature] = 0.0
                                                        print(f"Missing column '{source_col}' - using default 0.0")
                                            else:
                                                # For unmapped features, keep the zero default
                                                X_test_prepared.loc[0, feature] = 0.0
                                                print(f"Using default value 0.0 for unmapped feature: {feature}")

                                        # Set lgb_aligned_df to the properly ordered X_test_prepared for prediction
                                        lgb_aligned_df = X_test_prepared.copy()
                                        print(f"Feature alignment complete: {lgb_aligned_df.shape[1]} features prepared for model")

                                        # Save the dtypes to df_dtypes.csv
                                        df_dtypes = pd.DataFrame(X_test_prepared.dtypes, columns=['dtype'])
                                        df_dtypes.to_csv('df_dtypes.csv')
                                        print(f"✅ Saved {len(df_dtypes)} feature data types to df_dtypes.csv")

                                        dtypes_mapping = {}
                                        dtypes_file = 'df_dtypes.csv'
                                        if os.path.exists(dtypes_file):
                                            try:
                                                dtype_df = pd.read_csv(dtypes_file, index_col=0)
                                                dtypes_mapping = dtype_df['dtype'].to_dict()
                                                print(f"Loaded {len(dtypes_mapping)} data types")
                                            except Exception as e:
                                                print(f"Error loading data types: {e}")
                                                
                                        # Check data types to ensure they match model expectations
                                        dtype_mismatches = []
                                        for col in lgb_aligned_df.columns:
                                            if col in dtypes_mapping:
                                                expected_dtype = dtypes_mapping[col]
                                                current_dtype = str(lgb_aligned_df[col].dtype)
                                                if not current_dtype.startswith(expected_dtype.replace('64', '').replace('32', '')):
                                                    dtype_mismatches.append(f"{col}: expected {expected_dtype}, got {current_dtype}")

                                        if dtype_mismatches:
                                            print(f"Warning: {len(dtype_mismatches)} data type mismatches found:")
                                            for mismatch in dtype_mismatches[:5]:  # Show first 5 mismatches
                                                print(f"  - {mismatch}")
                                            if len(dtype_mismatches) > 5:
                                                print(f"  - ... and {len(dtype_mismatches) - 5} more")
                                        else:
                                            print("All feature data types match expected types ✅")
                                            
                                        probability = model.predict_proba(lgb_aligned_df)[0, 1] if hasattr(model, "predict_proba") else None
                                        #xgbprobability = xgb_model.predict_proba(lgb_aligned_df)[0, 1] if hasattr(xgb_model, "predict_proba") else None
                                        #lgbprobability = lgb_model.predict_proba(lgb_aligned_df)[0, 1] if hasattr(lgb_model, "predict_proba") else None
                                        lgb_aligned_df.dtypes.to_csv('X_test_clean_dtypes.csv', header=['dtype'])
                                        print(f"✅ Model prediction: {probability:.4f} probability of batting team winning")
                                        
                                         
                                        
                                                    
                                
                                
                                # Make prediction with perfectly aligned features
                                try:
                                    #probability = model.predict_proba(lgb_aligned_df)[0, 1] if hasattr(model, "predict_proba") else None
                                    #xgbprobability = xgb_model.predict_proba(lgb_aligned_df)[0, 1] if hasattr(xgb_model, "predict_proba") else None
                                    #lgbprobability = lgb_model.predict_proba(lgb_aligned_df)[0, 1] if hasattr(lgb_model, "predict_proba") else None
                                    #lgb_aligned_df.dtypes.to_csv('X_test_clean_dtypes.csv', header=['dtype'])
                                    #print(f"✅ Model prediction: {probability:.4f} probability of batting team winning")
                                    #print(f"✅ XGBoost model prediction: {xgbprobability:.4f} probability of batting team winning")
                                    #print(f"✅ LightGBM model prediction: {lgbprobability:.4f} probability of batting team winning")
                        
                                    
                                    # Add to final_data for saving
                                    final_data['model_win_probability'] = probability
                                    
                                    # Check if this differs from market probability
                                    market_prob = final_data['win_percentage'] / 100
                                    # Adjust market probability based on which team is favored
                                    if final_data['favored_team'] != final_data['batting_team']:
                                        market_prob = 1 - market_prob
                                        
                                    edge = probability - market_prob
                                    print(f"📈 Market probability: {market_prob:.4f}")
                                    print(f"🔍 Model edge: {edge:.4f}")
                                    
                                    # Signal betting opportunity
                                    if abs(edge) > 0.1:  # 10% edge threshold
                                        print(f"⚠️ SIGNIFICANT EDGE DETECTED: {abs(edge)*100:.1f}%")
                                        
                                except Exception as e:
                                    print(f"❌ Error making prediction: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print("Failed to process features for prediction")
                        else:
                            print("No model available for prediction")
                    
                    except Exception as e:
                        print(f"❌ Error in prediction pipeline: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Save the data to a CSV file
                    file_name = f"{match_id}_ball_feeders.csv"
                    file_path = os.path.join(match_dir, file_name)
                    # Check if file exists
                    if os.path.exists(file_path):
                        # Append to existing file
                        with open(file_path, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=final_data.keys())
                            writer.writerow(final_data)
                    else:
                        # Create a new file
                        with open(file_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=final_data.keys())
                            writer.writeheader()
                            writer.writerow(final_data)
     
                    
                except KeyboardInterrupt:
                    print("Monitoring stopped by user")
                except Exception as e:
                    print(f"Error during monitoring: {e}")
                    traceback.print_exc()
                    time.sleep(5)  # Wait before retrying
        
        print(f"All operations completed. Data saved in: {match_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing match: {e}")
        traceback.print_exc()
        return False

def align_features_for_model(processed_features, model):
    """
    Creates a DataFrame with EXACTLY the features the model expects, using fuzzy matching
    to handle slight differences in feature names.
    
    Args:
        processed_features: DataFrame with processed features
        model: Trained model with feature_names_in_ attribute
        
    Returns:
        DataFrame with exactly the features the model expects
    """
    import pandas as pd
    from fuzzywuzzy import process
    import re
    
    # Get the exact feature names the model expects
    if not hasattr(model, 'feature_names_in_'):
        print("Model doesn't have feature_names_in_ attribute!")
        return processed_features
    
    model_features = list(model.feature_names_in_)
    print(f"Model expects exactly {len(model_features)} features")
    
    # Create empty DataFrame with exactly the columns the model expects
    aligned_df = pd.DataFrame(columns=model_features)
    aligned_df.loc[0] = 0.0  # Initialize all values to zero
    
    # Track feature matching stats
    exact_matches = 0
    fuzzy_matches = 0
    matched_features = set()  # Track which features have been matched
    
    # Team name mappings (abbreviated to full name and vice versa)
    team_mappings = {
        'CSK': 'Chennai_Super_Kings',
        'Chennai_Super_Kings': 'CSK',
        'MI': 'Mumbai_Indians',
        'Mumbai_Indians': 'MI',
        'RCB': 'Royal_Challengers_Bangalore',
        'Royal_Challengers_Bangalore': 'RCB',
        'DC': 'Delhi_Capitals',
        'Delhi_Capitals': 'DC',
        'KKR': 'Kolkata_Knight_Riders',
        'Kolkata_Knight_Riders': 'KKR',
        'PBKS': 'Punjab_Kings',
        'Punjab_Kings': 'PBKS',
        'RR': 'Rajasthan_Royals',
        'Rajasthan_Royals': 'RR',
        'SRH': 'Sunrisers_Hyderabad',
        'Sunrisers_Hyderabad': 'SRH'
    }
    
    # First try exact matches
    processed_cols = list(processed_features.columns)
    for feature in model_features:
        if feature in processed_cols:
            aligned_df.loc[0, feature] = processed_features[feature].iloc[0]
            exact_matches += 1
            processed_cols.remove(feature)  # Remove matched columns
            matched_features.add(feature)
    
    # List of venue features that need default values when missing
    venue_features = [
        'venue_Eden_Gardens_Kolkata',
        'venue_MA_Chidambaram_Stadium_Chepauk_Chennai',
        'venue_Narendra_Modi_Stadium_Ahmedabad'
    ]
    
    # For remaining features, try special pattern matching first, then fuzzy matching
    for feature in model_features:
        if feature in matched_features:
            continue  # Skip already matched features
            
        # Special case for team name features (batting_team, toss_winner, etc.)
        team_feature_match = False
        for prefix in ['batting_team_', 'bowling_team_', 'toss_winner_', 'favored_team_']:
            if feature.startswith(prefix):
                team_name = feature[len(prefix):]  # Extract the team name part
                
                # Try to find a matching column using team name mappings
                for team_abbr, team_full in team_mappings.items():
                    if team_name == team_full:
                        # Look for abbreviated version in processed columns
                        alt_feature = f"{prefix}{team_abbr}"
                        if alt_feature in processed_cols:
                            aligned_df.loc[0, feature] = processed_features[alt_feature].iloc[0]
                            processed_cols.remove(alt_feature)
                            fuzzy_matches += 1
                            matched_features.add(feature)
                            team_feature_match = True
                            print(f"Team name matched: {feature} -> {alt_feature}")
                            break
                    elif team_name == team_abbr:
                        # Look for full version in processed columns
                        alt_feature = f"{prefix}{team_full}"
                        if alt_feature in processed_cols:
                            aligned_df.loc[0, feature] = processed_features[alt_feature].iloc[0]
                            processed_cols.remove(alt_feature)
                            fuzzy_matches += 1
                            matched_features.add(feature)
                            team_feature_match = True
                            print(f"Team name matched: {feature} -> {alt_feature}")
                            break
        
        if team_feature_match:
            continue
            
        # Special case for known features that need defaults
        if feature == 'bowler2_historical_economy_corrected':
            aligned_df.loc[0, feature] = 0  # Set to 0 directly
            exact_matches += 1 
            matched_features.add(feature)
            print(f"Added {feature} with default value 0")
            continue
        elif feature == 'total_score':
            aligned_df.loc[0, feature] = 0  # Set default value
            exact_matches += 1
            matched_features.add(feature)
            print(f"Added {feature} with default value 0")
            continue
        elif feature in venue_features:
            aligned_df.loc[0, feature] = 0  # Set venue features to 0
            exact_matches += 1
            matched_features.add(feature)
            print(f"Added venue feature {feature} with default value 0")
            continue
            
        # Find best fuzzy match among remaining columns
        if processed_cols:  # Only if we have unmatched columns left
            best_match, score = process.extractOne(feature, processed_cols)
            if score >= 60:  # Minimum similarity threshold
                aligned_df.loc[0, feature] = processed_features[best_match].iloc[0]
                processed_cols.remove(best_match)  # Remove matched column
                fuzzy_matches += 1
                matched_features.add(feature)
                print(f"Fuzzy matched: {feature} -> {best_match} (score: {score})")
    
    print(f"Feature matching summary:")
    print(f"- Exact matches: {exact_matches}")
    print(f"- Fuzzy matches: {fuzzy_matches}")
    print(f"- Total matched: {exact_matches + fuzzy_matches} of {len(model_features)}")
    
    # Find and print unmatched features
    unmatched_features = set(model_features) - matched_features
    if unmatched_features:
        print(f"\n⚠️ Warning: {len(unmatched_features)} features could not be matched:")
        for feature in unmatched_features:
            print(f"- Missing feature: {feature}")
    
    # Load expected data types
    dtypes_mapping = {}
    dtypes_file = 'df_dtypes.csv'
    if os.path.exists(dtypes_file):
        try:
            dtype_df = pd.read_csv(dtypes_file, index_col=0)
            dtypes_mapping = dtype_df['dtype'].to_dict()
            print(f"Loaded {len(dtypes_mapping)} data types from {dtypes_file}")
        except Exception as e:
            print(f"Error loading data types: {e}")

    # First pass: Convert all string/object types to appropriate numeric values
    for feature in aligned_df.columns:
        if feature in dtypes_mapping and aligned_df[feature].dtype == 'object':
            try:
                # Special handling for known problematic fields
                if feature == 'matches_played' or feature in ['batsman1_historical_average', 'batsman1_historical_strike_rate']:
                    # Convert 'N/A' or other non-numeric strings to NaN, then to 0
                    aligned_df[feature] = pd.to_numeric(aligned_df[feature], errors='coerce').fillna(0)
                elif feature.startswith('batting_team_') or feature.startswith('favored_team_') or feature.startswith('toss_winner_'):
                    # Boolean features stored as objects
                    bool_val = aligned_df[feature].iloc[0]
                    if isinstance(bool_val, str):
                        aligned_df[feature] = bool_val.lower() in ['true', '1', 't', 'yes']
                    else:
                        aligned_df[feature] = bool(bool_val)
            except Exception as e:
                print(f"Initial conversion error for {feature}: {e}")

    # Second pass: Apply exact data types from the CSV
    for feature in aligned_df.columns:
        try:
            if feature in dtypes_mapping:
                dtype = dtypes_mapping[feature]
                current_value = aligned_df.loc[0, feature]
                
                # Special handling for problematic fields
                if feature == 'over_number':  # Ensure this is int64
                    aligned_df[feature] = aligned_df[feature].astype('int64')
                    print(f"Forced {feature} to int64")
                    
                elif dtype == 'float64':
                    aligned_df[feature] = aligned_df[feature].astype('float64')
                    
                elif dtype == 'int64' or dtype == 'int32':
                    # First make sure it's float (handles 'object' types)
                    aligned_df[feature] = pd.to_numeric(aligned_df[feature], errors='coerce').fillna(0)
                    # Then convert to int
                    aligned_df[feature] = aligned_df[feature].astype('int32' if dtype == 'int32' else 'int64')
                    
                elif dtype == 'bool':
                    # First ensure it's numeric (0 or 1)
                    if aligned_df[feature].dtype == 'object':
                        # Convert string booleans to actual booleans
                        bool_val = aligned_df[feature].iloc[0]
                        if isinstance(bool_val, str):
                            aligned_df.loc[0, feature] = bool_val.lower() in ['true', '1', 't', 'yes']
                        else:
                            aligned_df.loc[0, feature] = bool(bool_val)
                    # Now convert to bool type
                    aligned_df[feature] = aligned_df[feature].astype('bool')
            
        except Exception as e:
            print(f"Warning: Could not convert {feature} to {dtypes_mapping.get(feature, 'unknown')}: {e}")
            # Use appropriate defaults
            if dtypes_mapping.get(feature) == 'bool':
                aligned_df.loc[0, feature] = False
            elif dtypes_mapping.get(feature) == 'float64':
                aligned_df.loc[0, feature] = 0.0
            elif dtypes_mapping.get(feature) in ['int64', 'int32']:
                aligned_df.loc[0, feature] = 0

    # Final verification - print dtypes of all columns
    print("\nFinal data types:")
    for col, dtype in aligned_df.dtypes.items():
        expected = dtypes_mapping.get(col, 'unknown')
        match = "✓" if str(dtype).startswith(expected.replace('32', '').replace('64', '')) else "✗"
        print(f"{col}: {dtype} (expected: {expected}) {match}")

    # Final check for any remaining object types
    object_cols = aligned_df.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"\n⚠️ Warning: {len(object_cols)} columns still have object type: {list(object_cols)}")
        # Force convert remaining object columns to float
        for col in object_cols:
            aligned_df[col] = 0.0
            print(f"Forced {col} to float64")

    print("\n✅ Data type conversion complete")
    
    return aligned_df


def align_features_for_lgbm(processed_features, lgb_model):
    """
    Creates a DataFrame with EXACTLY the features the LightGBM model expects,
    precisely matching the training process with clean_feature_names.txt support.
    
    Args:
        processed_features: DataFrame with processed features
        lgb_model: LightGBM model with feature_name_ attribute
        
    Returns:
        DataFrame with exactly the features the model expects
    """
    import pandas as pd
    import numpy as np
    import os
    from fuzzywuzzy import process, fuzz
    
    # Get required feature names following training code approach
    if hasattr(lgb_model, 'feature_name_'):
        # LightGBM uses feature_name_ 
        required_features = list(lgb_model.feature_name_)
    elif hasattr(lgb_model, 'feature_names_in_'):
        # Fallback
        required_features = list(lgb_model.feature_names_in_)
    else:
        print("LightGBM model doesn't have feature_name_ attribute!")
        return processed_features
    
    print(f"LightGBM model requires exactly {len(required_features)} specific features")
    
    # Try to load feature name mapping from training
    feature_map = {}
    try:
        if os.path.exists('feature_name_mapping.pkl'):
            import joblib
            feature_map = joblib.load('feature_name_mapping.pkl')
            print(f"Loaded feature mapping with {len(feature_map)} entries from file")
        
        # Try loading clean feature names if available for reference
        clean_features = []
        if os.path.exists('clean_feature_names.txt'):
            with open('clean_feature_names.txt', 'r') as f:
                clean_features = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(clean_features)} clean feature names for reference")
    except Exception as e:
        print(f"Could not load feature mapping: {e}")
    
    # Helper function to normalize feature names exactly as done in training
    def normalize_name(name):
        return name.lower().replace('_', '').replace(' ', '')
    
    # Clean test data column names in the same way as training
    processed_cols = list(processed_features.columns)
    
    # Create empty DataFrame with exactly the columns the model expects
    aligned_df = pd.DataFrame(columns=required_features)
    aligned_df.loc[0] = 0.0  # Initialize all values to zero
    
    # Track feature matching stats
    exact_matches = 0
    fuzzy_matches = 0
    mapped_features = set()
    
    # First try exact matches
    for req_feat in required_features:
        if req_feat in processed_cols:
            aligned_df.loc[0, req_feat] = processed_features[req_feat].iloc[0]
            exact_matches += 1
            mapped_features.add(req_feat)
    
    # Use feature_map if available
    if feature_map:
        for req_feat in [f for f in required_features if f not in mapped_features]:
            if req_feat in feature_map:
                col_name = feature_map[req_feat]
                if col_name in processed_features.columns:
                    aligned_df.loc[0, req_feat] = processed_features[col_name].iloc[0]
                    exact_matches += 1
                    mapped_features.add(req_feat)
    
    # Extended team name mappings
    team_mappings = {
        'CSK': 'Chennai_Super_Kings',
        'Chennai_Super_Kings': 'CSK',
        'MI': 'Mumbai_Indians',
        'Mumbai_Indians': 'MI',
        'RCB': 'Royal_Challengers_Bangalore',
        'Royal_Challengers_Bangalore': 'RCB',
        'Royal_Challengers_Bengaluru': 'RCB',  # Ensure this maps to RCB
        'DC': 'Delhi_Capitals',
        'Delhi_Capitals': 'DC',
        'KKR': 'Kolkata_Knight_Riders',
        'Kolkata_Knight_Riders': 'KKR',
        'PBKS': 'Punjab_Kings',
        'Punjab_Kings': 'PBKS',
        'RR': 'Rajasthan_Royals',
        'Rajasthan_Royals': 'RR',
        'SRH': 'Sunrisers_Hyderabad',
        'Sunrisers_Hyderabad': 'SRH',
        'GT': 'Gujarat_Titans', 
        'Gujarat_Titans': 'GT',
        'LSG': 'Lucknow_Super_Giants',
        'Lucknow_Super_Giants': 'LSG'
    }
    
    # Special handling for team features
    team_prefixes = ['batting_team_', 'bowling_team_', 'toss_winner_', 'favored_team_']
    
    # Handle special case features
    corrected_boolean_features = [
        'bowler2_historical_average_corrected',
        'bowler2_historical_strike_rate_corrected', 
        'bowler1_historical_average_corrected',
        'bowler1_historical_strike_rate_corrected',
        'batsman1_historical_average_corrected',
        'batsman1_historical_strike_rate_corrected',
        'batsman2_historical_average_corrected',
        'batsman2_historical_strike_rate_corrected'
    ]
    
    # Handle venue features - get current venue and set required venues to false if they don't match
    venue_features = [
        'venue_Narendra_Modi_Stadium_Ahmedabad',
        'venue_Eden_Gardens_Kolkata',
        'venue_MA_Chidambaram_Stadium_Chepauk_Chennai'
    ]
    
    # Get the current venue from processed features
    current_venue = None
    if 'venue' in processed_features.columns:
        
        # Handle team feature special cases
        team_matched = False
        for prefix in team_prefixes:
            if feature.startswith(prefix):
                team_name = feature[len(prefix):]  # Extract team part
                
                # Special handling for Royal_Challengers_Bengaluru -> RCB
                if team_name == 'Royal_Challengers_Bengaluru':
                    # Check if this should map to the current team (RCB)
                    for input_prefix in ['batting_team', 'toss_winner', 'favored_team']:
                        if input_prefix in processed_features.columns:
                            current_team = processed_features[input_prefix].iloc[0]
                            if current_team == 'RCB':
                                aligned_df.loc[0, feature] = True
                                print(f"Fixed mapping: '{feature}' -> RCB (score: 100)")
                                team_matched = True
                                fuzzy_matches += 1
                                mapped_features.add(feature)
                                break
                    
                    if not team_matched:
                        aligned_df.loc[0, feature] = False
                        fuzzy_matches += 1
                        mapped_features.add(feature)
                        team_matched = True
                    continue

                # Check if this matches the current team
                for input_prefix in ['batting_team', 'toss_winner', 'favored_team']:
                    if input_prefix in processed_features.columns:
                        current_team = processed_features[input_prefix].iloc[0]
                        # Check if this is our team using mappings
                        for abbr, full in team_mappings.items():
                            if (team_name == abbr and current_team == abbr) or \
                               (team_name == full and current_team == abbr) or \
                               (team_name == abbr and current_team == full):
                                aligned_df.loc[0, feature] = True  # Match found
                                team_matched = True
                                fuzzy_matches += 1
                                mapped_features.add(feature)
                                print(f"Team feature '{feature}' matched to current team '{current_team}'")
                                break
                        
                        if not team_matched:
                            # Not our team
                            aligned_df.loc[0, feature] = False
                            team_matched = True
                            fuzzy_matches += 1
                            mapped_features.add(feature)
        
                        if team_matched:
                            continue
            
        # Try fuzzy matching for remaining features using normalized names
        norm_feature = normalize_name(feature)
        best_match = None
        best_score = 0
        
        for col in processed_cols:
            norm_col = normalize_name(col)
            # Check exact normalized match
            if norm_col == norm_feature:
                best_match = col
                best_score = 100
                break
            
            # Check if one is prefix/suffix of the other with high overlap
            if (norm_feature in norm_col or norm_col in norm_feature) and \
               (len(norm_feature) >= 0.8*len(norm_col) or len(norm_col) >= 0.8*len(norm_feature)):
                score = 80
                if score > best_score:
                    best_match = col
                    best_score = score
        
        # If no good match found through prefix/suffix, use fuzzywuzzy
        if not best_match and processed_cols:
            best_match, score = process.extractOne(feature, processed_cols)
            if score >= 60:
                best_score = score
        
        if best_match and best_score >= 60:
            aligned_df.loc[0, feature] = processed_features[best_match].iloc[0]
            fuzzy_matches += 1
            mapped_features.add(feature)
            print(f"Fuzzy matched '{feature}' to '{best_match}' (score: {best_score})")
            
    # Report mapping status
    print(f"Feature mapping summary:")
    print(f"- Exact matches: {exact_matches}")
    print(f"- Fuzzy matches: {fuzzy_matches}")
    print(f"- Total matched: {len(mapped_features)} of {len(required_features)}")
    
    # Report on missing features
    still_missing = set(required_features) - mapped_features
    if still_missing:
        print(f"Missing {len(still_missing)} features - filled with zeros:")
        # Show the top N missing features (up to 10)
        missing_list = list(still_missing)
        for i, feature in enumerate(missing_list[:10]):
            print(f"  {i+1}. {feature}")
        if len(missing_list) > 10:
            print(f"  ... and {len(missing_list)-10} more")
    
    # Apply data types correctly
    dtypes_mapping = {}
    dtypes_file = 'df_dtypes.csv'
    if os.path.exists(dtypes_file):
        try:
            dtype_df = pd.read_csv(dtypes_file, index_col=0)
            dtypes_mapping = dtype_df['dtype'].to_dict()
            print(f"Loaded {len(dtypes_mapping)} data types")
        except Exception as e:
            print(f"Error loading data types: {e}")
    
    # Apply correct data types
    for feature in aligned_df.columns:
        try:
            if feature in dtypes_mapping:
                dtype = dtypes_mapping[feature]
                
                if dtype == 'float64':
                    aligned_df[feature] = aligned_df[feature].astype('float64')
                    
                elif dtype in ['int64', 'int32']:
                    aligned_df[feature] = pd.to_numeric(aligned_df[feature], errors='coerce').fillna(0)
                    aligned_df[feature] = aligned_df[feature].astype(dtype)
                    
                elif dtype == 'bool':
                    if isinstance(aligned_df[feature].iloc[0], str):
                        aligned_df.loc[0, feature] = aligned_df[feature].iloc[0].lower() in ['true', '1', 't', 'yes']
                    aligned_df[feature] = aligned_df[feature].astype('bool')
        except Exception as e:
            print(f"Warning: Could not convert {feature} to {dtype}: {e}")
    
    # Final check for any remaining object types
    object_cols = aligned_df.select_dtypes(include=['object']).columns
    if not object_cols.empty:
        print(f"Warning: {len(object_cols)} columns still have object type")
        for col in object_cols:
            aligned_df[col] = 0.0
    
    print("Feature alignment complete for LightGBM")
    return aligned_df

if __name__ == "__main__":
    
    
    
    # Check if URL is provided as command line argument
    if len(sys.argv) > 1:
        url = sys.argv[2]
        launch_prediction_display(url)
        print(f"Processing URL: {url}")
        main(url)
    else:
        url=None
        launch_prediction_display(url)    
        print("No URL provided, using default URL")       
        main()  # Use default URL