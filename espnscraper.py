import csv
import os
import re
import sys  # Add the missing sys import
import traceback  # Add traceback for exception handling
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from batting_stats import BattingHistoryDownloader

from enum import Enum

class IPLTeam(Enum):
    MI = "Mumbai Indians"
    DC = "Delhi Capitals"
    SRH = "Sunrisers Hyderabad"
    RCB = "Royal Challengers Bangalore"
    KKR = "Kolkata Knight Riders"
    KXIP = "Kings XI Punjab"  # This team is now Punjab Kings, but using KXIP as requested
    CSK = "Chennai Super Kings"
    RR = "Rajasthan Royals"
    
def fetch_team_names_and_player_sections(page):
    """
    Fetch team names and their corresponding player sections, correctly mapping bowling sections
    to the opposing team.
    """
    team_sections = []

    # Locate team names like "MI Innings" or "Mumbai Indians"
    team_name_elements = page.locator('span.ds-text-title-xs.ds-font-bold.ds-capitalize')
    team_names = [team_name_elements.nth(i).inner_text().strip() 
                 for i in range(team_name_elements.count())]

    # Locate all player table sections
    all_divs = page.locator('div.ds-w-full.ds-bg-fill-content-prime.ds-overflow-hidden.ds-rounded-xl.ds-border.ds-border-line.ds-mb-4')

    for i in range(len(team_names)):
        # Get current team name and the opposing team name
        current_team = team_names[i]
        opposing_team = team_names[1] if i == 0 else team_names[0]
        
        player_section = all_divs.nth(i)
        
        # Extract batting and bowling sections
        tables = player_section.locator('table')
        batting_section = tables.first
        bowling_section = tables.last
        
        # Add team and sections to list with correct team mappings
        team_sections.append({
            "team_name": current_team,
            "batting_section": batting_section,
            "team_bowling": opposing_team,  # The bowling section shows the opposing team's bowlers
            "bowling_section": bowling_section
        })

    print(f"Extracted team names and corresponding sections with bowling mappings.")
    return team_sections


def fetch_player_links(page):
    """
    Fetch all player profile links from the scorecard page.
    """
    player_links = []
    players = page.locator('a[href^="/cricketers/"]')  # Locate all player links
    
    for i in range(players.count()):
        try:
            player_name = players.nth(i).inner_text().strip()
            player_link = players.nth(i).get_attribute("href")
            player_id_match = re.search(r'/cricketers/.*?-(\d+)', player_link)
            if player_id_match:
                player_id = player_id_match.group(1)
                player_links.append({"name": player_name, "link": f"https://www.espncricinfo.com{player_link}", "id": player_link.replace('/cricketers/', '')})
        except Exception as e:
            print(f"Error fetching player link at index {i}: {e}")
    
    return player_links

def fetch_player_links_with_teams(team_sections):
    """
    Fetch player profile links and assign corresponding team names.
    Also handles bowling sections with correct team mapping.
    """
    player_links = []

    for section in team_sections:
        # Process batting section
        team_name = section["team_name"]
        team_name = next((team.name for team in IPLTeam if team.value == team_name), team_name)
        
        # Process batting players
        batting_section = section["batting_section"]
        batsmen = batting_section.locator('a[href^="/cricketers/"]')
        
        for player_index in range(batsmen.count()):
            try:
                player_name = batsmen.nth(player_index).inner_text().strip()
                player_link = batsmen.nth(player_index).get_attribute("href")
                player_id_match = re.search(r'/cricketers/.*?-(\d+)', player_link)
                if player_id_match:
                    player_id = player_id_match.group(1)
                    player_links.append({
                        "name": player_name,
                        "link": f"https://www.espncricinfo.com{player_link}",
                        "id": player_link.replace('/cricketers/', ''),
                        "team": team_name,
                        "role": "batsman"
                    })
            except Exception as e:
                print(f"Error fetching batsman link for team {team_name}: {e}")
        
        # Process bowling section (belongs to opposing team)
        bowling_team = section["team_bowling"]
        bowling_team = next((team.name for team in IPLTeam if team.value == bowling_team), bowling_team)
        bowling_section = section["bowling_section"]
        bowlers = bowling_section.locator('a[href^="/cricketers/"]')
        
        for player_index in range(bowlers.count()):
            try:
                player_name = bowlers.nth(player_index).inner_text().strip()
                player_link = bowlers.nth(player_index).get_attribute("href")
                player_id_match = re.search(r'/cricketers/.*?-(\d+)', player_link)
                if player_id_match:
                    player_id = player_id_match.group(1)
                    player_links.append({
                        "name": player_name,
                        "link": f"https://www.espncricinfo.com{player_link}",
                        "id": player_link.replace('/cricketers/', ''),
                        "team": bowling_team,
                        "role": "bowler"
                    })
            except Exception as e:
                print(f"Error fetching bowler link for team {bowling_team}: {e}")

    return player_links

def fetch_and_save_player_stats(player_links,page, match_id=None):
    """
    Fetch and save player statistics organized by team and role.
    Include match ID in filenames.
    """
    downloader = BattingHistoryDownloader()
    team_stats = {}

    # 1. Data Collection Phase
    for player in player_links:
        if not all(key in player for key in ["id", "name", "team", "role"]):
            print(f"Skipping player with incomplete data: {player}")
            continue

        player_id = player["id"]
        player_name = player["name"]
        player_team = player["team"]
        player_role = player["role"]

        print(f"Fetching IPL stats for {player_name} (ID: {player_id}, Team: {player_team}, Role: {player_role})...")

        try:
            # Initialize team stats if needed
            if player_team not in team_stats:
                team_stats[player_team] = {"batting": [], "bowling": []}

            # Fetch stats
            batting_df, bowling_df = downloader.get_ipl_batting_stats(player_id ,page)

            # Process batting stats (for batsmen or all-rounders)
            if batting_df is not None and not batting_df.empty:
                stats = process_batting_stats(player_name, player_role,player_team, batting_df)
                if stats:
                    team_stats[player_team]["batting"].extend(stats)

            # Process bowling stats (for bowlers or all-rounders)
            if bowling_df is not None and not bowling_df.empty:
                stats = process_bowling_stats(player_name, player_role,player_team, bowling_df)
                if stats:
                    team_stats[player_team]["bowling"].extend(stats)

        except Exception as e:
            print(f"Error fetching stats for {player_name}: {e}")
            continue

    # Save Phase with match ID
    save_team_stats(team_stats, match_id)
    return team_stats

def fetch_player_batting_summary(player_id):
    downloader = BattingHistoryDownloader()
    try:
        summary = downloader.get_career_summary(player_id)
        print(f"Batting Summary for Player {player_id}:")
        print(summary)
        return summary
    except Exception as e:
        print(f"Error fetching batting summary for {player_id}: {e}")
        return None

def save_to_csv_players(data, filename, is_batting=True, match_id=None):
    """
    Save the player stats into a CSV file, with separate files for each team and role.
    Include match ID in filename and save in appropriate folder.
    """
    # Get current directory and create season folder
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # season_folder, match_folder = get_match_info_from_url(match_id) if match_id else ("default_season", "default_match")
    
    # Create full path for match directory
    # match_dir = os.path.join(current_dir, season_folder, match_folder)
    ensure_directory_exists(match_id)

    # Rest of the existing code...
    team_role_data = {}
    for record in data:
        team = record["player_team"]
        role = record["player_role"]
        
        if team not in team_role_data:
            team_role_data[team] = {"batsmen": [], "bowlers": []}
            
        if role == "batsman" and is_batting:
            team_role_data[team]["batsmen"].append(record)
        elif role == "bowler" and not is_batting:
            team_role_data[team]["bowlers"].append(record)

    # Define fieldnames based on the type of stats
    batting_fieldnames = [
        "player_name", "player_role", "matches", "innings", "not_outs", 
        "runs", "balls", "fours", "sixes", "average", "strike_rate"
    ]
    
    bowling_fieldnames = [
        "player_name", "player_role", "matches", "innings", "overs", 
        "maidens", "runs_conceded", "wickets", "average", "economy", "strike_rate"
    ]

    # Save files in the match directory
    for team, roles in team_role_data.items():
        if is_batting and roles["batsmen"]:
            match_filename = os.path.join(match_id, f"{team}_batting_stats.csv")
            with open(match_filename, "w", newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=batting_fieldnames)
                writer.writeheader()
                for row in roles["batsmen"]:
                    # Remove team from row since it's in filename
                    row_copy = row.copy()
                    if "player_team" in row_copy:
                        del row_copy["player_team"]
                    writer.writerow(row_copy)
            print(f"Batting stats saved to {match_filename}")
            
        elif not is_batting and roles["bowlers"]:
            match_filename = os.path.join(match_id, f"{team}_bowling_stats.csv")
            with open(match_filename, "w", newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=bowling_fieldnames)
                writer.writeheader()
                for row in roles["bowlers"]:
                    # Remove team from row since it's in filename
                    row_copy = row.copy()  # Create row_copy for each row
                    if "player_team" in row_copy:
                        del row_copy["player_team"]
                    writer.writerow(row_copy)
            print(f"Bowling stats saved to {match_filename}")

def fetch_player_match_stats(player_id, format_str="T20Is"):
    downloader = BattingHistoryDownloader()
    try:
        match_stats = downloader.get_career_stats(player_id, format_str)
        print(f"Match Stats for {player_id} in {format_str}:")
        print(match_stats)
        return match_stats
    except Exception as e:
        print(f"Error fetching match stats for {player_id} in {format_str}: {e}")
        return None
    
# Function to fetch venue statistics using a separate tab in Playwright
def get_venue_stats(context, venue_id):
    stats = {
        "matches_played": "N/A",
        "total_runs": "N/A",
        "total_wickets": "N/A",
        "balls_bowled": "N/A",
        "average_runs_per_wicket": "N/A",
        "average_runs_per_over": "N/A"
    }
    
    venue_url = f"https://stats.espncricinfo.com/ci/engine/ground/{venue_id}.html?class=6;template=results;type=aggregate"

    # Open a new tab to fetch venue stats
    venue_page = context.new_page()
    venue_page.goto(venue_url, timeout=60000)

    # Extract page content
    page_content = venue_page.content()

    # Parse using BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Locate the correct table with caption "Overall figures"
    overall_table = None
    for table in soup.find_all("table", class_="engineTable"):
        caption = table.find("caption")
        if caption and "Overall figures" in caption.text:
            overall_table = table
            break

    if overall_table:
        # Locate the row containing the "overall" figures
        data_row = overall_table.find("tr", class_="data1")
        if data_row:
            cells = data_row.find_all("td")
            if len(cells) >= 13:
                stats["matches_played"] = cells[2].text.strip()
                stats["total_runs"] = cells[8].text.strip()
                stats["total_wickets"] = cells[9].text.strip()
                stats["balls_bowled"] = cells[10].text.strip()
                stats["average_runs_per_wicket"] = cells[11].text.strip()
                stats["average_runs_per_over"] = cells[12].text.strip()

    venue_page.close()  # Close the tab after fetching stats
    print(f"Fetched Venue Stats: {stats}")
    return stats

def process_csv_data(match_dir):
    """Process and save corrected data with proper innings separation."""
    input_file = os.path.join(match_dir, 'all_overs_data.csv')
    corrected_file = os.path.join(match_dir, 'corrected_data.csv')
    
    corrected_data = []
    
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # Extract over details
            over_match = re.match(r'Over (\d+).*?([A-Z]+) (\d+)/(\d+)', row['over_details'])
            if over_match:
                over_num, team, runs, wickets = over_match.groups()
                
                # Extract forecasts and probabilities
                forecasts = row['live_forecasts'].split(';')
                win_prob_str = row['win_probabilities']
                fav_team = ''
                win_prob = 0.0

                if win_prob_str:
                    # Split on the special character (�)
                    prob_parts = win_prob_str.split('\xa0')
                    if len(prob_parts) == 2:
                        fav_team = prob_parts[0]
                        # Remove % and convert to float
                        win_prob = float(prob_parts[1].replace('%', ''))
                
                # Extract projected score if available
                projected_score = None
                for forecast in forecasts:
                    if 'Live Forecast:' in forecast:
                        score_match = re.search(r'Live Forecast: \w+ (\d+)', forecast)
                        if score_match:
                            projected_score = score_match.group(1)
                
                # Extract required runs and balls if available
                required_info = None
                for forecast in forecasts:
                    if 'needed' in forecast:
                        req_match = re.search(r'needed (\d+) runs from (\d+) balls', forecast)
                        if req_match:
                            required_info = f"{req_match.group(1)} from {req_match.group(2)}"
                
                # Calculate CRR and RRR
                balls_bowled = int(over_num) * 6
                crr = format(float(runs) / (balls_bowled/6), '.2f') if balls_bowled > 0 else '0.00'
                
                data = {
                    "over": over_num,
                    "innings": team,
                    "score": f"{runs}/{wickets}",
                    "projected_score": projected_score if projected_score else "N/A",
                    "required": required_info if required_info else "N/A",
                    "CRR": crr,
                    "RRR": "N/A",  # Will be calculated if required_info exists
                    "fav_team": fav_team,
                    "win_probability": win_prob,
                    "venue": row["venue"],
                    "matches_played": row["matches_played"],
                    "total_runs": row["total_runs"],
                    "total_wickets": row["total_wickets"],
                    "balls_bowled": row["balls_bowled"],
                    "average_runs_per_wicket": row["average_runs_per_wicket"],
                    "average_runs_per_over": row["average_runs_per_over"]
                }
                
                # Calculate RRR if required info exists
                if required_info:
                    req_runs, req_balls = map(int, re.findall(r'\d+', required_info))
                    rrr = format(float(req_runs) / (req_balls/6), '.2f')
                    data["RRR"] = rrr
                
                corrected_data.append(data)

    # Write corrected data
    if corrected_data:
        fieldnames = [
            "over", "innings", "score", "projected_score", "required", "CRR", "RRR",
            "fav_team", "win_probability",
            "venue", "matches_played", "total_runs", "total_wickets", "balls_bowled",
            "average_runs_per_wicket", "average_runs_per_over"
        ]
        
        with open(corrected_file, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corrected_data)
        
        print(f"Corrected data has been written to {corrected_file}")
    else:
        print("No data to write")

def extract_match_info(page):
    """
    Extract simplified match information focusing on winner and toss details
    """
    match_info = {
        "winner": "",
        "toss_winner": "",
        "toss_decision": "",
        "team1": "",
        "team1_score": "",
        "team2": "",
        "team2_score": "",
    }
    
    try:
        # Extract match result first (just the winner)
        result_selector = 'p.ds-text-tight-s.ds-font-medium span'
        result_element = page.locator(result_selector)
        if result_element.count() > 0:
            result_text = result_element.first.inner_text().strip()
            # Extract just the team name from "CSK won by 5 wickets (with 4 balls remaining)"
            winner_match = re.match(r'(\w+)\s+won\s+by', result_text)
            if winner_match:
                match_info["winner"] = winner_match.group(1)
            else:
                # Handle other result formats like "Match tied" or "No result"
                match_info["winner"] = result_text
        
        # Extract team names and scores using JavaScript for more reliable extraction
        team_data = page.evaluate('''() => {
            // Get all elements that might contain team information and take just the last 2
            const allTeamScores = Array.from(document.querySelectorAll('.ci-team-score, .ds-flex.ds-justify-between.ds-items-center'));
            const teamScores = allTeamScores.slice(-2); // Get only the last 2 elements
            
            return teamScores.map(element => {
                // Look for team name - could be in various elements
                let nameElement = element.querySelector('span.ds-text-tight-l, span.ds-block.ds-truncate');
                // Look for score - typically in a strong element
                let scoreElement = element.querySelector('strong, div.ds-text-compact-m strong');
                
                return {
                    name: nameElement ? nameElement.textContent.trim() : "",
                    score: scoreElement ? scoreElement.textContent.trim() : ""
                };
            }).filter(item => item.name && item.score); // Filter out empty entries
        }''')
        
        # Apply the extracted team data
        if len(team_data) >= 1:
            match_info["team1"] = team_data[0]["name"]
            match_info["team1_score"] = team_data[0]["score"]
        
        if len(team_data) >= 2:
            match_info["team2"] = team_data[1]["name"]
            match_info["team2_score"] = team_data[1]["score"]
        
        # Extract toss information
        toss_row = page.locator('tr td:has-text("Toss") + td')
        if toss_row.count() > 0:
            toss_text = toss_row.inner_text().strip()
            toss_parts = toss_text.split(', elected to ')
            if len(toss_parts) >= 2:
                match_info["toss_winner"] = toss_parts[0].strip()
                match_info["toss_decision"] = toss_parts[1].strip()
            
    except Exception as e:
        print(f"Error extracting match info: {e}")
        
        # Fallback methods if the above fails
        try:
            # Alternative method for match result
            result_text = page.evaluate('''() => {
                const resultElement = document.querySelector('p.ds-text-tight-s.ds-font-medium.ds-truncate.ds-text-typo span');
                return resultElement ? resultElement.textContent.trim() : '';
            }''')
            
            if result_text:
                winner_match = re.match(r'(\w+)\s+won\s+by', result_text)
                if winner_match:
                    match_info["winner"] = winner_match.group(1)
                else:
                    match_info["winner"] = result_text
                    
            # Alternative method for team names and scores
            team_data = page.evaluate('''() => {
                // Get all elements that might contain team information and take just the last 2
                const allTeamScores = Array.from(document.querySelectorAll('.ci-team-score, .ds-flex.ds-justify-between.ds-items-center'));
                const teamScores = allTeamScores.slice(-2); // Get only the last 2 elements
                
                return teamScores.map(element => {
                    // Look for team name - could be in various elements
                    let nameElement = element.querySelector('span.ds-text-tight-l, span.ds-block.ds-truncate');
                    // Look for score - typically in a strong element
                    let scoreElement = element.querySelector('strong, div.ds-text-compact-m strong');
                    
                    return {
                        name: nameElement ? nameElement.textContent.trim() : "",
                        score: scoreElement ? scoreElement.textContent.trim() : ""
                    };
                }).filter(item => item.name && item.score); // Filter out empty entries
            }''')
            
            if team_data.length >= 1:
                match_info["team1"] = team_data[0]["name"]
                match_info["team1_score"] = team_data[0]["score"]
            
            if team_data.length >= 2:
                match_info["team2"] = team_data[1]["name"]
                match_info["team2_score"] = team_data[1]["score"]
            
        except Exception as e2:
            print(f"Alternative extraction also failed: {e2}")
    
    return match_info

# Function to scrape the data
def scrape_all_overs_data(url):  # Add url parameter
    """
    Scrape match data from a specific URL.
    Args:
        url (str): The ESPN Cricinfo match URL to scrape
    Returns:
        tuple: (match_dir, match_id)
    """
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Use the provided URL instead of hardcoded one
        season_folder, match_id = get_match_info_from_url(url)
        
        # Create directory structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        match_dir = os.path.join(current_dir, season_folder, match_id, "ball_by_ball")
        ensure_directory_exists(match_dir)

        page.goto(url)

        # Wait explicitly for 30 seconds to ensure everything loads
        page.wait_for_timeout(30000)
        
        # Extract venue details using corrected selector
        venue_selector = 'a[href*="cricket-grounds"]'  # Locate link containing the stadium name
        venue_element = page.locator(venue_selector)
        venue_info = venue_element.inner_text().strip()  # Get venue name
        venue_link = venue_element.get_attribute("href")  # Get href link

        # Extract venue ID from the link
        venue_id_match = re.search(r'(\d+)$', venue_link)
        venue_id = venue_id_match.group(1) if venue_id_match else None
        print(f"Venue: {venue_info}, ID: {venue_id}")
        

        # Fetch venue stats using a separate tab
        venue_stats = get_venue_stats(context, venue_id) if venue_id else {
            "matches_played": "N/A",
            "total_runs": "N/A",
            "total_wickets": "N/A",
            "balls_bowled": "N/A",
            "average_runs_per_wicket": "N/A",
            "average_runs_per_over": "N/A"
        }
        
        # Wait for the slider element to appear
        slider_selector = "div.ds-absolute.ds-w-0\\.5.ds-h-\\[200px\\]"
        page.wait_for_selector(slider_selector, timeout=30000)
        slider = page.locator(slider_selector)

        # Get the bounding box of the graph to calculate drag positions
        graph_selector = "div.ds-w-\\[300px\\].ds-relative"
        graph_box = page.locator(graph_selector).bounding_box()
        if not graph_box:
            print("Graph bounding box not found!")
            return

        graph_x_start = graph_box["x"]
        graph_x_end = graph_x_start + graph_box["width"]
        graph_y_center = graph_box["y"] + graph_box["height"] / 2

        # Adjust steps to match every over (20 overs with fractional precision)
        total_overs = 20
        steps_per_over = 10
        steps = total_overs * steps_per_over
        step_size = (graph_x_end - graph_x_start) / steps

        # Data collection
        all_overs_data = []
        seen_overs = set()

        # Drag the slider across the graph
        for step in range(steps + 1):
            current_x = graph_x_start + step * step_size

            try:
                slider.hover()
                page.mouse.down()
                page.mouse.move(current_x, graph_y_center)
                page.mouse.up()

                over_details = page.locator("div.ds-px-4.ds-pb-3 p").inner_text()

                # Avoid duplicates by checking if this over was already captured
                if (over_details) not in seen_overs:
                    seen_overs.add((over_details))
                    all_live_forecasts = page.locator("div.ds-px-4.ds-pb-3 span").all_inner_texts()
                    win_probabilities = page.locator("div.ds-text-tight-s.ds-font-bold.ds-ml-1").all_inner_texts()

                    # Store the data
                    data = {
                        "over_details": over_details,
                        "live_forecasts": ";".join(all_live_forecasts),  # Join lists with semicolons
                        "win_probabilities": ";".join(win_probabilities),
                        "venue": venue_info,
                        "matches_played": venue_stats["matches_played"],
                        "total_runs": venue_stats["total_runs"],
                        "total_wickets": venue_stats["total_wickets"],
                        "balls_bowled": venue_stats["balls_bowled"],
                        "average_runs_per_wicket": venue_stats["average_runs_per_wicket"],
                        "average_runs_per_over": venue_stats["average_runs_per_over"]
                    }
                    all_overs_data.append(data)

                    print(f"Step {step}: {data}")
            except Exception as e:
                print(f"Error at step {step}: {e}")

        # Navigate to full scorecard URL first to get match information
        scorecard_url = url
        if "ball-by-ball-commentary" in url:
            scorecard_url = url.replace("ball-by-ball-commentary", "full-scorecard")
        
        # Create a new page for the scorecard
        scorecard_page = context.new_page()
        scorecard_page.goto(scorecard_url, timeout=60000)
        scorecard_page.wait_for_timeout(5000)  # Wait for the page to load
        
        # Extract match information
        match_info = extract_match_info(scorecard_page)
        print(f"Extracted match info: {match_info}")
        
        # Save match info to a separate file
        match_info_file = os.path.join(match_dir, 'match_info.csv')
        with open(match_info_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=match_info.keys())
            writer.writeheader()
            writer.writerow(match_info)
        
        print(f"Match information saved to {match_info_file}")
        
        # Close the scorecard page and continue with the rest of the scraping
        scorecard_page.close()

        # Write the scraped data to CSV in match directory
        output_file = os.path.join(match_dir, 'all_overs_data.csv')
        with open(output_file, 'w', newline='') as outfile:
            fieldnames = [
                "over_details", "live_forecasts", "win_probabilities", "venue", "matches_played",
                "total_runs", "total_wickets", "balls_bowled", "average_runs_per_wicket", "average_runs_per_over"
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in all_overs_data:
                writer.writerow(data)

        print(f"Scraped data saved to {output_file}")
        
        team_sections = fetch_team_names_and_player_sections(page)
        player_links = fetch_player_links_with_teams(team_sections)
        print(f"Found {len(player_links)} players. Fetching their stats...")

        # Pass the match_id here
        fetch_and_save_player_stats(player_links,page, match_id=match_dir)

        browser.close()
        return match_dir, match_id  # Return match_dir and match_id for use in main()

def save_to_csv(data_frame, filename):
    data_frame.to_csv(filename, index=False)
    print(f"Saved data to {filename}")
    
def process_batting_stats(player_name, player_role,player_team, batting_df):
    """Process batting statistics for a player."""
    stats = []
    for _, row in batting_df.iterrows():
        stats.append({
            "player_name": player_name,
            "player_role": player_role,
            "player_team": player_team,
            "matches": row.get("Matches", ""),
            "innings": row.get("Innings", ""),
            "not_outs": row.get("Not Outs", ""),
            "runs": row.get("Runs", ""),
            "balls": row.get("Balls Faced", ""),
            "fours": row.get("Fours", ""),
            "sixes": row.get("Sixes", ""),
            "average": row.get("Average", ""),
            "strike_rate": row.get("Strike Rate", "")
        })
    return stats

def process_bowling_stats(player_name, player_role,player_team, bowling_df):
    """Process bowling statistics for a player."""
    stats = []
    for _, row in bowling_df.iterrows():
        stats.append({
            "player_name": player_name,
            "player_role": player_role,
            "player_team": player_team,
            "matches": row.get("Matches", ""),
            "innings": row.get("Innings", ""),
            "overs": row.get("Overs", ""),
            "maidens": row.get("Maidens", ""),
            "runs_conceded": row.get("Runs Conceded", ""),
            "wickets": row.get("Wickets", ""),
            "average": row.get("Average", ""),
            "economy": row.get("Economy", ""),
            "strike_rate": row.get("Strike Rate", "")
        })
    return stats

def save_team_stats(team_stats, match_id=None):
    """Save team statistics to CSV files."""
    for team_name, stats in team_stats.items():
        if stats["batting"]:
            filename = f"{team_name}_batting_stats_{match_id}.csv" if match_id else f"{team_name}_batting_stats.csv"
            save_to_csv_players(stats["batting"], filename, is_batting=True, match_id=match_id)
        if stats["bowling"]:
            filename = f"{team_name}_bowling_stats_{match_id}.csv" if match_id else f"{team_name}_bowling_stats.csv"
            save_to_csv_players(stats["bowling"], filename, is_batting=False, match_id=match_id)

def get_match_id_from_url(url):
    """Extract match information from URL."""
    # Match pattern: ipl-{season}/{teams}-{match-number}
    match = re.search(r'ipl-(\d+-\d+)/(.+?)-(\d+(?:st|nd|rd|th)-match)-(\d+)', url)
    if match:
        season, teams, match_type, match_id = match.groups()
        return f"{season}_{teams}_{match_id}"
    return "default_match"

def get_match_info_from_url(url):
    """
    Extracts the season folder and match ID from the given URL.
    Example URL format:
    https://www.espncricinfo.com/series/ipl-2020-21-1210595/mumbai-indians-vs-chennai-super-kings-1st-match-1216492/...
    
    Returns:
        tuple: (season_folder, match_id)
        - season_folder: e.g. 'ipl-2020-21-1210595'
        - match_id: e.g. '1216492'
    """
    # Use regex to extract season folder and match ID
    match = re.search(r'/series/([^/]+)/([^/]+)/', url)
    if match:
        season_folder = match.group(1)  # Gets 'ipl-2020-21-1210595'
        match_id = match.group(2)       # Gets 'mumbai-indians-vs-chennai-super-kings-1st-match-1216492'
        
        # Extract just the numeric match ID from the end
        match_id_num = re.search(r'(\d+)$', match_id)
        if match_id_num:
            match_id = match_id_num.group(1)
        
        return season_folder, match_id
    return "default_season", "default_match"

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

import time

def wait_for_new_ball_update(page):
    """
    Continuously monitors the first commentary div and waits for a new ball update.
    Returns the updated ball commentary once detected.
    """
    print("⏳ Waiting for the next ball to be bowled...")

    # Locate the first commentary div
    commentary_divs = page.locator("xpath=//div[contains(@class, 'ds-hover-parent ds-relative')]")

    # Get the initial first div text (if available)
    initial_text = commentary_divs.nth(0).inner_text().strip() if commentary_divs.count() > 0 else ""
    
    while True:
        time.sleep(1)  # Poll every second (adjust if needed)

        # Fetch updated first div text
        latest_text = commentary_divs.nth(0).inner_text().strip() if commentary_divs.count() > 0 else ""

        # If new text appears, return it
        if latest_text and latest_text != initial_text:
            print(f"🎯 New Ball Update Detected: {latest_text}")
            return latest_text  # Return the latest ball commentary


def main(url=None):
    """Main function to run the scraper and process the data."""
    if not url:
        # Default URL if none is provided
        url = "https://www.espncricinfo.com/series/ipl-2020-21-1210595/mumbai-indians-vs-chennai-super-kings-1st-match-1216492/full-scorecard"
    
    # Get match info and create directory structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    season_folder, match_id = get_match_info_from_url(url)
    match_dir = os.path.join(current_dir, season_folder, match_id, "ball_by_ball")
    ensure_directory_exists(match_dir)
    
    try:
        # First, import and run the ball-by-ball scraper to create innings summary files
        print("Running ball-by-ball scraper first...")
        from espnscraper_ballbyball import main as run_ballbyball
        run_ballbyball(url)
        
        # Then scrape the scorecard data with URL parameter
        print("Starting scorecard data scraping...")
        match_dir, match_id = scrape_all_overs_data(url)  # Pass url here
        
        print("Processing CSV data...")
        process_csv_data(match_dir)
        
        # Import and use the data merger
        from data_merger import merge_match_data, generate_team_summary
        
        try:
            # Both First_innings_summary.csv and Second_innings_summary.csv should now exist in match_dir
            merged_data = merge_match_data(match_dir, match_dir)
            
            # Generate team summary
            team_summary = generate_team_summary(merged_data)
            
            # Save team summary
            summary_file = os.path.join(match_dir, 'team_summary.csv')
            team_summary.to_csv(summary_file, index=False)
            print(f"Team summary saved to: {summary_file}")
            
        except FileNotFoundError as e:
            print(f"Error: Required files not found - {e}")
            print(f"Looking in directory: {match_dir}")
            print("Make sure both First_innings_summary.csv and Second_innings_summary.csv exist in this directory")
        except Exception as e:
            print(f"Error during data merging: {e}")
        
        print(f"All operations completed. Data saved in: {match_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing match: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if URL is provided as command line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"Processing URL: {url}")
        main(url)
    else:
        main()  # Use default URL

