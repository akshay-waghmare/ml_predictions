import sys
import time
import csv
import re
import os
from playwright.sync_api import sync_playwright

def parse_batsman_stats(stat_str):
    """Parse batsman statistics string into components"""
    match = re.match(r'(\d+) \((\d+)b(?: (\d+)x4)?(?: (\d+)x6)?\)', stat_str)
    if match:
        return {
            'runs': int(match.group(1)),
            'balls': int(match.group(2)),
            'fours': int(match.group(3)) if match.group(3) else 0,
            'sixes': int(match.group(4)) if match.group(4) else 0
        }
    return None

def parse_bowler_stats(stat_str):
    """Parse bowler statistics string into components"""
    match = re.match(r'(\d+)-(\d+)-(\d+)-(\d+)', stat_str)
    if match:
        return {
            'overs': float(match.group(1)),
            'maidens': int(match.group(2)),
            'runs': int(match.group(3)),
            'wickets': int(match.group(4))
        }
    return None

def extract_ball_runs(runs_or_event):
    """Extract runs from the ball indicator"""
    if runs_or_event.isdigit():
        return int(runs_or_event)
    elif runs_or_event == "W":
        return 0
    elif runs_or_event == "•":
        return 0
    else:
        # For extras like "1wd", "4lb", etc.
        match = re.match(r'(\d+)', runs_or_event)
        return int(match.group(1)) if match else 0

def parse_ball_details(ball_number, runs_or_event, short_commentary):
    """Parse details from a single ball commentary"""
    over_part = float(ball_number)
    over_number = int(over_part)
    ball_in_over = int(round((over_part - over_number) * 10))
    
    runs = extract_ball_runs(runs_or_event)
    is_wicket = runs_or_event == "W"
    is_dot = runs_or_event == "•"
    is_boundary = runs in [4, 6]
    
    # Determine extras
    is_wide = "wide" in short_commentary.lower()
    is_no_ball = "no ball" in short_commentary.lower()
    is_bye = "bye" in short_commentary.lower() or "leg bye" in short_commentary.lower()
    is_extra = is_wide or is_no_ball or is_bye
    
    # Extract bowler and batsman names
    bowler_match = re.search(r'([A-Za-z\s]+) to ([A-Za-z\s]+)', short_commentary)
    if (bowler_match):
        bowler_name = bowler_match.group(1).strip()
        batsman_name = bowler_match.group(2).strip()
    else:
        bowler_name = "Unknown"
        batsman_name = "Unknown"
    
    # Extract out batsman for wickets
    out_batsman = batsman_name
    if is_wicket and "c " in short_commentary:
        # For caught dismissals, use regex to get the actual batsman
        out_match = re.search(r'([A-Za-z\s]+) c [A-Za-z\s]+ b', short_commentary)
        if out_match:
            out_batsman = out_match.group(1).strip()
    
    return {
        'over_number': over_number,
        'ball_in_over': ball_in_over,
        'full_over': over_part,
        'batsman_name': batsman_name,
        'bowler_name': bowler_name,
        'striker_batsman': batsman_name,  # Add striker_batsman field explicitly
        'runs': runs,
        'is_wicket': is_wicket,
        'is_dot': is_dot,
        'is_boundary': is_boundary,
        'is_extra': is_extra,
        'is_wide': is_wide,
        'is_no_ball': is_no_ball,
        'is_bye': is_bye,
        'out_batsman': out_batsman if is_wicket else None,
        'commentary': short_commentary
    }

def extract_new_batsman_from_commentary(commentary_text):
    """Extract new batsman name from commentary when a wicket falls"""
    # Try to find patterns like "New batsman is X" or "Y comes to the crease"
    patterns = [
        r'([A-Za-z\s]+) comes to the crease',
        r'New batsman is ([A-Za-z\s]+)',
        r'([A-Za-z\s]+) walks in'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, commentary_text)
        if match:
            return match.group(1).strip()
    
    return None

def scrape_innings(page, innings_label, output_file):
    """
    Scrapes the commentary data for the current innings with ball-by-ball stats.
    Maintains running scorecards for batsmen and bowlers.
    """
    
    batting_team = get_current_batting_team(page)
    print(f"Scraping commentary for {batting_team} ({innings_label} innings)...")
    
    # Verify we have commentary data before proceeding
    commentary_locator = page.locator("xpath=//div[contains(@class, 'lg:hover:ds-bg-ui-fill-translucent ds-hover-parent ds-relative')]")
    initial_count = commentary_locator.count()
    print(f"Initial commentary count: {initial_count}")
    
    if initial_count == 0:
        print("WARNING: No commentary data found on current page.")
        # Try to find any text explaining why data might be missing
        try:
            page_content = page.content()
            if "not available" in page_content.lower() or "no content" in page_content.lower():
                print("Page indicates commentary is not available.")
            page.screenshot(path=f"{innings_label}_error_screenshot.png")
            print(f"Screenshot saved to {innings_label}_error_screenshot.png")
        except Exception as e:
            print(f"Error capturing debug info: {e}")
        return {"batting_team": batting_team, "error": "No commentary data found"}
    
    # Reset page scroll position to ensure we start from the top
    page.evaluate("window.scrollTo(0, 0);")
    time.sleep(1)
    
    # Scroll to load all commentary content
    retry_scroll_up = 0
    max_scroll_attempts = 15  # Increase from 10 to 15 for more thorough loading
    target_div_count = 120  # Target number of commentary divs to collect
    
    while True:
        commentary_locator = page.locator("xpath=//div[contains(@class, 'lg:hover:ds-bg-ui-fill-translucent ds-hover-parent ds-relative')]")
        previous_div_count = commentary_locator.count()
        print(f"Current commentary div count: {previous_div_count}")
        
        # Stop if we've reached the target number of commentary divs
        if previous_div_count >= target_div_count:
            print(f"Reached target of {target_div_count} commentary divs. Stopping scroll.")
            break
            
        page.mouse.wheel(0, 5000)
        time.sleep(2)  # Increase wait time to ensure content loads properly
        
        new_div_count = commentary_locator.count()
        print(f"New commentary div count after scroll: {new_div_count}")
        
        if new_div_count == previous_div_count:
            retry_scroll_up += 1
            print(f"No new content, trying scroll combination (attempt {retry_scroll_up}/{max_scroll_attempts})")
            
            # More aggressive scroll recovery - try both up and down in different patterns
            for i in range(3):
                page.mouse.wheel(0, -1000 * (i + 1))  # Scroll up with increasing distance
                time.sleep(1)
            page.mouse.wheel(0, 2000)  # Then scroll down further
            time.sleep(1)
            
            if retry_scroll_up >= max_scroll_attempts:
                print("Maximum scroll attempts reached. Using available commentary data.")
                break
        else:
            print(f"Found {new_div_count - previous_div_count} new commentary items")
            retry_scroll_up = 0

    # Collect all commentary divs
    commentary_divs = page.locator(
        "xpath=//div[contains(@class, 'lg:hover:ds-bg-ui-fill-translucent ds-hover-parent ds-relative')]"
    )
    
    final_div_count = commentary_divs.count()
    print(f"Final commentary div count: {final_div_count}")
    
    if final_div_count == 0:
        print(f"ERROR: Failed to find any commentary divs for {innings_label} innings")
        page.screenshot(path=f"{innings_label}_no_commentary_error.png")
        return {"batting_team": batting_team, "error": "No commentary data found after scrolling"}
    
    # Prepare data structures for ball-by-ball data and player stats
    all_balls_data = []
    batsmen_stats = {}  # Dictionary to track batsman stats
    bowler_stats = {}   # Dictionary to track bowler stats
    
    # Track current batsmen at crease
    batsmen_at_crease = []
    
    # Track current over and previous over's bowlers
    current_over_bowler = None
    previous_over_bowler = None
    last_over_number = -1
    
    # Process commentary divs in reverse order (oldest first)
    num_commentary = commentary_divs.count()
    for i in range(num_commentary - 1, -1, -1):
        try:
            div = commentary_divs.nth(i)
            
            # Extract ball information
            ball_number = div.locator(
                "xpath=.//span[contains(@class, 'ds-text-tight-s') and contains(@class, 'ds-font-regular') and contains(@class, 'ds-text-typo-mid1')]"
            ).inner_text().strip()
            
            # Skip if this is not a ball (e.g., a between-overs comment)
            if not re.match(r'^\d+\.\d+$', ball_number):
                continue
                
            runs_or_event = div.locator(
                "xpath=.//div[contains(@class, 'ds-text-tight-m') and contains(@class, 'ds-font-bold')]/span"
            ).inner_text().strip()
            
            short_commentary = div.locator(
                "xpath=.//div[contains(@class, 'ds-leading-')]"
            ).evaluate_all("nodes => nodes.map(node => node.textContent.trim()).join(' ')")
            
            detailed_commentary = div.locator("xpath=.//p[contains(@class, 'ci-html-content')]").inner_text().strip()
            
            # Parse the ball details
            ball_details = parse_ball_details(ball_number, runs_or_event, short_commentary)
            
            # Update bowler stats
            bowler_name = ball_details['bowler_name']
            
            # Check if over has changed
            current_over = ball_details['over_number']
            if current_over != last_over_number:
                if current_over_bowler:
                    # Current over has changed, previous over's bowler becomes last over bowler
                    previous_over_bowler = current_over_bowler
                # Set current over's bowler
                current_over_bowler = bowler_name
                last_over_number = current_over
            
            if bowler_name not in bowler_stats:
                bowler_stats[bowler_name] = {
                    'name': bowler_name,
                    'overs': 0,
                    'maidens': 0,
                    'runs_conceded': 0,
                    'wickets': 0,
                    'balls_bowled': 0,
                    'maiden_in_progress': True  # Track if the current over is a potential maiden
                }
            
            # Update bowler ball count
            if not ball_details['is_wide']:  # Wides don't count as balls bowled
                bowler_stats[bowler_name]['balls_bowled'] += 1
                
                # Update overs count
                if bowler_stats[bowler_name]['balls_bowled'] % 6 == 0:
                    bowler_stats[bowler_name]['overs'] = bowler_stats[bowler_name]['balls_bowled'] / 6
                    
                    # Check if the completed over was a maiden
                    if bowler_stats[bowler_name]['maiden_in_progress']:
                        bowler_stats[bowler_name]['maidens'] += 1
                        
                    # Reset maiden tracking for next over
                    bowler_stats[bowler_name]['maiden_in_progress'] = True
            
            # Update runs conceded and wickets
            if not ball_details['is_bye']:  # Byes aren't charged to the bowler
                bowler_stats[bowler_name]['runs_conceded'] += ball_details['runs']
                
            # Check if maiden is still possible
            if ball_details['runs'] > 0 or ball_details['is_wide'] or ball_details['is_no_ball']:
                bowler_stats[bowler_name]['maiden_in_progress'] = False
                
            # Update wickets
            if ball_details['is_wicket']:
                # Only count if it's not a run out (attributable to the bowler)
                if "run out" not in short_commentary.lower():
                    bowler_stats[bowler_name]['wickets'] += 1
            
            # Update batsman stats
            batsman_name = ball_details['batsman_name']
            
            # Initialize batsman stats if first appearance
            if batsman_name not in batsmen_stats:
                batsmen_stats[batsman_name] = {
                    'name': batsman_name,
                    'runs': 0,
                    'balls_faced': 0,
                    'fours': 0,
                    'sixes': 0,
                    'out': False
                }
                
                # Add to batsmen at crease if not already there
                if batsman_name not in batsmen_at_crease:
                    batsmen_at_crease.append(batsman_name)
                    # Keep only most recent two batsmen
                    if len(batsmen_at_crease) > 2:
                        batsmen_at_crease.pop(0)
            
            # Update batsman stats (don't count wides as balls faced)
            if not ball_details['is_wide']:
                batsmen_stats[batsman_name]['balls_faced'] += 1
            
            batsmen_stats[batsman_name]['runs'] += ball_details['runs']
            
            if ball_details['is_boundary']:
                if ball_details['runs'] == 4:
                    batsmen_stats[batsman_name]['fours'] += 1
                elif ball_details['runs'] == 6:
                    batsmen_stats[batsman_name]['sixes'] += 1
            
            # Handle wicket - mark batsman as out
            if ball_details['is_wicket']:
                # Use out_batsman in case it's different from the facing batsman (like run out)
                out_batsman = ball_details['out_batsman'] 
                if out_batsman in batsmen_stats:
                    batsmen_stats[out_batsman]['out'] = True
                
                # Remove out batsman from crease
                if out_batsman in batsmen_at_crease:
                    batsmen_at_crease.remove(out_batsman)
                
                # Try to find new batsman from commentary
                new_batsman = extract_new_batsman_from_commentary(detailed_commentary)
                if new_batsman and new_batsman not in batsmen_stats:
                    batsmen_stats[new_batsman] = {
                        'name': new_batsman,
                        'runs': 0,
                        'balls_faced': 0,
                        'fours': 0,
                        'sixes': 0,
                        'out': False
                    }
                    batsmen_at_crease.append(new_batsman)
            
            # Create a dict with current state for this ball
            ball_state = {
                "batting_team": batting_team,
                "over_number": ball_details['over_number'],
                "ball_number": ball_details['full_over'],
                "runs_scored": ball_details['runs'],
                "boundaries": 1 if ball_details['is_boundary'] else 0,
                "dot_balls": 1 if ball_details['is_dot'] else 0,
                "wickets": 1 if ball_details['is_wicket'] else 0,
                "extras": 1 if ball_details['is_extra'] else 0,
                "striker_batsman": ball_details['striker_batsman'],  # Add striker_batsman to ball_state
            }
            
            # Add the current batsmen stats
            for idx, batsman in enumerate(batsmen_at_crease[:2], 1):
                if batsman in batsmen_stats:
                    ball_state[f"batsman{idx}_name"] = batsman
                    ball_state[f"batsman{idx}_runs"] = batsmen_stats[batsman]['runs']
                    ball_state[f"batsman{idx}_balls_faced"] = batsmen_stats[batsman]['balls_faced']
                    ball_state[f"batsman{idx}_fours"] = batsmen_stats[batsman]['fours']
                    ball_state[f"batsman{idx}_sixes"] = batsmen_stats[batsman]['sixes']
            
            # Ensure both batsman1 and batsman2 fields exist
            for idx in [1, 2]:
                if f"batsman{idx}_name" not in ball_state:
                    ball_state[f"batsman{idx}_name"] = ""
                    ball_state[f"batsman{idx}_runs"] = 0
                    ball_state[f"batsman{idx}_balls_faced"] = 0
                    ball_state[f"batsman{idx}_fours"] = 0
                    ball_state[f"batsman{idx}_sixes"] = 0
            
            # Add bowler stats - use the tracked bowlers instead of sorting by balls bowled
            # Current over's bowler
            if current_over_bowler and current_over_bowler in bowler_stats:
                bowler = bowler_stats[current_over_bowler]
                ball_state["bowler1_name"] = bowler['name']
                ball_state["bowler1_overs_bowled"] = bowler['overs']
                ball_state["bowler1_maidens_bowled"] = bowler['maidens']
                ball_state["bowler1_runs_conceded"] = bowler['runs_conceded']
                ball_state["bowler1_wickets_taken"] = bowler['wickets']
            else:
                ball_state["bowler1_name"] = ""
                ball_state["bowler1_overs_bowled"] = 0
                ball_state["bowler1_maidens_bowled"] = 0
                ball_state["bowler1_runs_conceded"] = 0
                ball_state["bowler1_wickets_taken"] = 0
                
            # Previous over's bowler
            if previous_over_bowler and previous_over_bowler in bowler_stats:
                bowler = bowler_stats[previous_over_bowler]
                ball_state["bowler2_name"] = bowler['name']
                ball_state["bowler2_overs_bowled"] = bowler['overs']
                ball_state["bowler2_maidens_bowled"] = bowler['maidens']
                ball_state["bowler2_runs_conceded"] = bowler['runs_conceded']
                ball_state["bowler2_wickets_taken"] = bowler['wickets']
            else:
                ball_state["bowler2_name"] = ""
                ball_state["bowler2_overs_bowled"] = 0
                ball_state["bowler2_maidens_bowled"] = 0
                ball_state["bowler2_runs_conceded"] = 0
                ball_state["bowler2_wickets_taken"] = 0
            
            all_balls_data.append(ball_state)
            
        except Exception as e:
            print(f"Error processing ball data at index {i}: {e}")
    
    # Reverse the list to get chronological order (oldest to newest)
    all_balls_data.reverse()
    
    # Group by over for the final output
    over_summary_data = []
    current_over = None
    current_over_data = None
    
    for ball in all_balls_data:
        if current_over != ball['over_number']:
            # If we have a previous over data, save it
            if current_over_data:
                over_summary_data.append(current_over_data)
            
            # Create new over data
            current_over = ball['over_number']
            current_over_data = {
                "batting_team": batting_team,
                "over_number": current_over,
                "ball_number": f"{current_over}.1",  # Start of the over
                "runs_scored": 0,
                "boundaries": 0,
                "dot_balls": 0,
                "wickets": 0,
                "extras": 0,
                "striker_batsman": ball["striker_batsman"],  # Add striker_batsman to over summary
                # Copy latest batsmen and bowler data
                "batsman1_name": ball["batsman1_name"],
                "batsman1_runs": ball["batsman1_runs"],
                "batsman1_balls_faced": ball["batsman1_balls_faced"],
                "batsman1_fours": ball["batsman1_fours"],
                "batsman1_sixes": ball["batsman1_sixes"],
                "batsman2_name": ball["batsman2_name"],
                "batsman2_runs": ball["batsman2_runs"],
                "batsman2_balls_faced": ball["batsman2_balls_faced"],
                "batsman2_fours": ball["batsman2_fours"],
                "batsman2_sixes": ball["batsman2_sixes"],
                "bowler1_name": ball["bowler1_name"],
                "bowler1_overs_bowled": ball["bowler1_overs_bowled"],
                "bowler1_maidens_bowled": ball["bowler1_maidens_bowled"],
                "bowler1_runs_conceded": ball["bowler1_runs_conceded"],
                "bowler1_wickets_taken": ball["bowler1_wickets_taken"],
                "bowler2_name": ball["bowler2_name"],
                "bowler2_overs_bowled": ball["bowler2_overs_bowled"],
                "bowler2_maidens_bowled": ball["bowler2_maidens_bowled"],
                "bowler2_runs_conceded": ball["bowler2_runs_conceded"],
                "bowler2_wickets_taken": ball["bowler2_wickets_taken"]
            }
        
        # Accumulate stats for this over
        current_over_data["runs_scored"] += ball["runs_scored"]
        current_over_data["boundaries"] += ball["boundaries"]
        current_over_data["dot_balls"] += ball["dot_balls"]
        current_over_data["wickets"] += ball["wickets"]
        current_over_data["extras"] += ball["extras"]
        
        # Update with the latest batsman and bowler stats
        current_over_data["batsman1_name"] = ball["batsman1_name"]
        current_over_data["batsman1_runs"] = ball["batsman1_runs"]
        current_over_data["batsman1_balls_faced"] = ball["batsman1_balls_faced"]
        current_over_data["batsman1_fours"] = ball["batsman1_fours"]
        current_over_data["batsman1_sixes"] = ball["batsman1_sixes"]
        current_over_data["batsman2_name"] = ball["batsman2_name"]
        current_over_data["batsman2_runs"] = ball["batsman2_runs"]
        current_over_data["batsman2_balls_faced"] = ball["batsman2_balls_faced"]
        current_over_data["batsman2_fours"] = ball["batsman2_fours"]
        current_over_data["batsman2_sixes"] = ball["batsman2_sixes"]
        current_over_data["bowler1_name"] = ball["bowler1_name"]
        current_over_data["bowler1_overs_bowled"] = ball["bowler1_overs_bowled"]
        current_over_data["bowler1_maidens_bowled"] = ball["bowler1_maidens_bowled"]
        current_over_data["bowler1_runs_conceded"] = ball["bowler1_runs_conceded"]
        current_over_data["bowler1_wickets_taken"] = ball["bowler1_wickets_taken"]
        current_over_data["bowler2_name"] = ball["bowler2_name"]
        current_over_data["bowler2_overs_bowled"] = ball["bowler2_overs_bowled"]
        current_over_data["bowler2_maidens_bowled"] = ball["bowler2_maidens_bowled"]
        current_over_data["bowler2_runs_conceded"] = ball["bowler2_runs_conceded"]
        current_over_data["bowler2_wickets_taken"] = ball["bowler2_wickets_taken"]
    
    # Add the last over
    if current_over_data:
        over_summary_data.append(current_over_data)
    
    # Save both ball-by-ball data and over summary to separate files
    ball_by_ball_file = output_file.replace("_summary.csv", "_ball_by_ball.csv")
    
    # Define fieldnames for both files
    fieldnames = [
        "batting_team", "over_number", "ball_number", "runs_scored", "boundaries", 
        "dot_balls", "wickets", "extras", "striker_batsman",  # Add striker_batsman to fieldnames
        "batsman1_name", "batsman1_runs", "batsman1_balls_faced", "batsman1_fours", "batsman1_sixes",
        "batsman2_name", "batsman2_runs", "batsman2_balls_faced", "batsman2_fours", "batsman2_sixes",
        "bowler1_name", "bowler1_overs_bowled", "bowler1_maidens_bowled", "bowler1_runs_conceded", "bowler1_wickets_taken",
        "bowler2_name", "bowler2_overs_bowled", "bowler2_maidens_bowled", "bowler2_runs_conceded", "bowler2_wickets_taken"
    ]

    # Write ball-by-ball data
    with open(ball_by_ball_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_balls_data)
    
    print(f"Ball-by-ball data saved to {ball_by_ball_file}")
    
    # Write over summary data
    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(over_summary_data)
    
    print(f"Over summary data saved to {output_file}")
    
    return {
        "ball_by_ball_file": ball_by_ball_file,
        "over_summary_file": output_file,
        "batsmen_stats": batsmen_stats,
        "bowler_stats": bowler_stats
    }

def select_other_innings(page):
    """
    Clicks the innings dropdown, and selects the other innings.
    Returns True if successful, False otherwise.
    """
    print("Attempting to switch innings...")
    
    try:
        # Check if dropdown selector exists
        dropdown_button_selector = 'div.ds-flex.ds-items-center.ds-cursor-pointer'
        if not page.query_selector(dropdown_button_selector):
            print("Error: Innings dropdown not found")
            return False
            
        # Click the dropdown button
        page.wait_for_selector(dropdown_button_selector, timeout=5000)
        page.click(dropdown_button_selector)
        time.sleep(2)  # Wait for the dropdown list to appear
        
        # Debug information - check what options are available
        dropdown_options = page.locator("xpath=//ul[contains(@class, 'ds-flex')]/li")
        options_count = dropdown_options.count()
        print(f"Found {options_count} innings options in the dropdown")
        
        if options_count == 0:
            print("No innings options found in the dropdown")
            return False

        # Get all list items in the dropdown
        li_locator = page.locator("xpath=//ul[contains(@class, 'ds-flex')]/li")
        li_count = li_locator.count()
        target_option = None
        current_innings = None
        target_innings_text = None

        # Loop through each li element and select the one that is not currently selected
        for i in range(li_count):
            li_element = li_locator.nth(i)
            inner_div = li_element.locator("div.ds-cursor-pointer")
            class_attr = inner_div.get_attribute("class") or ""
            inner_text = inner_div.inner_text().strip()
            
            # Debug: print each option
            print(f"Option {i+1}: '{inner_text}' - Selected: {'ds-font-bold' in class_attr}")
            
            if "ds-font-bold" in class_attr:
                current_innings = inner_text
            else:
                target_option = inner_div
                target_innings_text = inner_text

        if target_option is not None:
            print(f"Currently selected: '{current_innings}'")
            print(f"Switching to: '{target_innings_text}'")
            
            # Click the target option
            target_option.click()
            
            # Don't try to verify with dropdown - just assume it worked and check content changes
            print("Clicked on target innings option. Waiting for page to update...")
            
            # Wait for the page to update - be more generous with time but avoid timeouts
            time.sleep(5)
            
            # Try to determine if switch worked by checking content changes
            # Rather than opening the dropdown again (which can cause timeouts),
            # check if the currently visible batting team changed
            
            # First try to see if there's an innings header showing the team
            try:
                # Look for team name in the page without using the dropdown
                visible_team = page.locator("h5.ds-text-title-xs span:not(.ds-opacity-50)").first.inner_text().strip()
                print(f"Current visible team in header: '{visible_team}'")
                
                # If we can see the target innings name in the header, it probably worked
                if target_innings_text and target_innings_text in visible_team:
                    print(f"Successfully switched to {target_innings_text} innings based on header.")
                    return True
            except Exception as e:
                print(f"Could not verify switch with header: {e}")
            
            # As a fallback, just assume it worked if we clicked successfully
            print("Assuming innings switch was successful based on click.")
            return True
        else:
            print("No alternative innings option found.")
            return False
    except Exception as e:
        print(f"Error while switching innings: {e}")
        return False

def get_current_batting_team(page):
    """
    Extract the name of the team currently batting from the dropdown or page header.
    """
    try:
        # Locate the dropdown and extract the team name from the <span>
        team_name = page.locator("div.ds-flex.ds-items-center.ds-cursor-pointer span.ds-text-tight-s.ds-font-regular").first.inner_text().strip()
        print(f"Currently batting team: {team_name}")
        return team_name
    except Exception as e:
        print(f"Error fetching the batting team: {e}")
        return "Unknown"

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
    if (match):
        season_folder = match.group(1)  # Gets 'ipl-2020-21-1210595'
        match_id = match.group(2)       # Gets 'mumbai-indians-vs-chennai-super-kings-1st-match-1216492'
        
        # Extract just the numeric match ID from the end
        match_id_num = re.search(r'(\d+)$', match_id)
        if match_id_num:
            match_id = match_id_num.group(1)
        
        return season_folder, match_id
    return "default_season", "default_match"

def main(url=None):
    """Modified main function to accept URL parameter"""
    if not url:
        # Get URL from command line arguments if not provided
        if len(sys.argv) != 2:
            print("Usage: python espnscraper_ballbyball.py <match_url>")
            sys.exit(1)
        url = sys.argv[1]
    
    # Convert scorecard URL to ball-by-ball URL if needed
    if "full-scorecard" in url:
        url = url.replace("full-scorecard", "ball-by-ball-commentary")
    elif "live-cricket-score" in url:
        url = url.replace("live-cricket-score", "ball-by-ball-commentary")

    with sync_playwright() as p:
        # Create a browser with a larger viewport to ensure all UI elements are visible
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}  # Larger viewport
        )
        page = context.new_page()

        # Enable console logs from the browser
        page.on("console", lambda msg: print(f"BROWSER LOG: {msg.text}"))

        # Navigate to the commentary page
        page.goto(url, timeout=60000)
        time.sleep(5)  # Increased wait time after page load
        
        # Get match info and create directory structure
        season_folder, match_id = get_match_info_from_url(url)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ball_by_ball_dir = os.path.join(current_dir, season_folder, match_id, 'ball_by_ball')
        
        # Create directory if it doesn't exist
        if not os.path.exists(ball_by_ball_dir):
            os.makedirs(ball_by_ball_dir)

        # First check if there's any indicators that content isn't available
        if "content not available" in page.content().lower() or "no ball-by-ball data" in page.content().lower():
            print("WARNING: Page indicates ball-by-ball content may not be available")
            page.screenshot(path=os.path.join(ball_by_ball_dir, "initial_page_state.png"))

        # Scrape first innings
        print("\n=== SCRAPING FIRST INNINGS ===")
        output_file = os.path.join(ball_by_ball_dir, "First_innings_summary.csv")
        first_innings_result = scrape_innings(page, "First", output_file)
        
        # Take a screenshot after first innings
        page.screenshot(path=os.path.join(ball_by_ball_dir, "after_first_innings.png"))
        
        # Scrape second innings - just try to select it without verification or page reload
        print("\n=== ATTEMPTING TO SWITCH TO SECOND INNINGS ===")
        
        switch_success = False
        for attempt in range(3):
            try:
                print(f"Switch attempt #{attempt+1}")
                
                # Click the dropdown button
                dropdown_button_selector = 'div.ds-flex.ds-items-center.ds-cursor-pointer'
                page.wait_for_selector(dropdown_button_selector, timeout=5000)
                page.click(dropdown_button_selector)
                time.sleep(2)  # Wait for the dropdown
                
                # Find the unselected option and click it
                innings_options = page.locator("xpath=//ul[contains(@class, 'ds-flex')]/li/div[not(contains(@class, 'ds-font-bold'))]")
                options_count = innings_options.count()
                
                if options_count > 0:
                    # Click the first unselected option
                    innings_options.first.click()
                    print("Clicked on the other innings option.")
                    switch_success = True
                    
                    # Let the page update - just wait, no refresh
                    print("Waiting for page to update after innings switch...")
                    time.sleep(10)  # Give it plenty of time to load
                    
                    # Take screenshot after switching
                    page.screenshot(path=os.path.join(ball_by_ball_dir, f"after_innings_switch_attempt{attempt+1}.png"))
                    
                    # Always attempt to scrape the second innings
                    print("\n=== SCRAPING SECOND INNINGS ===")
                    output_file = os.path.join(ball_by_ball_dir, "Second_innings_summary.csv")
                    scrape_innings(page, "Second", output_file)
                    break
                else:
                    print("No alternative innings option found.")
                    if attempt < 2:  # Close dropdown and try again
                        # Click outside to close dropdown
                        page.mouse.click(10, 10)
                        time.sleep(1)
                    
            except Exception as e:
                print(f"Error during switch attempt #{attempt+1}: {e}")
                if attempt < 2:  # Try again if not the last attempt
                    time.sleep(2)
        
        if not switch_success:
            print("ERROR: Failed to switch innings after all attempts. Skipping second innings scraping.")

        browser.close()

if __name__ == "__main__":
    main()
