import requests
from bs4 import BeautifulSoup
import time
import csv
import os

def fetch_ipl_seasons():
    """
    Fetch all IPL season links from the main seasons page
    """
    url = "https://www.espncricinfo.com/ci/engine/series/index.html?search=ipl;view=season"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        soup = BeautifulSoup(response.content, 'html.parser')
        seasons = []
        
        # Find all season links in the slider calendar section
        slider_calendar = soup.find('section', id='slidercalendar')
        if slider_calendar:
            season_links = slider_calendar.find_all('a')
            for link in season_links:
                season_url = link.get('href')
                season_year = link.find('span', class_='year').text.strip()
                full_url = f"https://www.espncricinfo.com{season_url}"
                seasons.append({
                    'name': f"IPL {season_year}",
                    'url': full_url
                })
        else:
            # Fallback to previous method if slider calendar not found
            links = soup.select('.teams a')
            for link in links:
                season_url = link.get('href')
                season_name = link.text.strip()
                if 'ipl' in season_url.lower() or 'indian-premier-league' in season_url.lower():
                    full_url = f"https://www.espncricinfo.com{season_url}"
                    seasons.append({
                        'name': season_name,
                        'url': full_url
                    })
        
        return seasons
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching IPL seasons: {e}")
        return []

def fetch_ipl_series_from_season(season_url):
    """
    Fetch IPL series links from a specific season page
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(season_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        series_list = []
        
        # Find the Twenty20 section
        t20_sections = soup.find_all('div', class_='match-section-head')
        t20_section = None
        
        for section in t20_sections:
            if section.find('h2') and section.find('h2').text.strip() == 'Twenty20':
                t20_section = section
                break
        
        if t20_section:
            # Find the next series-summary-wrap section
            series_wrap = t20_section.find_next('section', class_='series-summary-wrap')
            if series_wrap:
                series_blocks = series_wrap.find_all('section', class_='series-summary-block')
                for block in series_blocks:
                    series_id = block.get('data-series-id')
                    teams_div = block.find('div', class_='teams')
                    if teams_div and teams_div.find('a'):
                        link = teams_div.find('a')
                        series_name = link.text.strip()
                        series_url = link.get('href')
                        
                        # Check if it's an IPL series
                        if 'indian-premier-league' in series_url.lower() or 'ipl' in series_name.lower():
                            full_url = f"https://www.espncricinfo.com{series_url}" if series_url.startswith('/') else series_url
                            
                            # Get winner info
                            result_info = block.find('div', class_='result-info')
                            winner = result_info.text.strip().replace('Winner', '').strip() if result_info else "Unknown"
                            
                            # Get date and location
                            date_loc = block.find('div', class_='date-location')
                            date_location = date_loc.text.strip() if date_loc else "Unknown"
                            
                            series_list.append({
                                'id': series_id,
                                'name': series_name,
                                'url': full_url,
                                'winner': winner,
                                'date_location': date_location
                            })
        
        return series_list
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching series from {season_url}: {e}")
        return []

def save_to_csv(series_list, filename='ipl_series_data.csv'):
    """
    Save the series data to a CSV file
    
    Args:
        series_list (list): List of dictionaries containing series data
        filename (str): Name of the CSV file to save
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    try:
        # Define the fields to include in the CSV
        fieldnames = ['id', 'name', 'winner', 'date_location', 'url']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for series in series_list:
                # Create a new dict with just the fields we want
                row = {field: series.get(field, '') for field in fieldnames}
                writer.writerow(row)
        
        print(f"\nData saved to {filepath}")
        return filepath
    except PermissionError:
        alt_filename = f"ipl_series_data_{int(time.time())}.csv"
        print(f"Permission denied when writing to {filename}. Trying alternative filename: {alt_filename}")
        return save_to_csv(series_list, alt_filename)

def fetch_match_links_from_series(series_url):
    """
    Fetch all match links from a specific IPL series URL
    
    Args:
        series_url (str): URL of the IPL series fixtures page
        
    Returns:
        list: List of dictionaries containing match data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Fetching matches from: {series_url}")
        response = requests.get(series_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        matches = []
        
        # Looking for match links directly - these URLs are already fixture pages
        # First try to find all full-scorecard links
        match_links = soup.select('a[href*="/full-scorecard"]')
        
        if not match_links:
            print("No match links found using primary selector, trying alternative...")
            # Try alternative selectors
            match_links = soup.select('.match-info-link-FIXTURES')
            
            if not match_links:
                # Try another approach
                match_links = soup.select('a.match-center-link')
        
        print(f"Found {len(match_links)} potential match links")
        
        # Extract match URLs
        for link in match_links:
            match_url = link.get('href')
            if not match_url.startswith('http'):
                match_url = f"https://www.espncricinfo.com{match_url}"
            
            # Only include links to full scorecards
            if '/full-scorecard' in match_url:
                match_id = match_url.split('/')[-1].split('?')[0]
                matches.append({
                    'id': match_id,
                    'url': match_url
                })
        
        print(f"Successfully extracted {len(matches)} match links")
        return matches
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches from {series_url}: {e}")
        return []

def read_csv_and_fetch_2024_matches(csv_file='ipl_series_data.csv'):
    """
    Read the CSV file, filter for the 2024 season, and fetch all match links
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        list: List of match links for the 2024 season
    """
    filepath = os.path.join(os.path.dirname(__file__), csv_file)
    
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if '2023' in row['name'] or '2023' in row['date_location']:
                    print(f"\nFound 2023 IPL season: {row['name']}")
                    print(f"URL: {row['url']}")
                    
                    print("\nFetching match links...")
                    match_links = fetch_match_links_from_series(row['url'])
                    
                    if match_links:
                        print(f"Found {len(match_links)} matches in the 2023 IPL season:")
                        
                        # Save match links to CSV
                        matches_csv = 'ipl_2023_matches.csv'
                        matches_path = os.path.join(os.path.dirname(__file__), matches_csv)
                        
                        try:
                            with open(matches_path, 'w', newline='', encoding='utf-8') as matches_file:
                                fieldnames = ['id', 'url']
                                writer = csv.DictWriter(matches_file, fieldnames=fieldnames)
                                writer.writeheader()
                                for match in match_links:
                                    writer.writerow(match)
                                    print(f"Match ID: {match['id']} - {match['url']}")
                            
                            print(f"\nMatch links saved to {matches_path}")
                        except PermissionError:
                            alt_filename = f"ipl_2023_matches_{int(time.time())}.csv"
                            print(f"Permission denied when writing to {matches_path}. Trying alternative filename: {alt_filename}")
                            alt_path = os.path.join(os.path.dirname(__file__), alt_filename)
                            with open(alt_path, 'w', newline='', encoding='utf-8') as matches_file:
                                fieldnames = ['id', 'url']
                                writer = csv.DictWriter(matches_file, fieldnames=fieldnames)
                                writer.writeheader()
                                for match in match_links:
                                    writer.writerow(match)
                                print(f"\nMatch links saved to {alt_path}")
                        
                        return match_links
                    else:
                        print("No matches found for the 2023 IPL season.")
                        return []
            
            print("2024 IPL season not found in the CSV file.")
            return []
            
    except FileNotFoundError:
        print(f"CSV file not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading CSV or fetching match links: {e}")
        return []

def extract_match_id_from_url(match_url):
    """
    Extract the match ID from a match URL
    
    Args:
        match_url (str): URL of the match
        
    Returns:
        str: Match ID
    """
    # The match ID is typically the numeric value before "/full-scorecard"
    try:
        # Split by "/" and get the part before "full-scorecard"
        parts = match_url.split('/')
        for i, part in enumerate(parts):
            if part == "full-scorecard" and i > 0:
                return parts[i-1]
        
        # Alternative method if the above fails
        if "final-" in match_url:
            return match_url.split("final-")[-1].split("/")[0]
        
        # Another fallback method
        return match_url.split('/')[-2]
    except Exception:
        print(f"Could not extract match ID from URL: {match_url}")
        return "unknown"

def test_match_id_extraction():
    """
    Test the match ID extraction with a known URL
    """
    test_url = "https://www.espncricinfo.com/series/indian-premier-league-2024-1410320/kolkata-knight-riders-vs-sunrisers-hyderabad-final-1426312/full-scorecard"
    match_id = extract_match_id_from_url(test_url)
    print(f"Test URL: {test_url}")
    print(f"Extracted Match ID: {match_id}")
    return match_id

if __name__ == "__main__":
    # Add option to test match ID extraction
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test-match-id':
        print("Testing match ID extraction...")
        match_id = test_match_id_extraction()
        print(f"For the 2024 IPL final, the match ID is: {match_id}")
    else:
        # Update the fetch_match_links_from_series function to use the extract_match_id_from_url function
        old_fetch_match_links = fetch_match_links_from_series
        
        def fetch_match_links_from_series(series_url):
            matches = old_fetch_match_links(series_url)
            # Update the match IDs using the more robust extraction function
            for match in matches:
                match['id'] = extract_match_id_from_url(match['url'])
            return matches
        
        print("Using existing CSV data to fetch 2024 IPL matches...")
        print("Reading from ipl_series_data.csv and fetching match links...")
        read_csv_and_fetch_2024_matches()
