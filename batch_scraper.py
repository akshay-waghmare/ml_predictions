import os
import csv
import sys
import time
import argparse
import traceback
import subprocess
from datetime import datetime
import importlib

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed with better error handling"""
    # Dictionary mapping import names to package names (when they differ)
    package_map = {
        "bs4": "beautifulsoup4",  # BeautifulSoup can be imported as bs4
        "beautifulsoup4": "beautifulsoup4",
        "playwright": "playwright",
        "pandas": "pandas"
    }
    
    missing_packages = []
    
    # Check each dependency with direct import attempt instead of pkg_resources
    for import_name, pkg_name in package_map.items():
        try:
            # Try to import directly
            importlib.import_module(import_name)
            print(f"✓ {pkg_name} is installed")
        except ImportError:
            missing_packages.append(pkg_name)
            print(f"✗ {pkg_name} not found")
    
    # Additional check specifically for BeautifulSoup
    if "beautifulsoup4" in missing_packages:
        try:
            # Try importing bs4 directly
            import bs4
            print("✓ beautifulsoup4 is installed as bs4")
            # If we can import bs4, remove it from missing packages
            missing_packages.remove("beautifulsoup4")
        except ImportError:
            pass
    
    if missing_packages:
        print("\nERROR: Missing required dependencies.")
        print("Please install the following packages using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        
        if "playwright" in missing_packages:
            print("\nFor playwright, you also need to install the browsers:")
            print("python -m playwright install")
        
        print("\nIf you've already installed these packages but still see this message:")
        print("1. Make sure you're using the correct Python environment")
        print("2. Try running: pip list")
        print("3. Check if the packages are installed under different names")
        return False
    
    print("All required dependencies are installed.")
    return True

def log_message(message, log_file=None):
    """Log a message to console and optionally to a log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

def read_match_urls_from_csv(csv_file):
    """Read match URLs from the CSV file"""
    match_urls = []
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                match_urls.append({
                    'id': row['id'],
                    'url': row['url']
                })
        
        return match_urls
    
    except FileNotFoundError:
        log_message(f"CSV file not found: {csv_file}")
        return []
    except Exception as e:
        log_message(f"Error reading match URLs from CSV: {e}")
        return []

def process_match(match_url, match_id, log_file=None):
    """Process a single match URL by calling espnscraper.py directly"""
    try:
        log_message(f"Processing match: {match_id} - {match_url}", log_file)
        
        # Make sure the URL points to full-scorecard
        if "/full-scorecard" not in match_url:
            match_url = match_url.replace("ball-by-ball-commentary", "full-scorecard")
            log_message(f"Adjusted URL to: {match_url}", log_file)
        
        # Run the espnscraper.py script directly as a subprocess
        log_message(f"Executing espnscraper.py with URL: {match_url}", log_file)
        
        # Construct the command to run espnscraper.py
        python_executable = sys.executable  # Current Python interpreter
        espnscraper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'espnscraper.py')
        
        # Execute the command and capture output
        process = subprocess.Popen(
            [python_executable, espnscraper_path, match_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read output in real-time
        for line in process.stdout:
            log_message(f"espnscraper: {line.strip()}", log_file)
        
        # Get the return code
        return_code = process.wait()
        
        if return_code != 0:
            # Read error output
            stderr = process.stderr.read()
            log_message(f"espnscraper.py execution failed with return code {return_code}", log_file)
            log_message(f"Error output: {stderr}", log_file)
            return False
        else:
            log_message(f"Successfully processed match {match_id}", log_file)
            return True
        
    except Exception as e:
        log_message(f"Error processing match {match_id}: {str(e)}", log_file)
        log_message(traceback.format_exc(), log_file)
        return False

def filter_ipl_matches(match_urls):
    """Filter out non-IPL matches from the list of match URLs"""
    ipl_matches = []
    non_ipl_matches = []
    
    for match in match_urls:
        # Check if URL contains IPL identifier
        if 'indian-premier-league' in match['url'].lower():
            ipl_matches.append(match)
        else:
            non_ipl_matches.append(match)
    
    return ipl_matches, non_ipl_matches

def process_batch(csv_file, delay=30, start_idx=0, max_matches=None, ipl_only=True):
    """Process a batch of matches from the CSV file"""
    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    ensure_directory_exists(logs_dir)
    
    # Create log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"batch_scraper_{timestamp}.log")
    
    log_message(f"Starting batch processing of matches from {csv_file}", log_file)
    
    # Read match URLs from CSV
    all_match_urls = read_match_urls_from_csv(csv_file)
    
    if not all_match_urls:
        log_message(f"No match URLs found in {csv_file}", log_file)
        return
    
    # Filter for IPL matches if requested
    if ipl_only:
        match_urls, skipped_matches = filter_ipl_matches(all_match_urls)
        if skipped_matches:
            log_message(f"Skipping {len(skipped_matches)} non-IPL matches", log_file)
            for match in skipped_matches:
                log_message(f"  - {match['id']}: {match['url']}", log_file)
    else:
        match_urls = all_match_urls
    
    # Save filtered matches to a new CSV if needed
    if ipl_only and len(match_urls) < len(all_match_urls):
        filtered_csv = csv_file.replace('.csv', '_ipl_only.csv')
        with open(filtered_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'url'])
            writer.writeheader()
            writer.writerows(match_urls)
        log_message(f"Saved {len(match_urls)} IPL matches to {filtered_csv}", log_file)
    
    total_matches = len(match_urls)
    end_idx = min(total_matches, start_idx + max_matches) if max_matches else total_matches
    
    log_message(f"Found {total_matches} matches to process", log_file)
    log_message(f"Processing matches from index {start_idx} to {end_idx-1}", log_file)
    
    successes = 0
    failures = 0
    
    for i in range(start_idx, end_idx):
        match = match_urls[i]
        match_id = match['id']
        match_url = match['url']
        
        log_message(f"Processing match {i+1-start_idx} of {end_idx-start_idx} (ID: {match_id})", log_file)
        
        success = process_match(match_url, match_id, log_file)
        
        if success:
            successes += 1
        else:
            failures += 1
        
        if i < end_idx - 1:  # If not the last match
            log_message(f"Waiting {delay} seconds before processing the next match...", log_file)
            time.sleep(delay)
    
    log_message(f"Batch processing completed. Total processed: {successes + failures}", log_file)
    log_message(f"Successful: {successes}, Failed: {failures}", log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process IPL match data from CSV file')
    parser.add_argument('--csv', default='ipl_2024_matches.csv', 
                        help='Path to the CSV file containing match URLs')
    parser.add_argument('--delay', type=int, default=30, 
                        help='Delay in seconds between processing matches')
    parser.add_argument('--start', type=int, default=0,  # Fixed syntax error: added equals sign
                        help='Index of the first match to process')
    parser.add_argument('--max', type=int, default=None, 
                        help='Maximum number of matches to process')
    parser.add_argument('--url', help='Process a single URL instead of reading from CSV')
    parser.add_argument('--install-deps', action='store_true', 
                        help='Install required dependencies')
    parser.add_argument('--ipl-only', action='store_true', default=True,
                        help='Process only IPL matches (filter out others) - enabled by default')
    parser.add_argument('--all-matches', action='store_true',
                        help='Process all matches in the CSV, including non-IPL matches')
    parser.add_argument('--status', action='store_true',
                        help='Just show match status without processing')
    
    args = parser.parse_args()
    
    # Override ipl_only if all_matches is specified
    if args.all_matches:
        args.ipl_only = False
    
    if args.install_deps:
        print("Installing required dependencies...")
        os.system("pip install playwright beautifulsoup4 pandas")
        print("Installing playwright browsers...")
        os.system("python -m playwright install")
        print("Dependencies installed. Please run the script again without --install-deps.")
        sys.exit(0)
    
    if args.status:
        # Just show match status without processing
        match_urls = read_match_urls_from_csv(args.csv)
        ipl_matches, non_ipl_matches = filter_ipl_matches(match_urls)
        
        print(f"\nFound {len(match_urls)} total matches in CSV:")
        print(f"- {len(ipl_matches)} IPL matches")
        print(f"- {len(non_ipl_matches)} non-IPL matches")
        
        print(f"\nCurrent mode: {'IPL matches only' if args.ipl_only else 'All matches'}")
        print(f"Use --ipl-only to process only IPL matches (default)")
        print(f"Use --all-matches to process all matches")
        
        if non_ipl_matches:
            print("\nNon-IPL matches that will be skipped with --ipl-only:")
            for match in non_ipl_matches:
                print(f"- {match['id']}: {match['url']}")
        
        sys.exit(0)
    
    if args.url:
        # Process a single URL
        match_id = args.url.split('/')[-2].split('-')[-1]  # Extract match ID from URL
        process_match(args.url, match_id)
    else:
        # Process batch from CSV
        process_batch(args.csv, args.delay, args.start, args.max, args.ipl_only)