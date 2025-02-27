import os
import sys
import time
from datetime import datetime

def log_message(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_script(script_name, url=None):
    """Run a Python script with an optional URL parameter and return True if successful"""
    try:
        log_message(f"Starting {script_name}...")
        command = f"python {script_name}"
        if url:
            command += f" {url}"
        result = os.system(command)
        
        if result == 0:
            log_message(f"{script_name} completed successfully")
            return True
        else:
            log_message(f"Error: {script_name} failed with exit code {result}")
            return False
            
    except Exception as e:
        log_message(f"Error executing {script_name}: {str(e)}")
        return False

def main():
    # Get command line arguments
    if len(sys.argv) != 2:
        print("Usage: python run_scrapers.py <match_url>")
        sys.exit(1)
    
    match_url = sys.argv[1]
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(current_dir)
    
    # Define scripts to run in order
    scripts = [
        "espnscraper_ballbyball.py",
        "espnscraper.py"
    ]
    
    log_message("Starting scraping sequence")
    
    for script in scripts:
        # Verify script exists
        if not os.path.exists(script):
            log_message(f"Error: {script} not found in {current_dir}")
            sys.exit(1)
            
        # Run script with URL parameter
        success = run_script(script, match_url)
        
        # If script failed, exit
        if not success:
            log_message("Scraping sequence failed")
            sys.exit(1)
            
        # Add a small delay between scripts
        time.sleep(2)
    
    log_message("All scripts completed successfully")

if __name__ == "__main__":
    main()
