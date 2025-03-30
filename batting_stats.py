import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

import pandas as pd
from bs4 import BeautifulSoup

from com.cricket.stats.formats import Formats
import requests


class BattingHistoryDownloader:
    def __init__(self):
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        
    def get_ipl_stats_from_csv(self, player_id=None, player_name=None):
        """
        Fetches IPL stats for a player from local CSV files instead of going to the web.
        
        Args:
            player_id: The player ID (not used for CSV matching but included for API compatibility)
            player_name: The player name to search for
        
        Returns:
            Tuple of (batting_df, bowling_df) with player's IPL stats
        """
        try:
            if not player_name:
                print(f"Error: Player name is required for CSV matching")
                return None, None
                
            # Import fuzzywuzzy for fuzzy matching
            from fuzzywuzzy import process
            import os
            
            # Define paths to the CSV files
            batting_csv_path = os.path.join(os.path.dirname(__file__), "ipl_historical_batting_stats.csv")
            bowling_csv_path = os.path.join(os.path.dirname(__file__), "ipl_historical_bowling_stats.csv")
            
            # Load the CSV files
            batting_stats_df = pd.read_csv(batting_csv_path)
            bowling_stats_df = pd.read_csv(bowling_csv_path)
            
            # Find the player in each CSV using fuzzy matching
            best_batting_match = None
            best_bowling_match = None
            
            # Try exact match first
            batting_row = batting_stats_df[batting_stats_df["Player"].str.contains(player_name, case=False)]
            bowling_row = bowling_stats_df[bowling_stats_df["Player"].str.contains(player_name, case=False)]
            
            # If exact match fails, try fuzzy matching
            if batting_row.empty:
                batting_players = batting_stats_df["Player"].tolist()
                best_batting_match, score = process.extractOne(player_name, batting_players)
                if score >= 70:  # Only use match if score is good enough
                    batting_row = batting_stats_df[batting_stats_df["Player"] == best_batting_match]
            
            if bowling_row.empty:
                bowling_players = bowling_stats_df["Player"].tolist()
                best_bowling_match, score = process.extractOne(player_name, bowling_players)
                if score >= 70:  # Only use match if score is good enough
                    bowling_row = bowling_stats_df[bowling_stats_df["Player"] == best_bowling_match]
            
            # Process batting stats if found
            batting_df = None
            if not batting_row.empty:
                row = batting_row.iloc[0]
                print(f"Found batting stats for {player_name} (matched with {row['Player']})")
                
                # Define the columns expected in the output DataFrame
                batting_columns = [
                    "Tournament", "Teams", "Matches", "Innings", "Not Out", "Runs", "Highest", "Average", 
                    "Balls Faced", "Strike Rate", "100s", "50s", "4s", "6s", "Catches", "Stumpings"
                ]
                
                # Safely get values with fallbacks
                batting_data = [
                    "IPL",  # Tournament
                    self._safe_get(row, "Span", ""),  # Teams (using Span)
                    self._safe_get(row, "Mat", 0),  # Matches
                    self._safe_get(row, "Inns", 0),  # Innings
                    self._safe_get(row, "NO", 0),  # Not Out
                    self._safe_get(row, "RunsDescending", 0),  # Runs
                    self._safe_get(row, "HS", "0"),  # Highest
                    self._safe_get(row, "Ave", 0.0),  # Average
                    self._safe_get(row, "BF", 0),  # Balls Faced
                    self._safe_get(row, "SR", 0.0),  # Strike Rate
                    self._safe_get(row, "100", 0),  # 100s
                    self._safe_get(row, "50", 0),  # 50s
                    self._safe_get(row, "4s", 0),  # 4s
                    0,  # 6s (not available)
                    0,  # Catches (not available)
                    0   # Stumpings (not available)
                ]
                
                batting_df = pd.DataFrame([batting_data], columns=batting_columns)
            else:
                print(f"No batting stats found for {player_name}")
            
            # Process bowling stats if found
            bowling_df = None
            if not bowling_row.empty:
                row = bowling_row.iloc[0]
                print(f"Found bowling stats for {player_name} (matched with {row['Player']})")
                
                # Define the columns expected in the output DataFrame
                bowling_columns = [
                    "Tournament", "Teams", "Matches", "Innings", "Balls", "Runs", "Wickets", "Best Bowling Innings", 
                    "Best Bowling Match", "Average", "Economy", "Strike Rate", "4w", "5w", "10w"
                ]
                
                # Calculate balls from overs if available
                balls = 0
                if "Overs" in row:
                    overs_str = str(row["Overs"])
                    try:
                        if '.' in overs_str:
                            main_overs, partial_balls = overs_str.split('.')
                            balls = int(float(main_overs)) * 6 + int(float(partial_balls))
                        else:
                            balls = int(float(overs_str)) * 6
                    except (ValueError, TypeError):
                        balls = 0
                
                # Safely get values with fallbacks
                bowling_data = [
                    "IPL",  # Tournament
                    self._safe_get(row, "Span", ""),  # Teams (using Span)
                    self._safe_get(row, "Mat", 0),  # Matches
                    self._safe_get(row, "Inns", 0),  # Innings
                    balls,  # Balls (calculated from Overs)
                    self._safe_get(row, "Runs", 0),  # Runs
                    self._safe_get(row, "WktsDescending", 0),  # Wickets
                    self._safe_get(row, "BBI", "0/0"),  # Best Bowling Innings
                    "0/0",  # Best Bowling Match (not available)
                    self._safe_get(row, "Ave", 0.0),  # Average
                    self._safe_get(row, "Econ", 0.0),  # Economy
                    self._safe_get(row, "SR", 0.0),  # Strike Rate
                    self._safe_get(row, "4", 0),  # 4w
                    0,  # 5w (not available)
                    0   # 10w (not available)
                ]
                
                bowling_df = pd.DataFrame([bowling_data], columns=bowling_columns)
            else:
                print(f"No bowling stats found for {player_name}")
            
            return batting_df, bowling_df
        
        except Exception as e:
            print(f"Error fetching IPL stats from CSV: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _safe_get(self, row, column, default):
        """
        Safely get a value from a DataFrame row with a fallback default value.
        Handles missing columns and conversion errors.
        """
        if column not in row:
            return default
            
        try:
            value = row[column]
            if pd.isna(value):
                return default
            
            # For numeric values, try to convert to the appropriate type
            if isinstance(default, int):
                return int(float(value))
            elif isinstance(default, float):
                return float(value)
            return value
        except:
            return default
    
    def get_ipl_batting_stats(self, player_id):
        try:
            url = self.__create_profile_url(player_id)
            
            # Create a session to maintain cookies
            session = requests.Session()
            
            # Set a realistic user agent and other headers based on your successful request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Referer': 'https://www.espncricinfo.com/',
                'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1',
                'priority': 'u=0, i',
                'connection': 'keep-alive'
            }
            
            # First visit the homepage to get cookies
            session.get('https://www.espncricinfo.com/', headers=headers, verify=False)
            
            # Add a small delay to mimic human behavior
            import time
            import random
            time.sleep(random.uniform(1.0, 2.0))
            
            # Now request the player profile
            response = session.get(url, headers=headers, verify=False)
            response.raise_for_status()
            html = response.content
            
            # Continue with your existing parsing code...
            soup = BeautifulSoup(html, "html.parser")

            # Find Batting & Fielding table
            batting_tables = soup.find_all('p', text='Batting & Fielding')
            if len(batting_tables) < 2:
                raise RuntimeError("Not enough batting tables found")
            batting_table = batting_tables[1].find_next('table')
            if not batting_table:
                raise RuntimeError("Batting & Fielding table not found")

            # Find Bowling table
            bowling_tables = soup.find_all('p', text='Bowling')
            if len(bowling_tables) < 2:
                raise RuntimeError("Not enough bowling tables found")
            bowling_table = bowling_tables[1].find_next('table')
            if not bowling_table:
                raise RuntimeError("Bowling table not found")

            # Extract IPL row from Batting table
            tbody = batting_table.find('tbody')
            if not tbody:
                raise RuntimeError("Table body not found")

            # Find the IPL row specifically
            ipl_batting_row = None
            for row in tbody.find_all('tr'):
                tournament = row.find('td').text.strip()
                if tournament == "IPL":
                    ipl_batting_row = row
                    break

            if not ipl_batting_row:
                raise RuntimeError("IPL batting stats not found")

            ipl_batting_data = [td.text.strip() for td in ipl_batting_row.find_all('td')]

            # Similarly for bowling table
            tbody = bowling_table.find('tbody')
            if not tbody:
                raise RuntimeError("Table body not found")

            # Find the IPL row specifically
            ipl_bowling_row = None
            for row in tbody.find_all('tr'):
                tournament = row.find('td').text.strip()
                if tournament == "IPL":
                    ipl_bowling_row = row
                    break

            if not ipl_bowling_row:
                raise RuntimeError("IPL bowling stats not found")

            ipl_bowling_data = [td.text.strip() for td in ipl_bowling_row.find_all('td')]

            # Define columns for Batting and Bowling tables
            batting_columns = [
                "Tournament", "Teams", "Matches", "Innings", "Not Out", "Runs", "Highest", "Average", 
                "Balls Faced", "Strike Rate", "100s", "50s", "4s", "6s", "Catches", "Stumpings"
            ]
            bowling_columns = [
                "Tournament", "Teams", "Matches", "Innings", "Balls", "Runs", "Wickets", "Best Bowling Innings", 
                "Best Bowling Match", "Average", "Economy", "Strike Rate", "4w", "5w", "10w"
            ]

            # Create DataFrames for Batting and Bowling stats
            batting_df = None
            bowling_df = None
            if len(ipl_batting_data) == len(batting_columns):
                batting_df = pd.DataFrame([ipl_batting_data], columns=batting_columns)
            else:
                print(f"Expected {len(batting_columns)} batting columns, but got {len(ipl_batting_data)}")
                bowling_df = pd.DataFrame([ipl_batting_data], columns=bowling_columns)
                batting_df = pd.DataFrame([ipl_bowling_data], columns=batting_columns)
                return batting_df, bowling_df
                

            if len(ipl_bowling_data) == len(bowling_columns):
                bowling_df = pd.DataFrame([ipl_bowling_data], columns=bowling_columns)
            else:
                print(f"Expected {len(bowling_columns)} bowling columns, but got {len(ipl_bowling_data)}")
                batting_df = pd.DataFrame([ipl_bowling_data], columns=batting_columns)
                bowling_df = pd.DataFrame([ipl_batting_data], columns=bowling_columns)
                return batting_df, bowling_df

            return batting_df, bowling_df

        except Exception as e:
            print(f"Error fetching IPL stats: {e}")
            return None, None

    def __create_profile_url(self, player_id):
        """
        Constructs the new player profile URL using the modern format.
        Example: https://www.espncricinfo.com/cricketers/rohit-sharma-34102
        """
        return f"https://www.espncricinfo.com/cricketers/{player_id}"

    def get_ipl_batting_stats_with_playwright(self, player_id, context):
        """
        Fetches IPL stats for a player using Playwright.
        
        Args:
            player_id: The player ID from ESPNCricinfo
            context: An active Playwright browser context
        
        Returns:
            Tuple of (batting_df, bowling_df) with player's IPL stats
        """
        player_page = None
        try:
            url = self.__create_profile_url(player_id)
            print(f"Fetching stats for {player_id} from {url}")
            
            # Open a new page in the existing browser context
            player_page = context.new_page()
            
            # Set important headers to mimic Chrome browser
            player_page.set_extra_http_headers({
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'accept-language': 'en-US,en;q=0.9',
                'cache-control': 'max-age=0',
                'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'document',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'same-origin',
                'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1'
            })
            
            # First visit the homepage to get proper cookies
            player_page.goto('https://www.espncricinfo.com/')
            # Wait for 5 seconds
            
            # Add required cookies - these are critical values from the Chrome request
            player_page.evaluate("""() => {
                // Set some basic cookies that might help with authentication
                document.cookie = "edition=espncricinfo-en-in";
                document.cookie = "edition-view=espncricinfo-en-in";
                document.cookie = "region=unknown";
                document.cookie = "country=in";
            }""")
            
            # Short delay to ensure cookies are set
            player_page.wait_for_timeout(3000)
            
            # Now navigate to the player profile
            print(f"Navigating to {url}")
            player_page.goto(url, timeout=30000)
            
            # Wait for content to load
            player_page.wait_for_selector('p:has-text("Batting & Fielding")', timeout=10000)
            
            # Get the page content after JavaScript has rendered
            html = player_page.content()
            soup = BeautifulSoup(html, "html.parser")
            
            # Find Batting & Fielding table
            batting_tables = soup.find_all('p', text='Batting & Fielding')
            if len(batting_tables) < 2:
                raise RuntimeError("Not enough batting tables found")
            batting_table = batting_tables[1].find_next('table')
            if not batting_table:
                raise RuntimeError("Batting & Fielding table not found")

            # Find Bowling table
            bowling_tables = soup.find_all('p', text='Bowling')
            if len(bowling_tables) < 2:
                raise RuntimeError("Not enough bowling tables found")
            bowling_table = bowling_tables[1].find_next('table')
            if not bowling_table:
                raise RuntimeError("Bowling table not found")

# The rest of your table parsing code remains the same...
            # Extract IPL row from Batting table
            tbody = batting_table.find('tbody')
            if not tbody:
                raise RuntimeError("Table body not found")

            # Find the IPL row specifically
            ipl_batting_row = None
            for row in tbody.find_all('tr'):
                tournament = row.find('td').text.strip()
                if tournament == "IPL":
                    ipl_batting_row = row
                    break

            if not ipl_batting_row:
                raise RuntimeError("IPL batting stats not found")

            ipl_batting_data = [td.text.strip() for td in ipl_batting_row.find_all('td')]

            # Similarly for bowling table
            tbody = bowling_table.find('tbody')
            if not tbody:
                raise RuntimeError("Table body not found")

            # Find the IPL row specifically
            ipl_bowling_row = None
            for row in tbody.find_all('tr'):
                tournament = row.find('td').text.strip()
                if tournament == "IPL":
                    ipl_bowling_row = row
                    break

            if not ipl_bowling_row:
                raise RuntimeError("IPL bowling stats not found")

            ipl_bowling_data = [td.text.strip() for td in ipl_bowling_row.find_all('td')]

            # Define columns for Batting and Bowling tables
            batting_columns = [
                "Tournament", "Teams", "Matches", "Innings", "Not Out", "Runs", "Highest", "Average", 
                "Balls Faced", "Strike Rate", "100s", "50s", "4s", "6s", "Catches", "Stumpings"
            ]
            bowling_columns = [
                "Tournament", "Teams", "Matches", "Innings", "Balls", "Runs", "Wickets", "Best Bowling Innings", 
                "Best Bowling Match", "Average", "Economy", "Strike Rate", "4w", "5w", "10w"
            ]

            # Create DataFrames for Batting and Bowling stats
            batting_df = None
            bowling_df = None
            if len(ipl_batting_data) == len(batting_columns):
                batting_df = pd.DataFrame([ipl_batting_data], columns=batting_columns)
            else:
                print(f"Expected {len(batting_columns)} batting columns, but got {len(ipl_batting_data)}")
                bowling_df = pd.DataFrame([ipl_batting_data], columns=bowling_columns)
                batting_df = pd.DataFrame([ipl_bowling_data], columns=batting_columns)
                return batting_df, bowling_df
                
            if len(ipl_bowling_data) == len(bowling_columns):
                bowling_df = pd.DataFrame([ipl_bowling_data], columns=bowling_columns)
            else:
                print(f"Expected {len(bowling_columns)} bowling columns, but got {len(ipl_bowling_data)}")
                batting_df = pd.DataFrame([ipl_bowling_data], columns=batting_columns)
                bowling_df = pd.DataFrame([ipl_batting_data], columns=bowling_columns)
                return batting_df, bowling_df

            return batting_df, bowling_df

        except Exception as e:
            print(f"Error fetching IPL stats with Playwright: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        finally:
            # Always close the page to free resources
            if player_page:
                try:
                    player_page.close()
                except:
                    pass