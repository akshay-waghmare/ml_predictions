import re
from enum import Enum
from batting_stats import BattingHistoryDownloader

class PlayerStatsFactory:
    class IPLTeam(Enum):
        MI = "Mumbai Indians"
        DC = "Delhi Capitals"
        SRH = "Sunrisers Hyderabad"
        RCB = "Royal Challengers Bangalore"
        KKR = "Kolkata Knight Riders"
        KXIP = "Kings XI Punjab"
        CSK = "Chennai Super Kings"
        RR = "Rajasthan Royals"

    def __init__(self, context):
        self.context = context
        self.batting_downloader = BattingHistoryDownloader()

    def get_team_sections(self, page):
        team_sections = []
        team_elements = page.locator('span.ds-text-title-xs.ds-font-bold.ds-capitalize')
        
        for i in range(team_elements.count()):
            team_name = team_elements.nth(i).inner_text().strip()
            player_section = page.locator('div.ds-w-full.ds-bg-fill-content-prime').nth(i)
            team_sections.append({
                "team_name": team_name,
                "player_section": player_section
            })
        return team_sections

    def get_player_links(self, team_sections):
        player_links = []
        for section in team_sections:
            team_name = self._normalize_team_name(section["team_name"])
            players = section["player_section"].locator('a[href^="/cricketers/"]')
            
            for idx in range(players.count()):
                player = self._extract_player_info(players.nth(idx), team_name)
                if player: player_links.append(player)
        
        return player_links

    def _normalize_team_name(self, team_name):
        return next((t.name for t in self.IPLTeam if t.value == team_name), team_name)

    def _extract_player_info(self, player_element, team_name):
        try:
            name = player_element.inner_text().strip()
            link = player_element.get_attribute("href")
            player_id = re.search(r'-(\d+)$', link).group(1)
            return {
                "name": name,
                "link": f"https://www.espncricinfo.com{link}",
                "id": player_id,
                "team": team_name
            }
        except Exception as e:
            print(f"Error extracting player info: {e}")
            return None