from typing import Dict, List, Any
import requests
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NHLTeamStats:
    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1"
        self.teams = {
            "TOR": {"id": 10, "name": "Toronto Maple Leafs"},
            "FLA": {"id": 13, "name": "Florida Panthers"}
        }
        
    def get_team_roster(self, team_abbrev: str) -> List[Dict[str, Any]]:
        """Fetch full roster for a team."""
        url = f"{self.base_url}/roster/{team_abbrev}/current"
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch roster for {team_abbrev}: {response.status_code}")
            return []
            
        data = response.json()
        all_players = []
        # Combine forwards, defensemen, and goalies
        for position in ['forwards', 'defensemen', 'goalies']:
            if position in data:
                all_players.extend(data[position])
        return all_players
    
    def get_player_stats(self, player_id: int) -> Dict[str, Any]:
        """Fetch detailed stats for a player."""
        url = f"{self.base_url}/player/{player_id}/landing"
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Failed to fetch stats for player {player_id}: {response.status_code}")
            return {}
            
        return response.json()
    
    def get_full_team_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get complete data for both teams."""
        team_data = {}
        
        for team_abbrev in self.teams.keys():
            logger.info(f"Fetching data for {team_abbrev}")
            roster = self.get_team_roster(team_abbrev)
            
            player_data = []
            for player in roster:
                player_id = player['id']
                stats = self.get_player_stats(player_id)
                if stats:
                    player_data.append(stats)
            
            team_data[team_abbrev] = player_data
            
        return team_data
    
    def save_team_data(self, output_dir: str = "data") -> None:
        """Fetch and save team data to JSON files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        team_data = self.get_full_team_data()
        
        # Save combined data
        output_path = Path(output_dir) / "leafs_panthers_data.json"
        with open(output_path, 'w') as f:
            json.dump(team_data, f, indent=2)
            
        logger.info(f"Saved combined team data to {output_path}")

if __name__ == "__main__":
    stats = NHLTeamStats()
    stats.save_team_data() 