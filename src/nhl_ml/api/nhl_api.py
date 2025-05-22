"""
NHL API interaction module.
"""

import requests
import json
import logging
from typing import Dict, Optional, List


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Constants
NHL_API_BASE = "https://api-web.nhle.com/v1"


class NHLStats:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json'
        }
        self.teams = {
            "TOR": {"id": 10, "name": "Toronto Maple Leafs"},
            "FLA": {"id": 13, "name": "Florida Panthers"}
        }

    def get_team_roster(self, team_abbrev: str) -> List[Dict]:
        """Get the current roster for a team."""
        roster_url = f"{NHL_API_BASE}/roster/{team_abbrev}/current"

        try:
            logger.info(f"Fetching roster for {team_abbrev}")
            response = requests.get(roster_url, headers=self.headers)
            response.raise_for_status()
            roster_data = response.json()

            all_players = []
            player_types = ['forwards', 'defensemen', 'goalies']
            for player_type in player_types:
                if player_type in roster_data:
                    all_players.extend(roster_data[player_type])

            msg = f"Found {len(all_players)} players on {team_abbrev} roster"
            logger.info(msg)
            return all_players

        except Exception as e:
            msg = f"Error fetching roster for {team_abbrev}: {e}"
            logger.error(msg)
            return []

    def get_player_stats(self, player_id: int) -> Optional[Dict]:
        """Get detailed stats for a player."""
        stats_url = f"{NHL_API_BASE}/player/{player_id}/landing"

        try:
            logger.info(f"Fetching stats for player ID {player_id}")
            response = requests.get(stats_url, headers=self.headers)
            response.raise_for_status()
            stats_data = response.json()

            if stats_data:
                first_name = stats_data.get('firstName', {}).get('default', '')
                last_name = stats_data.get('lastName', {}).get('default', '')
                name = f"{first_name} {last_name}".strip()
                logger.info(f"Found stats for {name}")
                return stats_data

            return None

        except Exception as e:
            msg = f"Error fetching stats for player ID {player_id}: {e}"
            logger.error(msg)
            return None

    def get_all_team_stats(self) -> Dict[str, List[Dict]]:
        """Get stats for all players on both teams."""
        all_stats = {}

        for team_abbrev in self.teams:
            logger.info(f"\nProcessing team: {team_abbrev}")

            # Get team roster
            roster = self.get_team_roster(team_abbrev)
            team_stats = []

            # Get stats for each player
            for player in roster:
                player_id = player.get('id')
                if player_id:
                    stats = self.get_player_stats(player_id)
                    if stats:
                        team_stats.append(stats)

            all_stats[team_abbrev] = team_stats
            msg = f"Processed {len(team_stats)} players for {team_abbrev}"
            logger.info(msg)

        return all_stats

    def save_team_stats(self, output_file: str = "output.json") -> None:
        """Fetch and save all team stats to a file."""
        all_stats = self.get_all_team_stats()

        # Save to file
        logger.info(f"\nSaving stats to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        logger.info("Stats saved successfully")


# Test section - only runs when this file is run directly
if __name__ == "__main__":
    # Test the NHL Stats client
    nhl = NHLStats()
    nhl.save_team_stats()
