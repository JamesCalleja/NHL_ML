"""
Module for retrieving and processing team-level NHL statistics.
"""

import requests
import json
import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class TeamStats:
    def __init__(self, team_id: int):
        """Initialize with team ID."""
        self.team_id = team_id
        self.base_url = "https://api-web.nhle.com/v1"
        self.headers = {'Accept': 'application/json'}

    def get_team_info(self) -> Optional[Dict]:
        """Fetch basic team information."""
        url = f"{self.base_url}/club-stats/team/{self.team_id}/now"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Error fetching team info: {e}")
            return None

    def get_team_schedule(self, season: str) -> List[Dict]:
        """Fetch team schedule for a given season."""
        url = (f"{self.base_url}/club-schedule-season/"
               f"{self.team_id}/{season}")

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            return data.get('games', [])

        except Exception as e:
            logger.error(f"Error fetching schedule: {e}")
            return []

    def get_team_stats(self) -> Dict:
        """Get comprehensive team statistics."""
        info = self.get_team_info()

        if not info:
            return {}

        # Extract relevant stats
        stats = {
            'team_id': self.team_id,
            'name': info.get('name', ''),
            'games_played': info.get('gamesPlayed', 0),
            'points': info.get('points', 0),
            'goals_for': info.get('goalsFor', 0),
            'goals_against': info.get('goalsAgainst', 0)
        }

        return stats


# Example usage
if __name__ == "__main__":
    # Toronto Maple Leafs (ID: 10)
    leafs = TeamStats(10)
    print(json.dumps(leafs.get_team_stats(), indent=2))
