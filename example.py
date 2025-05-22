"""
Example usage of the NHL ML package.
"""

import os
from nhl_ml.api import NHLStats
from nhl_ml.utils import TEAM_IDS
import pandas as pd

def main():
    # Make sure you have set the NHL_API_KEY environment variable:
    # export NHL_API_KEY='your-api-key'
    if not os.getenv('NHL_API_KEY'):
        print("Warning: NHL_API_KEY environment variable not set")
    
    # Initialize the NHL Stats client
    nhl = NHLStats()
    
    # Example: Get Max Domi's stats from Toronto Maple Leafs
    player_name = "Max Domi"
    team_id = TEAM_IDS['TOR']  # Toronto Maple Leafs
    season = "20232024"
    
    # Get the stats
    stats = nhl.get_player_stats(player_name, season, team_id)
    
    if stats:
        # Convert to DataFrame for better visualization
        df = pd.DataFrame([stats])
        
        # Select key statistics
        key_stats = [
            'gamesPlayed', 'goals', 'assists', 'points',
            'plusMinus', 'pim', 'shots', 'timeOnIcePerGame'
        ]
        
        print(f"\n{player_name}'s {season} Statistics:")
        print(df[key_stats].to_string(index=False))
    else:
        print(f"Could not find stats for {player_name}")

if __name__ == "__main__":
    main() 