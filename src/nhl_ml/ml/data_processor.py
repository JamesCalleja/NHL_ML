import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NHLDataProcessor:
    def __init__(self, json_path: str = "output.json"):
        self.json_path = json_path
        self.teams_of_interest = {"TOR": "Toronto Maple Leafs", "FLA": "Florida Panthers"}
        
    def load_json_data(self) -> Dict[str, List[Dict]]:
        """Load and parse the JSON data from file."""
        try:
            with open(self.json_path, 'r') as f:
                content = f.read()
                
            # Find the JSON object at the end of the file
            json_str = content.split("Saving complete stats to file...")[-1].strip()
            return json.loads(json_str)
                
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return {}

    def extract_player_features(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from player data."""
        features = {}
        
        # Basic info
        features['player_id'] = player_data.get('playerId')
        features['team'] = player_data.get('currentTeamAbbrev')
        features['position'] = player_data.get('position')
        
        # Get name
        first_name = player_data['firstName'].get('default', '') if isinstance(player_data.get('firstName'), dict) else player_data.get('firstName', '')
        last_name = player_data['lastName'].get('default', '') if isinstance(player_data.get('lastName'), dict) else player_data.get('lastName', '')
        features['name'] = f"{first_name} {last_name}".strip()
        
        # Try to get birth date
        try:
            birth_date = player_data.get('birthDate', '1900-01-01')
            features['age'] = 2024 - int(birth_date.split('-')[0])
        except:
            features['age'] = None
        
        # Physical attributes
        features['height_cm'] = player_data.get('heightInCentimeters')
        features['weight_kg'] = player_data.get('weightInKilograms')
        
        # Current season stats
        reg_season = player_data.get('featuredStats', {}).get('regularSeason', {}).get('subSeason', {})
        features.update({
            'games_played': reg_season.get('gamesPlayed', 0),
            'goals': reg_season.get('goals', 0),
            'assists': reg_season.get('assists', 0),
            'points': reg_season.get('points', 0),
            'plus_minus': reg_season.get('plusMinus', 0),
            'pim': reg_season.get('pim', 0),
            'shots': reg_season.get('shots', 0),
            'shooting_pct': reg_season.get('shootingPctg', 0),
            'powerplay_goals': reg_season.get('powerPlayGoals', 0),
            'powerplay_points': reg_season.get('powerPlayPoints', 0)
        })
        
        # Career stats
        career = player_data.get('careerTotals', {}).get('regularSeason', {})
        features.update({
            'career_games': career.get('gamesPlayed', 0),
            'career_goals': career.get('goals', 0),
            'career_assists': career.get('assists', 0),
            'career_points': career.get('points', 0),
            'career_plus_minus': career.get('plusMinus', 0),
            'career_pim': career.get('pim', 0),
            'career_shots': career.get('shots', 0),
            'career_shooting_pct': career.get('shootingPctg', 0),
            'career_powerplay_goals': career.get('powerPlayGoals', 0),
            'career_powerplay_points': career.get('powerPlayPoints', 0)
        })
        
        # Calculate per-game metrics
        if features['games_played'] > 0:
            features['goals_per_game'] = features['goals'] / features['games_played']
            features['points_per_game'] = features['points'] / features['games_played']
            features['shots_per_game'] = features['shots'] / features['games_played']
        else:
            features['goals_per_game'] = 0
            features['points_per_game'] = 0
            features['shots_per_game'] = 0
            
        return features

    def create_dataset(self) -> pd.DataFrame:
        """Create a pandas DataFrame from the processed data."""
        all_data = self.load_json_data()
        
        # Extract player data
        all_players = []
        processed_ids = set()  # To avoid duplicates
        
        # Process each team's data
        for team_abbrev, team_data in all_data.items():
            logger.info(f"Processing {team_abbrev} data...")
            for player_data in team_data:
                player_id = player_data.get('playerId')
                if player_id and player_id not in processed_ids:
                    player_features = self.extract_player_features(player_data)
                    if player_features.get('team') in self.teams_of_interest:
                        all_players.append(player_features)
                        processed_ids.add(player_id)
                        logger.info(f"Processed player: {player_features.get('name', 'Unknown')}")

        # Create DataFrame
        df = pd.DataFrame(all_players)
        
        # Basic data cleaning
        if not df.empty:
            df = df.dropna(subset=['player_id', 'team'])  # Remove rows missing critical data
        
        logger.info(f"Created dataset with {len(df)} players")
        return df

    def save_processed_data(self, output_dir: str = "data") -> None:
        """Save the processed data to CSV."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = self.create_dataset()
        
        if len(df) > 0:
            output_path = Path(output_dir) / "processed_player_stats.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            # Print some basic statistics
            print("\nDataset Summary:")
            print("-" * 20)
            print(f"Total players: {len(df)}")
            print("\nTeam distribution:")
            print(df['team'].value_counts())
            print("\nPosition distribution:")
            print(df['position'].value_counts())
            print("\nTop 10 players by points:")
            print(df[['name', 'team', 'position', 'points', 'points_per_game']].sort_values('points', ascending=False).head(10))
            print("\nFeature statistics:")
            print(df.describe())
        else:
            logger.warning("No valid player data was found to save")

if __name__ == "__main__":
    processor = NHLDataProcessor()
    processor.save_processed_data() 