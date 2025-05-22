"""Tests for the data processor module."""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open
from src.nhl_ml.data.data_processor import NHLDataProcessor

@pytest.fixture
def sample_player_data():
    """Fixture for sample player data."""
    return {
        "TOR": [
            {
                "playerId": 8478483,
                "firstName": {"default": "John"},
                "lastName": {"default": "Doe"},
                "currentTeamAbbrev": "TOR",
                "position": "C",
                "birthDate": "1996-09-17",
                "heightInCentimeters": 183,
                "weightInKilograms": 88,
                "featuredStats": {
                    "regularSeason": {
                        "subSeason": {
                            "gamesPlayed": 82,
                            "goals": 30,
                            "assists": 50,
                            "points": 80,
                            "plusMinus": 15,
                            "pim": 20,
                            "shots": 200,
                            "shootingPctg": 15.0,
                            "powerPlayGoals": 10,
                            "powerPlayPoints": 25
                        }
                    }
                },
                "careerTotals": {
                    "regularSeason": {
                        "gamesPlayed": 300,
                        "goals": 100,
                        "assists": 150,
                        "points": 250,
                        "plusMinus": 45,
                        "pim": 80,
                        "shots": 800,
                        "shootingPctg": 12.5,
                        "powerPlayGoals": 30,
                        "powerPlayPoints": 75
                    }
                }
            }
        ]
    }

@pytest.fixture
def processor():
    """Fixture for NHLDataProcessor instance."""
    return NHLDataProcessor("test.json")

def test_extract_player_features(processor, sample_player_data):
    """Test player feature extraction."""
    player_data = sample_player_data["TOR"][0]
    features = processor.extract_player_features(player_data)
    
    assert features["player_id"] == 8478483
    assert features["name"] == "John Doe"
    assert features["team"] == "TOR"
    assert features["position"] == "C"
    assert features["age"] == 28  # 2024 - 1996
    assert features["height_cm"] == 183
    assert features["weight_kg"] == 88
    assert features["games_played"] == 82
    assert features["goals"] == 30
    assert features["points"] == 80
    assert features["career_games"] == 300
    assert features["career_points"] == 250
    assert features["goals_per_game"] == pytest.approx(30/82)

@patch("builtins.open", new_callable=mock_open)
def test_load_json_data(mock_file, processor, sample_player_data):
    """Test JSON data loading."""
    mock_file.return_value.read.return_value = (
        "Some log data\n"
        "Saving complete stats to file...\n"
        f"{json.dumps(sample_player_data)}"
    )
    
    data = processor.load_json_data()
    assert "TOR" in data
    assert len(data["TOR"]) == 1
    assert data["TOR"][0]["playerId"] == 8478483

@patch.object(Path, "mkdir")
@patch("pandas.DataFrame.to_csv")
def test_save_processed_data(mock_to_csv, mock_mkdir, processor, sample_player_data):
    """Test saving processed data."""
    with patch.object(processor, "load_json_data", return_value=sample_player_data):
        processor.save_processed_data("test_output")
        
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_to_csv.assert_called_once() 