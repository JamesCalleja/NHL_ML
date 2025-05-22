"""Tests for the NHL API module."""

import json
import pytest
from unittest.mock import patch, Mock
from src.nhl_ml.api.nhl_api import NHLStats, NHL_API_BASE

@pytest.fixture
def nhl_stats():
    """Fixture for NHLStats instance."""
    return NHLStats()

@pytest.fixture
def mock_player_response():
    """Fixture for mocked player response."""
    return {
        "firstName": {"default": "John"},
        "lastName": {"default": "Doe"},
        "playerId": 8478483,
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
        }
    }

def test_init(nhl_stats):
    """Test NHLStats initialization."""
    assert isinstance(nhl_stats, NHLStats)
    assert nhl_stats.headers == {'Accept': 'application/json'}
    assert "TOR" in nhl_stats.teams
    assert "FLA" in nhl_stats.teams

@patch('requests.get')
def test_get_team_roster(mock_get, nhl_stats):
    """Test getting team roster."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "forwards": [{"id": 1, "name": "Forward 1"}],
        "defensemen": [{"id": 2, "name": "Defense 1"}],
        "goalies": [{"id": 3, "name": "Goalie 1"}]
    }
    mock_get.return_value = mock_response

    roster = nhl_stats.get_team_roster("TOR")
    
    assert len(roster) == 3
    assert roster[0]["id"] == 1
    assert roster[1]["id"] == 2
    assert roster[2]["id"] == 3
    mock_get.assert_called_once_with(
        f"{NHL_API_BASE}/roster/TOR/current",
        headers=nhl_stats.headers
    )

@patch('requests.get')
def test_get_player_stats(mock_get, nhl_stats, mock_player_response):
    """Test getting player stats."""
    mock_response = Mock()
    mock_response.json.return_value = mock_player_response
    mock_get.return_value = mock_response

    player_id = 8478483
    stats = nhl_stats.get_player_stats(player_id)
    
    assert stats["playerId"] == player_id
    assert stats["firstName"]["default"] == "John"
    mock_get.assert_called_once_with(
        f"{NHL_API_BASE}/player/{player_id}/landing",
        headers=nhl_stats.headers
    )

@patch('builtins.open', create=True)
@patch.object(NHLStats, 'get_all_team_stats')
def test_save_team_stats(mock_get_stats, mock_open, nhl_stats):
    """Test saving team stats to file."""
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    test_data = {"TOR": [{"player": "data"}]}
    mock_get_stats.return_value = test_data
    
    nhl_stats.save_team_stats("test.json")
    
    mock_file.write.assert_called_once_with(json.dumps(test_data, indent=2))
    mock_open.assert_called_once_with("test.json", "w") 