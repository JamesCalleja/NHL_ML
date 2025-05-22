"""
NHL API constants and configuration values.
"""

# Team IDs
TEAM_IDS = {
    "TOR": 10,  # Toronto Maple Leafs
    "FLA": 13,  # Florida Panthers
    "BOS": 6,   # Boston Bruins
    "TBL": 14,  # Tampa Bay Lightning
    "MTL": 8,   # Montreal Canadiens
    "OTT": 9,   # Ottawa Senators
    "BUF": 7,   # Buffalo Sabres
    "DET": 17,  # Detroit Red Wings
}

# API Endpoints
API_BASE = "https://api-web.nhle.com/v1"
ROSTER_ENDPOINT = "/roster/{team}/current"
PLAYER_ENDPOINT = "/player/{id}/landing"
TEAM_STATS_ENDPOINT = "/club-stats/team/{id}/now"

# Data Processing
DEFAULT_SEASON = "20232024"
MIN_GAMES_PLAYED = 10
STATS_FEATURES = [
    'games_played',
    'goals',
    'assists',
    'points',
    'plus_minus',
    'pim',
    'shots',
    'shooting_pct',
    'powerplay_goals',
    'powerplay_points'
]
