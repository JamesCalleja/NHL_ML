"""
NHL API interaction module for fetching player and team statistics.
"""

from .nhl_api import NHLStats
from .team_stats import TeamStats

__all__ = ['NHLStats', 'TeamStats']
