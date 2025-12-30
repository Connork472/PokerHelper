"""
PokerBot - Automated Poker Assistant Module

This module provides a system-agnostic poker bot that can:
- Capture game state from any poker client via screen analysis
- Make decisions using Monte Carlo simulation with multiple play styles
- Execute actions via mouse/keyboard automation

Play Styles:
- STRICT: Pure equity-based decisions, no bluffs
- BLUFF: Exploitative play with aggression and deception
- BALANCED: GTO-approximation with mixed strategies
"""

from .game_state import GameState, GamePhase, Action
from .decision_engine import DecisionEngine, PlayStyle
from .bet_sizing import BetSizingController
from .action_executor import ActionExecutor, ExecutionMode
from .screen_analyzer import ScreenAnalyzer

__all__ = [
    'GameState',
    'GamePhase', 
    'Action',
    'DecisionEngine',
    'PlayStyle',
    'BetSizingController',
    'ActionExecutor',
    'ExecutionMode',
    'ScreenAnalyzer',
]

__version__ = '1.0.0'

