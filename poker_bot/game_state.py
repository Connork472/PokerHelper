"""
Game State Manager - Tracks the complete poker game state

Monitors:
- Current hand phase (preflop/flop/turn/river)
- Detected cards (hole cards + community)
- Pot size, current bet, stack sizes
- Hand history for the current session
- Automatic hand reset detection
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import json


class GamePhase(Enum):
    """Poker hand phases"""
    WAITING = auto()      # No hand in progress
    PREFLOP = auto()      # Hole cards dealt, no community cards
    FLOP = auto()         # 3 community cards
    TURN = auto()         # 4 community cards
    RIVER = auto()        # 5 community cards
    SHOWDOWN = auto()     # Hand complete


class Action(Enum):
    """Possible poker actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class PlayerInfo:
    """Information about a player at the table"""
    position: int
    stack: float = 0.0
    current_bet: float = 0.0
    is_active: bool = True
    is_hero: bool = False


@dataclass
class HandHistory:
    """Record of a completed hand"""
    timestamp: datetime
    hole_cards: List[str]
    community_cards: List[str]
    actions_taken: List[Dict]
    result: Optional[str] = None
    profit_loss: float = 0.0


@dataclass 
class GameState:
    """
    Complete game state for the current hand.
    
    Tracks all information needed for decision making:
    - Cards (hole + community)
    - Betting state (pot, bets, stacks)
    - Game phase
    - Available actions
    """
    
    # Card state
    hole_cards: List[str] = field(default_factory=list)
    community_cards: List[str] = field(default_factory=list)
    
    # Betting state
    pot_size: float = 0.0
    current_bet: float = 0.0
    hero_stack: float = 0.0
    to_call: float = 0.0
    min_raise: float = 0.0
    
    # Game info
    phase: GamePhase = GamePhase.WAITING
    num_players: int = 6
    hero_position: int = 0
    button_position: int = 0
    
    # Blinds
    small_blind: float = 0.5
    big_blind: float = 1.0
    
    # Available actions
    can_check: bool = False
    can_call: bool = False
    can_raise: bool = False
    can_fold: bool = True
    
    # Players
    players: Dict[int, PlayerInfo] = field(default_factory=dict)
    active_players: int = 0
    
    # Session tracking
    hand_number: int = 0
    session_profit: float = 0.0
    hands_played: int = 0
    
    # Hand history
    current_hand_actions: List[Dict] = field(default_factory=list)
    hand_history: List[HandHistory] = field(default_factory=list)
    
    # Timestamps
    last_update: Optional[datetime] = None
    hand_start_time: Optional[datetime] = None
    
    def update_phase(self) -> None:
        """Determine game phase based on community cards"""
        num_community = len(self.community_cards)
        
        if not self.hole_cards:
            self.phase = GamePhase.WAITING
        elif num_community == 0:
            self.phase = GamePhase.PREFLOP
        elif num_community == 3:
            self.phase = GamePhase.FLOP
        elif num_community == 4:
            self.phase = GamePhase.TURN
        elif num_community == 5:
            self.phase = GamePhase.RIVER
            
        self.last_update = datetime.now()
    
    def detect_new_hand(self, new_hole_cards: List[str]) -> bool:
        """
        Detect if a new hand has started.
        Returns True if we should reset for a new hand.
        """
        # New hand if hole cards changed significantly
        if set(new_hole_cards) != set(self.hole_cards) and len(new_hole_cards) == 2:
            return True
        
        # New hand if we had 5 community cards and now have 0
        if len(self.community_cards) == 5 and len(new_hole_cards) == 2:
            return True
            
        return False
    
    def start_new_hand(self, hole_cards: List[str]) -> None:
        """Reset state for a new hand"""
        # Save current hand to history if it was played
        if self.hole_cards and self.current_hand_actions:
            history = HandHistory(
                timestamp=self.hand_start_time or datetime.now(),
                hole_cards=self.hole_cards.copy(),
                community_cards=self.community_cards.copy(),
                actions_taken=self.current_hand_actions.copy()
            )
            self.hand_history.append(history)
        
        # Reset for new hand
        self.hole_cards = hole_cards
        self.community_cards = []
        self.pot_size = self.small_blind + self.big_blind
        self.current_bet = self.big_blind
        self.to_call = self.big_blind
        self.current_hand_actions = []
        self.hand_number += 1
        self.hands_played += 1
        self.hand_start_time = datetime.now()
        self.phase = GamePhase.PREFLOP
        
    def update_cards(self, hole_cards: List[str], community_cards: List[str]) -> None:
        """Update card state and detect phase changes"""
        # Check for new hand
        if self.detect_new_hand(hole_cards):
            self.start_new_hand(hole_cards)
            return
            
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.update_phase()
    
    def record_action(self, action: Action, amount: float = 0.0) -> None:
        """Record an action taken"""
        action_record = {
            'phase': self.phase.name,
            'action': action.value,
            'amount': amount,
            'pot_before': self.pot_size,
            'timestamp': datetime.now().isoformat()
        }
        self.current_hand_actions.append(action_record)
        
    def update_available_actions(self) -> None:
        """Determine which actions are currently available"""
        self.can_fold = True  # Can always fold
        self.can_check = (self.to_call == 0)
        self.can_call = (self.to_call > 0 and self.to_call <= self.hero_stack)
        self.can_raise = (self.hero_stack > self.to_call)
        
    def get_spr(self) -> float:
        """Calculate Stack-to-Pot Ratio"""
        if self.pot_size == 0:
            return float('inf')
        return self.hero_stack / self.pot_size
    
    def get_pot_odds(self) -> float:
        """Calculate pot odds as a percentage"""
        if self.to_call == 0:
            return 0.0
        total_pot = self.pot_size + self.to_call
        return self.to_call / total_pot
        
    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization"""
        return {
            'hole_cards': self.hole_cards,
            'community_cards': self.community_cards,
            'pot_size': self.pot_size,
            'current_bet': self.current_bet,
            'hero_stack': self.hero_stack,
            'to_call': self.to_call,
            'phase': self.phase.name,
            'num_players': self.num_players,
            'active_players': self.active_players,
            'hand_number': self.hand_number,
            'can_check': self.can_check,
            'can_call': self.can_call,
            'can_raise': self.can_raise,
            'spr': self.get_spr(),
            'pot_odds': self.get_pot_odds()
        }
    
    def save_to_file(self, filepath: str = "output/bot_state.json") -> None:
        """Save current state to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def __str__(self) -> str:
        return (f"GameState(phase={self.phase.name}, "
                f"hole={self.hole_cards}, "
                f"board={self.community_cards}, "
                f"pot={self.pot_size}, "
                f"to_call={self.to_call})")

