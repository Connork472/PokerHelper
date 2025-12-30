"""
Bet Sizing Controller - Configurable bet sizing per street

Provides intelligent bet sizing based on:
- Street (preflop/flop/turn/river)
- Stack-to-pot ratio (SPR)
- Betting action (open, 3-bet, c-bet, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import json


class BetType(Enum):
    """Types of bets"""
    OPEN = "open"           # First bet/raise preflop
    THREE_BET = "3bet"      # Re-raise preflop
    FOUR_BET = "4bet"       # Re-re-raise preflop
    CBET = "cbet"           # Continuation bet
    PROBE = "probe"         # Betting into aggressor
    VALUE = "value"         # Value bet
    BLUFF = "bluff"         # Bluff bet
    ALL_IN = "all_in"       # All-in


@dataclass
class StreetSizing:
    """Bet sizing configuration for a single street"""
    # Sizing as pot multiplier (e.g., 0.5 = half pot)
    small: float = 0.33
    medium: float = 0.5
    large: float = 0.75
    overbet: float = 1.25
    
    def get_size(self, pot: float, size_type: str = "medium") -> float:
        """Get bet size for given pot"""
        multipliers = {
            "small": self.small,
            "medium": self.medium,
            "large": self.large,
            "overbet": self.overbet
        }
        return pot * multipliers.get(size_type, self.medium)


@dataclass
class PreflopSizing:
    """Preflop-specific sizing"""
    open_raise_bb: float = 2.5          # Open raise in BB
    open_raise_btn_bb: float = 2.2      # Button open
    three_bet_multiplier: float = 3.0   # 3-bet as multiplier of open
    four_bet_multiplier: float = 2.2    # 4-bet as multiplier of 3-bet
    limp_raise_bb: float = 4.0          # Raise after limpers
    
    def get_open_size(self, big_blind: float, position: str = "middle") -> float:
        """Get open raise size"""
        if position in ("btn", "button", "sb"):
            return big_blind * self.open_raise_btn_bb
        return big_blind * self.open_raise_bb
    
    def get_3bet_size(self, open_raise: float, in_position: bool = True) -> float:
        """Get 3-bet size"""
        multiplier = self.three_bet_multiplier
        if not in_position:
            multiplier += 0.5  # Larger OOP
        return open_raise * multiplier
    
    def get_4bet_size(self, three_bet: float) -> float:
        """Get 4-bet size"""
        return three_bet * self.four_bet_multiplier


@dataclass
class BetSizingController:
    """
    Main bet sizing controller with configurable presets.
    
    Handles all bet sizing decisions based on:
    - Street
    - Action type
    - Stack-to-pot ratio
    - Position
    """
    
    # Street-specific sizing
    preflop: PreflopSizing = field(default_factory=PreflopSizing)
    flop: StreetSizing = field(default_factory=lambda: StreetSizing(0.33, 0.5, 0.75, 1.0))
    turn: StreetSizing = field(default_factory=lambda: StreetSizing(0.5, 0.66, 0.85, 1.25))
    river: StreetSizing = field(default_factory=lambda: StreetSizing(0.5, 0.75, 1.0, 1.5))
    
    # All-in thresholds
    all_in_spr_threshold: float = 2.0      # Go all-in if SPR below this
    commit_threshold: float = 0.33          # Commit if bet > this % of stack
    
    # Randomization for balance
    sizing_variance: float = 0.1           # +/- variance for unpredictability
    
    def get_bet_size(
        self,
        street: str,
        pot: float,
        stack: float,
        big_blind: float,
        bet_type: str = "medium",
        to_call: float = 0.0,
        raise_to: float = 0.0
    ) -> float:
        """
        Calculate optimal bet size for the situation.
        
        Args:
            street: Current street (preflop/flop/turn/river)
            pot: Current pot size
            stack: Hero's stack
            big_blind: Big blind amount
            bet_type: Size category (small/medium/large/overbet)
            to_call: Amount to call (for raises)
            raise_to: Previous raise amount (for re-raises)
            
        Returns:
            Recommended bet/raise amount
        """
        import random
        
        # Calculate SPR
        spr = stack / pot if pot > 0 else float('inf')
        
        # Check for all-in situations
        if spr <= self.all_in_spr_threshold:
            return stack  # All-in
        
        # Get base sizing
        if street == "preflop":
            if raise_to > 0:
                # This is a re-raise
                if to_call > 0:
                    size = self.preflop.get_3bet_size(raise_to)
                else:
                    size = self.preflop.get_open_size(big_blind)
            else:
                size = self.preflop.get_open_size(big_blind)
        else:
            # Postflop sizing
            sizing = getattr(self, street, self.flop)
            size = sizing.get_size(pot, bet_type)
            
            # Add call amount for raises
            if to_call > 0:
                size += to_call
        
        # Add randomization for unpredictability
        if self.sizing_variance > 0:
            variance = size * self.sizing_variance
            size += random.uniform(-variance, variance)
        
        # Round to reasonable amount
        size = self._round_bet(size, big_blind)
        
        # Cap at stack
        size = min(size, stack)
        
        # Check if we're committed
        if size > stack * self.commit_threshold:
            # Consider going all-in if close
            remaining = stack - size
            if remaining < pot * 0.5:
                size = stack  # All-in
        
        return size
    
    def _round_bet(self, amount: float, big_blind: float) -> float:
        """Round bet to sensible amount"""
        if amount < big_blind * 10:
            # Round to 0.5 BB for small bets
            return round(amount * 2 / big_blind) * big_blind / 2
        elif amount < big_blind * 100:
            # Round to 1 BB for medium bets
            return round(amount / big_blind) * big_blind
        else:
            # Round to 5 BB for large bets
            return round(amount / (5 * big_blind)) * 5 * big_blind
    
    def get_value_bet_size(
        self,
        street: str,
        pot: float,
        stack: float,
        hand_strength: float
    ) -> float:
        """Get value bet size based on hand strength"""
        # Stronger hands = larger bets for value
        if hand_strength > 0.85:
            size_type = "large"
        elif hand_strength > 0.7:
            size_type = "medium"
        else:
            size_type = "small"
            
        return self.get_bet_size(street, pot, stack, 1.0, size_type)
    
    def get_bluff_size(
        self,
        street: str,
        pot: float,
        stack: float
    ) -> float:
        """Get bluff bet size - typically smaller to risk less"""
        # Bluffs use smaller sizing to risk less
        return self.get_bet_size(street, pot, stack, 1.0, "small")
    
    def should_go_all_in(
        self,
        stack: float,
        pot: float,
        bet_size: float
    ) -> bool:
        """Determine if we should just go all-in instead of betting"""
        spr = stack / pot if pot > 0 else float('inf')
        
        # All-in if SPR is very low
        if spr <= self.all_in_spr_threshold:
            return True
        
        # All-in if bet commits most of stack
        remaining = stack - bet_size
        if remaining < pot * 0.3:
            return True
            
        return False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'preflop': {
                'open_raise_bb': self.preflop.open_raise_bb,
                'three_bet_multiplier': self.preflop.three_bet_multiplier,
                'four_bet_multiplier': self.preflop.four_bet_multiplier
            },
            'flop': {
                'small': self.flop.small,
                'medium': self.flop.medium,
                'large': self.flop.large
            },
            'turn': {
                'small': self.turn.small,
                'medium': self.turn.medium,
                'large': self.turn.large
            },
            'river': {
                'small': self.river.small,
                'medium': self.river.medium,
                'large': self.river.large
            },
            'all_in_spr_threshold': self.all_in_spr_threshold
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BetSizingController':
        """Create from dictionary"""
        controller = cls()
        
        if 'preflop' in data:
            p = data['preflop']
            controller.preflop = PreflopSizing(
                open_raise_bb=p.get('open_raise_bb', 2.5),
                three_bet_multiplier=p.get('three_bet_multiplier', 3.0),
                four_bet_multiplier=p.get('four_bet_multiplier', 2.2)
            )
        
        for street in ['flop', 'turn', 'river']:
            if street in data:
                s = data[street]
                setattr(controller, street, StreetSizing(
                    small=s.get('small', 0.33),
                    medium=s.get('medium', 0.5),
                    large=s.get('large', 0.75),
                    overbet=s.get('overbet', 1.0)
                ))
        
        if 'all_in_spr_threshold' in data:
            controller.all_in_spr_threshold = data['all_in_spr_threshold']
            
        return controller
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'BetSizingController':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

