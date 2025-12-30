"""
Decision Engine - Poker decision making with multiple play styles

Play Styles:
- STRICT: Pure equity-based. Only acts when equity exceeds threshold. No bluffs.
- BLUFF: Exploitative. Adds aggression, bluffs weak hands, slow-plays monsters.
- BALANCED: GTO-approximation. Mixed strategy with weighted randomization.

Uses the existing Monte Carlo simulator for equity calculations.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import random
import sys
import os

# Add path for Monte Carlo simulator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'simulator'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulator'))

try:
    from poker_cli_session import tokenize_cards, to_treys, simulate, kelly_even_money
except ImportError:
    # Fallback definitions if import fails
    def tokenize_cards(s): return s.split()
    def to_treys(cards): return cards
    def simulate(p, h, b, t): return (0.5, 0.0, 0.5)
    def kelly_even_money(e): return max(0, 2*e - 1)

from .game_state import GameState, GamePhase, Action
from .bet_sizing import BetSizingController


class PlayStyle(Enum):
    """Available play styles"""
    STRICT = auto()      # Pure equity-based, no bluffs
    BLUFF = auto()       # Exploitative with aggression
    BALANCED = auto()    # GTO-approximation


@dataclass
class Decision:
    """Represents a poker decision"""
    action: Action
    amount: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    equity: float = 0.0
    pot_odds: float = 0.0


@dataclass 
class PlayStyleConfig:
    """Configuration for a play style"""
    # Equity thresholds
    fold_threshold: float = 0.3        # Fold if equity below this
    call_threshold: float = 0.4        # Call if equity above this
    raise_threshold: float = 0.55      # Raise if equity above this
    value_bet_threshold: float = 0.65  # Value bet if equity above this
    
    # Bluffing parameters
    bluff_frequency: float = 0.0       # How often to bluff (0-1)
    semi_bluff_frequency: float = 0.0  # How often to semi-bluff with draws
    slow_play_frequency: float = 0.0   # How often to slow-play monsters
    
    # Aggression
    aggression_factor: float = 1.0     # Multiplier for bet sizing
    three_bet_light_frequency: float = 0.0  # Light 3-betting
    
    # Position adjustments
    position_bonus: float = 0.05       # Equity bonus for position
    
    # Randomization for balance
    action_variance: float = 0.0       # Random variance in thresholds


# Predefined play style configurations
STRICT_CONFIG = PlayStyleConfig(
    fold_threshold=0.35,
    call_threshold=0.45,
    raise_threshold=0.6,
    value_bet_threshold=0.7,
    bluff_frequency=0.0,
    semi_bluff_frequency=0.0,
    slow_play_frequency=0.0,
    aggression_factor=0.9,
    three_bet_light_frequency=0.0,
    position_bonus=0.03,
    action_variance=0.0
)

BLUFF_CONFIG = PlayStyleConfig(
    fold_threshold=0.25,
    call_threshold=0.35,
    raise_threshold=0.5,
    value_bet_threshold=0.6,
    bluff_frequency=0.2,           # Bluff 20% of the time
    semi_bluff_frequency=0.4,      # Semi-bluff draws 40%
    slow_play_frequency=0.25,      # Slow-play monsters 25%
    aggression_factor=1.3,         # 30% more aggressive sizing
    three_bet_light_frequency=0.15,
    position_bonus=0.08,
    action_variance=0.05
)

BALANCED_CONFIG = PlayStyleConfig(
    fold_threshold=0.3,
    call_threshold=0.4,
    raise_threshold=0.55,
    value_bet_threshold=0.65,
    bluff_frequency=0.12,          # ~12% bluff frequency (GTO-ish)
    semi_bluff_frequency=0.3,
    slow_play_frequency=0.1,
    aggression_factor=1.1,
    three_bet_light_frequency=0.08,
    position_bonus=0.05,
    action_variance=0.03
)


class DecisionEngine:
    """
    Makes poker decisions based on game state and play style.
    
    Uses Monte Carlo simulation for equity calculation and
    applies play style rules for final decision.
    """
    
    def __init__(
        self,
        play_style: PlayStyle = PlayStyle.BALANCED,
        bet_sizing: Optional[BetSizingController] = None,
        num_simulations: int = 10000
    ):
        self.play_style = play_style
        self.bet_sizing = bet_sizing or BetSizingController()
        self.num_simulations = num_simulations
        
        # Load style config
        self._load_style_config()
        
        # Decision history for pattern analysis
        self.decision_history: List[Decision] = []
        
    def _load_style_config(self) -> None:
        """Load configuration for current play style"""
        configs = {
            PlayStyle.STRICT: STRICT_CONFIG,
            PlayStyle.BLUFF: BLUFF_CONFIG,
            PlayStyle.BALANCED: BALANCED_CONFIG
        }
        self.config = configs.get(self.play_style, BALANCED_CONFIG)
    
    def set_play_style(self, style: PlayStyle) -> None:
        """Change play style"""
        self.play_style = style
        self._load_style_config()
    
    def calculate_equity(
        self,
        hole_cards: List[str],
        board_cards: List[str],
        num_opponents: int = 1
    ) -> Tuple[float, float, float]:
        """
        Calculate equity using Monte Carlo simulation.
        
        Returns:
            Tuple of (win_probability, tie_probability, equity)
        """
        try:
            # Convert cards to treys format
            my_cards = to_treys(hole_cards)
            board = to_treys(board_cards) if board_cards else []
            
            # Run simulation
            win_p, tie_p, equity = simulate(
                num_opponents + 1,  # Total players including hero
                my_cards,
                board,
                self.num_simulations
            )
            
            return win_p, tie_p, equity
            
        except Exception as e:
            print(f"Equity calculation error: {e}")
            return 0.5, 0.0, 0.5  # Default to 50% on error
    
    def make_decision(self, game_state: GameState) -> Decision:
        """
        Make a poker decision based on current game state.
        
        This is the main entry point for decision making.
        """
        # Validate we have cards
        if not game_state.hole_cards or len(game_state.hole_cards) != 2:
            return Decision(
                action=Action.FOLD,
                reasoning="No valid hole cards detected"
            )
        
        # Calculate equity
        num_opponents = max(1, game_state.active_players - 1)
        win_p, tie_p, equity = self.calculate_equity(
            game_state.hole_cards,
            game_state.community_cards,
            num_opponents
        )
        
        # Calculate pot odds
        pot_odds = game_state.get_pot_odds()
        
        # Get decision based on play style
        decision = self._decide_action(game_state, equity, pot_odds)
        decision.equity = equity
        decision.pot_odds = pot_odds
        
        # Record decision
        self.decision_history.append(decision)
        
        return decision
    
    def _decide_action(
        self,
        game_state: GameState,
        equity: float,
        pot_odds: float
    ) -> Decision:
        """Determine action based on play style"""
        
        # Apply position bonus
        adjusted_equity = equity
        if game_state.hero_position > game_state.button_position:
            adjusted_equity += self.config.position_bonus
        
        # Add variance for unpredictability
        if self.config.action_variance > 0:
            variance = random.uniform(
                -self.config.action_variance,
                self.config.action_variance
            )
            adjusted_equity += variance
        
        # Get street for sizing
        street = game_state.phase.name.lower()
        if street == "waiting":
            street = "preflop"
        
        # Check for bluff opportunity
        if self._should_bluff(game_state, adjusted_equity):
            return self._create_bluff_decision(game_state, street)
        
        # Check for slow-play
        if self._should_slow_play(game_state, adjusted_equity):
            return self._create_slow_play_decision(game_state)
        
        # Standard decision logic
        return self._standard_decision(
            game_state, adjusted_equity, pot_odds, street
        )
    
    def _standard_decision(
        self,
        game_state: GameState,
        equity: float,
        pot_odds: float,
        street: str
    ) -> Decision:
        """Standard equity-based decision making"""
        
        # Facing a bet
        if game_state.to_call > 0:
            # Not enough equity to continue
            if equity < self.config.fold_threshold:
                return Decision(
                    action=Action.FOLD,
                    reasoning=f"Equity {equity:.1%} below fold threshold {self.config.fold_threshold:.1%}"
                )
            
            # Have odds to call but not raise
            elif equity < self.config.raise_threshold:
                # Check pot odds
                if equity > pot_odds:
                    return Decision(
                        action=Action.CALL,
                        amount=game_state.to_call,
                        confidence=equity - pot_odds,
                        reasoning=f"Equity {equity:.1%} > pot odds {pot_odds:.1%}, calling"
                    )
                else:
                    return Decision(
                        action=Action.FOLD,
                        reasoning=f"Equity {equity:.1%} < pot odds {pot_odds:.1%}"
                    )
            
            # Strong hand - raise
            else:
                raise_amount = self.bet_sizing.get_bet_size(
                    street,
                    game_state.pot_size,
                    game_state.hero_stack,
                    game_state.big_blind,
                    "large" if equity > self.config.value_bet_threshold else "medium",
                    game_state.to_call
                )
                
                # Check if should all-in
                if self.bet_sizing.should_go_all_in(
                    game_state.hero_stack,
                    game_state.pot_size,
                    raise_amount
                ):
                    return Decision(
                        action=Action.ALL_IN,
                        amount=game_state.hero_stack,
                        confidence=equity,
                        reasoning=f"Strong equity {equity:.1%}, going all-in"
                    )
                
                return Decision(
                    action=Action.RAISE,
                    amount=raise_amount,
                    confidence=equity,
                    reasoning=f"Strong equity {equity:.1%}, raising for value"
                )
        
        # No bet to face - we can check or bet
        else:
            if game_state.can_check:
                # Weak hand - check
                if equity < self.config.call_threshold:
                    return Decision(
                        action=Action.CHECK,
                        reasoning=f"Equity {equity:.1%} too weak to bet"
                    )
                
                # Medium hand - small bet or check
                elif equity < self.config.value_bet_threshold:
                    if random.random() < 0.5:
                        bet_amount = self.bet_sizing.get_bet_size(
                            street,
                            game_state.pot_size,
                            game_state.hero_stack,
                            game_state.big_blind,
                            "small"
                        )
                        return Decision(
                            action=Action.BET,
                            amount=bet_amount,
                            confidence=equity,
                            reasoning=f"Medium equity {equity:.1%}, small bet"
                        )
                    else:
                        return Decision(
                            action=Action.CHECK,
                            reasoning="Pot control with medium hand"
                        )
                
                # Strong hand - value bet
                else:
                    bet_amount = self.bet_sizing.get_value_bet_size(
                        street,
                        game_state.pot_size,
                        game_state.hero_stack,
                        equity
                    )
                    
                    return Decision(
                        action=Action.BET,
                        amount=bet_amount,
                        confidence=equity,
                        reasoning=f"Strong equity {equity:.1%}, betting for value"
                    )
            
            # Shouldn't reach here in normal play
            return Decision(
                action=Action.CHECK,
                reasoning="Default check"
            )
    
    def _should_bluff(self, game_state: GameState, equity: float) -> bool:
        """Determine if we should bluff"""
        if self.config.bluff_frequency == 0:
            return False
        
        # Only bluff with weak hands
        if equity > self.config.fold_threshold:
            return False
        
        # Random bluff based on frequency
        if random.random() > self.config.bluff_frequency:
            return False
        
        # Better bluff spots:
        # - Late position
        # - Scary boards
        # - Fewer opponents
        bluff_bonus = 0.0
        
        if game_state.active_players <= 2:
            bluff_bonus += 0.1
        
        if game_state.hero_position >= game_state.num_players - 2:
            bluff_bonus += 0.05
        
        # Check if draw for semi-bluff
        if self._has_draw(game_state):
            if random.random() < self.config.semi_bluff_frequency:
                return True
        
        return random.random() < (self.config.bluff_frequency + bluff_bonus)
    
    def _has_draw(self, game_state: GameState) -> bool:
        """Check if we have a draw (simplified)"""
        if len(game_state.community_cards) < 3:
            return False
        
        # Count suits and connected cards (simplified)
        all_cards = game_state.hole_cards + game_state.community_cards
        suits = [c[-1] for c in all_cards]
        
        # Flush draw
        for suit in 'hdcs':
            if suits.count(suit) >= 4:
                return True
        
        return False
    
    def _create_bluff_decision(self, game_state: GameState, street: str) -> Decision:
        """Create a bluff decision"""
        bluff_size = self.bet_sizing.get_bluff_size(
            street,
            game_state.pot_size,
            game_state.hero_stack
        )
        
        bluff_size *= self.config.aggression_factor
        
        return Decision(
            action=Action.BET if game_state.to_call == 0 else Action.RAISE,
            amount=bluff_size,
            confidence=0.3,
            reasoning="Bluffing with weak hand"
        )
    
    def _should_slow_play(self, game_state: GameState, equity: float) -> bool:
        """Determine if we should slow-play"""
        if self.config.slow_play_frequency == 0:
            return False
        
        # Only slow-play very strong hands
        if equity < 0.8:
            return False
        
        # Don't slow-play on wet boards
        if len(game_state.community_cards) >= 3:
            return False
        
        return random.random() < self.config.slow_play_frequency
    
    def _create_slow_play_decision(self, game_state: GameState) -> Decision:
        """Create a slow-play decision"""
        if game_state.to_call > 0:
            return Decision(
                action=Action.CALL,
                amount=game_state.to_call,
                confidence=0.9,
                reasoning="Slow-playing strong hand"
            )
        else:
            return Decision(
                action=Action.CHECK,
                confidence=0.9,
                reasoning="Slow-playing strong hand"
            )
    
    def get_kelly_recommendation(self, equity: float) -> float:
        """Get Kelly criterion bet size recommendation"""
        return kelly_even_money(equity)
    
    def get_decision_stats(self) -> Dict:
        """Get statistics about recent decisions"""
        if not self.decision_history:
            return {}
        
        recent = self.decision_history[-50:]  # Last 50 decisions
        
        action_counts = {}
        total_equity = 0
        
        for d in recent:
            action_name = d.action.value
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            total_equity += d.equity
        
        return {
            'total_decisions': len(recent),
            'action_distribution': action_counts,
            'average_equity': total_equity / len(recent),
            'play_style': self.play_style.name
        }
    
    def reset_history(self) -> None:
        """Clear decision history"""
        self.decision_history.clear()

