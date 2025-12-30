"""
Action Executor - Mouse/keyboard automation for poker actions

Provides two execution modes:
- FULL_AUTO: Automatically clicks buttons and enters bets
- SEMI_AUTO: Displays recommendation, waits for user confirmation

Safety features:
- Emergency stop hotkey (ESC)
- Configurable action delays
- Visual indicators when active
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Dict
import time
import random
import threading


class ExecutionMode(Enum):
    """Bot execution modes"""
    FULL_AUTO = auto()    # Fully automated - bot clicks everything
    SEMI_AUTO = auto()    # Semi-automated - bot suggests, user confirms
    DISABLED = auto()     # No execution - suggestion only


@dataclass
class ButtonRegion:
    """Screen region for a button"""
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of button"""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class ActionExecutorConfig:
    """Configuration for action executor"""
    # Timing
    min_delay: float = 1.0          # Minimum delay before action (seconds)
    max_delay: float = 3.0          # Maximum delay before action
    click_duration: float = 0.1     # How long to hold click
    typing_delay: float = 0.05      # Delay between keystrokes
    
    # Safety
    emergency_stop_key: str = 'escape'
    confirmation_key: str = 'space'  # Key to confirm in semi-auto
    
    # Humanization
    add_mouse_jitter: bool = True
    jitter_pixels: int = 5
    
    # Regions (set during setup)
    fold_button: Optional[ButtonRegion] = None
    check_button: Optional[ButtonRegion] = None
    call_button: Optional[ButtonRegion] = None
    raise_button: Optional[ButtonRegion] = None
    bet_input: Optional[ButtonRegion] = None
    all_in_button: Optional[ButtonRegion] = None


class ActionExecutor:
    """
    Executes poker actions via mouse/keyboard automation.
    
    Supports two modes:
    - FULL_AUTO: Bot automatically clicks buttons
    - SEMI_AUTO: Bot suggests action, user confirms with hotkey
    """
    
    def __init__(self, config: Optional[ActionExecutorConfig] = None):
        self.config = config or ActionExecutorConfig()
        self.mode = ExecutionMode.SEMI_AUTO  # Default to safer mode
        self.is_active = False
        self.emergency_stop = False
        
        # Callbacks
        self.on_action_suggested: Optional[Callable] = None
        self.on_action_executed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Pending action in semi-auto mode
        self.pending_action: Optional[Dict] = None
        
        # Try to import pyautogui
        self._pyautogui = None
        self._init_automation()
        
    def _init_automation(self) -> None:
        """Initialize automation library"""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = 0.1      # Small pause between actions
            self._pyautogui = pyautogui
        except ImportError:
            print("Warning: pyautogui not available. Install with: pip install pyautogui")
            self._pyautogui = None
    
    def set_mode(self, mode: ExecutionMode) -> None:
        """Set execution mode"""
        self.mode = mode
        
    def start(self) -> None:
        """Start the executor"""
        self.is_active = True
        self.emergency_stop = False
        
    def stop(self) -> None:
        """Stop the executor"""
        self.is_active = False
        self.pending_action = None
        
    def emergency_halt(self) -> None:
        """Emergency stop all actions"""
        self.emergency_stop = True
        self.is_active = False
        self.pending_action = None
        
    def execute_fold(self) -> bool:
        """Execute fold action"""
        return self._execute_button_click('fold', self.config.fold_button)
    
    def execute_check(self) -> bool:
        """Execute check action"""
        return self._execute_button_click('check', self.config.check_button)
    
    def execute_call(self) -> bool:
        """Execute call action"""
        return self._execute_button_click('call', self.config.call_button)
    
    def execute_raise(self, amount: float) -> bool:
        """Execute raise action with specific amount"""
        if self.mode == ExecutionMode.SEMI_AUTO:
            self.pending_action = {'action': 'raise', 'amount': amount}
            if self.on_action_suggested:
                self.on_action_suggested('raise', amount)
            return True
            
        return self._execute_raise_with_amount(amount)
    
    def execute_bet(self, amount: float) -> bool:
        """Execute bet action with specific amount"""
        return self.execute_raise(amount)  # Same mechanics
    
    def execute_all_in(self) -> bool:
        """Execute all-in action"""
        if self.config.all_in_button:
            return self._execute_button_click('all_in', self.config.all_in_button)
        # Fallback: click raise and enter max
        return self._execute_raise_with_amount(float('inf'))
    
    def suggest_action(self, action: str, amount: float = 0.0) -> None:
        """
        Suggest an action without executing.
        Used in SEMI_AUTO mode.
        """
        self.pending_action = {'action': action, 'amount': amount}
        if self.on_action_suggested:
            self.on_action_suggested(action, amount)
    
    def confirm_pending_action(self) -> bool:
        """
        Execute the pending action (SEMI_AUTO mode).
        Called when user presses confirmation key.
        """
        if not self.pending_action:
            return False
            
        action = self.pending_action['action']
        amount = self.pending_action['amount']
        self.pending_action = None
        
        # Execute the action
        if action == 'fold':
            return self._do_execute_fold()
        elif action == 'check':
            return self._do_execute_check()
        elif action == 'call':
            return self._do_execute_call()
        elif action in ('raise', 'bet'):
            return self._execute_raise_with_amount(amount)
        elif action == 'all_in':
            return self._do_execute_all_in()
            
        return False
    
    def _execute_button_click(self, action_name: str, button: Optional[ButtonRegion]) -> bool:
        """Execute a button click action"""
        if self.mode == ExecutionMode.SEMI_AUTO:
            self.pending_action = {'action': action_name, 'amount': 0}
            if self.on_action_suggested:
                self.on_action_suggested(action_name, 0)
            return True
        
        if self.mode == ExecutionMode.DISABLED:
            return False
            
        # FULL_AUTO mode
        if action_name == 'fold':
            return self._do_execute_fold()
        elif action_name == 'check':
            return self._do_execute_check()
        elif action_name == 'call':
            return self._do_execute_call()
        elif action_name == 'all_in':
            return self._do_execute_all_in()
            
        return False
    
    def _do_execute_fold(self) -> bool:
        """Actually execute fold"""
        if not self.config.fold_button or not self._pyautogui:
            return False
        return self._click_button(self.config.fold_button)
    
    def _do_execute_check(self) -> bool:
        """Actually execute check"""
        if not self.config.check_button or not self._pyautogui:
            return False
        return self._click_button(self.config.check_button)
    
    def _do_execute_call(self) -> bool:
        """Actually execute call"""
        if not self.config.call_button or not self._pyautogui:
            return False
        return self._click_button(self.config.call_button)
    
    def _do_execute_all_in(self) -> bool:
        """Actually execute all-in"""
        if self.config.all_in_button and self._pyautogui:
            return self._click_button(self.config.all_in_button)
        return False
    
    def _execute_raise_with_amount(self, amount: float) -> bool:
        """Execute raise with specific amount"""
        if not self._pyautogui:
            return False
            
        if self.emergency_stop or not self.is_active:
            return False
        
        try:
            # Add human-like delay
            self._human_delay()
            
            # Click bet input field
            if self.config.bet_input:
                self._click_button(self.config.bet_input)
                time.sleep(0.2)
                
                # Clear existing amount
                self._pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.1)
                
                # Type amount
                amount_str = str(int(amount)) if amount != float('inf') else ""
                self._pyautogui.typewrite(amount_str, interval=self.config.typing_delay)
                time.sleep(0.1)
            
            # Click raise button
            if self.config.raise_button:
                self._click_button(self.config.raise_button)
            
            if self.on_action_executed:
                self.on_action_executed('raise', amount)
                
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def _click_button(self, button: ButtonRegion) -> bool:
        """Click a button with human-like behavior"""
        if not self._pyautogui or self.emergency_stop:
            return False
            
        try:
            # Add human-like delay
            self._human_delay()
            
            # Get click position with optional jitter
            x, y = button.center
            if self.config.add_mouse_jitter:
                x += random.randint(-self.config.jitter_pixels, self.config.jitter_pixels)
                y += random.randint(-self.config.jitter_pixels, self.config.jitter_pixels)
            
            # Move and click
            self._pyautogui.moveTo(x, y, duration=random.uniform(0.1, 0.3))
            time.sleep(random.uniform(0.05, 0.15))
            self._pyautogui.click(x, y)
            
            if self.on_action_executed:
                self.on_action_executed(button.label, 0)
                
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def _human_delay(self) -> None:
        """Add random human-like delay before action"""
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        
        # Check for emergency stop during delay
        start = time.time()
        while time.time() - start < delay:
            if self.emergency_stop:
                raise InterruptedError("Emergency stop triggered")
            time.sleep(0.1)
    
    def set_button_region(self, button_name: str, region: Tuple[int, int, int, int]) -> None:
        """Set a button region from (x, y, w, h) tuple"""
        x, y, w, h = region
        button = ButtonRegion(x, y, w, h, button_name)
        
        if button_name == 'fold':
            self.config.fold_button = button
        elif button_name == 'check':
            self.config.check_button = button
        elif button_name == 'call':
            self.config.call_button = button
        elif button_name == 'raise':
            self.config.raise_button = button
        elif button_name == 'bet_input':
            self.config.bet_input = button
        elif button_name == 'all_in':
            self.config.all_in_button = button
    
    def save_config(self, filepath: str) -> None:
        """Save button configuration to file"""
        import json
        
        config_dict = {
            'min_delay': self.config.min_delay,
            'max_delay': self.config.max_delay,
            'buttons': {}
        }
        
        for name in ['fold', 'check', 'call', 'raise', 'bet_input', 'all_in']:
            button = getattr(self.config, f'{name}_button', None)
            if button:
                config_dict['buttons'][name] = {
                    'x': button.x,
                    'y': button.y,
                    'width': button.width,
                    'height': button.height
                }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, filepath: str) -> None:
        """Load button configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.config.min_delay = config_dict.get('min_delay', 1.0)
        self.config.max_delay = config_dict.get('max_delay', 3.0)
        
        for name, region in config_dict.get('buttons', {}).items():
            self.set_button_region(name, (
                region['x'], region['y'],
                region['width'], region['height']
            ))

