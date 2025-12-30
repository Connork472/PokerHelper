#!/usr/bin/env python3
"""
Poker Bot GUI - Automated Poker Assistant Interface

Features:
- Region setup for cards and buttons
- Play style selection (Strict/Bluff/Balanced)
- Auto/Semi-auto execution modes
- Bet sizing configuration
- Live game state display
- Action log and hand history
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import json
import os
import sys
import platform

# Add paths for imports
sys.path.append(os.path.dirname(__file__))

from poker_bot import (
    GameState, GamePhase, Action,
    DecisionEngine, PlayStyle,
    BetSizingController,
    ActionExecutor, ExecutionMode,
    ScreenAnalyzer
)

# ═══════════════════════════════════════════════════════════════════════════════
# FONT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
if platform.system() == 'Darwin':
    FONT_FAMILY = 'Helvetica Neue'
    FONT_MONO = 'Menlo'
elif platform.system() == 'Windows':
    FONT_FAMILY = 'Segoe UI'
    FONT_MONO = 'Consolas'
else:
    FONT_FAMILY = 'DejaVu Sans'
    FONT_MONO = 'DejaVu Sans Mono'

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Dark theme with accent colors
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'bg_dark': '#0d1117',
    'bg_card': '#161b22',
    'bg_elevated': '#21262d',
    'bg_hover': '#30363d',
    'bg_input': '#0d1117',
    'border': '#30363d',
    'border_light': '#484f58',
    'border_focus': '#1f6feb',
    
    'text_primary': '#f0f6fc',
    'text_secondary': '#8b949e',
    'text_muted': '#6e7681',
    
    'accent_green': '#238636',
    'accent_green_hover': '#2ea043',
    'accent_red': '#da3633',
    'accent_red_hover': '#f85149',
    'accent_blue': '#1f6feb',
    'accent_blue_hover': '#388bfd',
    'accent_gold': '#d29922',
    'accent_purple': '#8957e5',
    'accent_cyan': '#39c5cf',
    
    'card_red': '#ef4444',
    'card_black': '#1a1a1a',
    'card_bg': '#ffffff',
    
    'status_active': '#22c55e',
    'status_warning': '#eab308',
    'status_inactive': '#6e7681',
    'status_error': '#ef4444',
}


class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, color='blue', width=160, height=44, icon=None, **kwargs):
        # Get bg from parent or default
        bg = kwargs.pop('bg', COLORS['bg_dark'])
        super().__init__(parent, width=width, height=height, 
                        bg=bg, highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.icon = icon
        self.btn_width = width
        self.btn_height = height
        self.enabled = True
        self.active = False
        
        color_map = {
            'blue': (COLORS['accent_blue'], COLORS['accent_blue_hover']),
            'green': (COLORS['accent_green'], COLORS['accent_green_hover']),
            'red': (COLORS['accent_red'], COLORS['accent_red_hover']),
            'gold': (COLORS['accent_gold'], '#e3b341'),
            'purple': (COLORS['accent_purple'], '#a371f7'),
            'cyan': (COLORS['accent_cyan'], '#56d4dd'),
        }
        self.normal_color, self.hover_color = color_map.get(color, color_map['blue'])
        self.current_color = self.normal_color
        
        self.draw_button()
        
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        
    def draw_button(self):
        self.delete('all')
        
        if not self.enabled:
            color = COLORS['bg_elevated']
            text_color = COLORS['text_muted']
        elif self.active:
            color = self.hover_color
            text_color = COLORS['text_primary']
        else:
            color = self.current_color
            text_color = COLORS['text_primary']
        
        self.create_rounded_rect(2, 2, self.btn_width-2, self.btn_height-2, 
                                radius=10, fill=color, outline='')
        
        display_text = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(self.btn_width//2, self.btn_height//2, text=display_text,
                        fill=text_color, font=(FONT_FAMILY, 11, 'bold'))
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, event):
        if self.enabled and not self.active:
            self.current_color = self.hover_color
            self.draw_button()
            self.config(cursor='hand2')
        
    def on_leave(self, event):
        if not self.active:
            self.current_color = self.normal_color
            self.draw_button()
        
    def on_click(self, event):
        if self.enabled and self.command:
            self.command()
            
    def set_enabled(self, enabled):
        self.enabled = enabled
        self.draw_button()
        
    def set_active(self, active, new_text=None):
        self.active = active
        if new_text:
            self.text = new_text
        self.draw_button()


class CardDisplay(tk.Canvas):
    """Visual card display widget"""
    def __init__(self, parent, card_text="?", width=50, height=70, **kwargs):
        bg = kwargs.pop('bg', COLORS['bg_card'])
        super().__init__(parent, width=width, height=height,
                        bg=bg, highlightthickness=0, **kwargs)
        self.card_text = card_text
        self.card_width = width
        self.card_height = height
        self.draw_card()
        
    def draw_card(self):
        self.delete('all')
        
        if not self.card_text or self.card_text == "?":
            self.create_rounded_rect(2, 2, self.card_width-2, self.card_height-2,
                                    radius=5, fill=COLORS['bg_elevated'], 
                                    outline=COLORS['border'])
            self.create_text(self.card_width//2, self.card_height//2, text="?",
                           fill=COLORS['text_muted'], font=(FONT_FAMILY, 16, 'bold'))
        else:
            self.create_rounded_rect(2, 2, self.card_width-2, self.card_height-2,
                                    radius=5, fill=COLORS['card_bg'], outline='')
            
            suit = self.card_text[-1].lower() if len(self.card_text) > 0 else ''
            rank = self.card_text[:-1] if len(self.card_text) > 1 else self.card_text
            
            suit_symbols = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}
            color = COLORS['card_red'] if suit in ['h', 'd'] else COLORS['card_black']
            suit_symbol = suit_symbols.get(suit, '')
            
            self.create_text(self.card_width//2, self.card_height//2 - 8, text=rank,
                           fill=color, font=(FONT_FAMILY, 14, 'bold'))
            self.create_text(self.card_width//2, self.card_height//2 + 12, text=suit_symbol,
                           fill=color, font=(FONT_FAMILY, 16))
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def set_card(self, card_text):
        self.card_text = card_text
        self.draw_card()


class EquityGauge(tk.Canvas):
    """Visual equity gauge/meter"""
    def __init__(self, parent, width=200, height=24, **kwargs):
        bg = kwargs.pop('bg', COLORS['bg_card'])
        super().__init__(parent, width=width, height=height,
                        bg=bg, highlightthickness=0, **kwargs)
        self.gauge_width = width
        self.gauge_height = height
        self.value = 0
        self.draw_gauge()
        
    def draw_gauge(self):
        self.delete('all')
        
        # Background
        self.create_rounded_rect(0, 4, self.gauge_width, self.gauge_height-4,
                                radius=6, fill=COLORS['bg_elevated'], outline='')
        
        # Color based on value
        if self.value >= 0.6:
            color = COLORS['status_active']
        elif self.value >= 0.4:
            color = COLORS['status_warning']
        else:
            color = COLORS['status_error']
        
        # Fill bar
        fill_width = max(6, int((self.gauge_width - 4) * self.value))
        if self.value > 0:
            self.create_rounded_rect(2, 6, fill_width, self.gauge_height-6,
                                    radius=4, fill=color, outline='')
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def set_value(self, value):
        self.value = min(1.0, max(0.0, value))
        self.draw_gauge()


class PokerBotGUI:
    """Main Poker Bot GUI Window"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Poker Bot Pro")
        self.window.geometry("1200x900")
        self.window.configure(bg=COLORS['bg_dark'])
        
        # Center window
        self.window.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 600
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 450
        self.window.geometry(f'+{max(0, x)}+{max(0, y)}')
        
        # Initialize components
        self.screen_analyzer = ScreenAnalyzer()
        self.decision_engine = DecisionEngine(PlayStyle.BALANCED)
        self.action_executor = ActionExecutor()
        self.game_state = GameState()
        
        # Bot state
        self.bot_active = False
        self.bot_thread = None
        self.execution_mode = ExecutionMode.SEMI_AUTO
        self.play_style = PlayStyle.BALANCED
        
        # Card displays
        self.hand_cards = []
        self.board_cards = []
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.window.bind('<Escape>', lambda e: self.emergency_stop())
        self.window.bind('<space>', lambda e: self.confirm_action())
        self.window.focus_set()
        
    def setup_ui(self):
        """Setup the user interface"""
        main_container = tk.Frame(self.window, bg=COLORS['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=25, pady=20)
        
        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        header = tk.Frame(main_container, bg=COLORS['bg_dark'])
        header.pack(fill='x', pady=(0, 15))
        
        header_left = tk.Frame(header, bg=COLORS['bg_dark'])
        header_left.pack(side='left')
        
        tk.Label(header_left, text="🤖  Poker Bot Pro",
                font=(FONT_FAMILY, 24, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['text_primary']).pack(anchor='w')
        
        tk.Label(header_left, text="Automated poker assistant with configurable play styles",
                font=(FONT_FAMILY, 12),
                bg=COLORS['bg_dark'], fg=COLORS['text_secondary']).pack(anchor='w')
        
        # Status indicators
        status_frame = tk.Frame(header, bg=COLORS['bg_dark'])
        status_frame.pack(side='right')
        
        self.bot_status_dot = tk.Label(status_frame, text="●", font=('', 12),
                                       bg=COLORS['bg_dark'], fg=COLORS['status_inactive'])
        self.bot_status_dot.pack(side='left')
        
        self.bot_status_label = tk.Label(status_frame, text="Bot Inactive",
                                        font=(FONT_FAMILY, 12, 'bold'),
                                        bg=COLORS['bg_dark'], fg=COLORS['text_secondary'])
        self.bot_status_label.pack(side='left', padx=(8, 0))
        
        # ═══════════════════════════════════════════════════════════════════
        # MAIN CONTENT - Two columns
        # ═══════════════════════════════════════════════════════════════════
        content = tk.Frame(main_container, bg=COLORS['bg_dark'])
        content.pack(fill='both', expand=True)
        
        # Left column - Configuration
        left_col = tk.Frame(content, bg=COLORS['bg_dark'], width=450)
        left_col.pack(side='left', fill='y', padx=(0, 15))
        left_col.pack_propagate(False)
        
        # Right column - Game state and log
        right_col = tk.Frame(content, bg=COLORS['bg_dark'])
        right_col.pack(side='left', fill='both', expand=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # LEFT COLUMN SECTIONS
        # ═══════════════════════════════════════════════════════════════════
        
        # --- Region Setup Section ---
        self._create_region_setup(left_col)
        
        # --- Play Style Section ---
        self._create_play_style_section(left_col)
        
        # --- Execution Mode Section ---
        self._create_execution_mode_section(left_col)
        
        # --- Bet Sizing Section ---
        self._create_bet_sizing_section(left_col)
        
        # --- Control Buttons ---
        self._create_control_buttons(left_col)
        
        # ═══════════════════════════════════════════════════════════════════
        # RIGHT COLUMN SECTIONS
        # ═══════════════════════════════════════════════════════════════════
        
        # --- Game State Display ---
        self._create_game_state_display(right_col)
        
        # --- Action Recommendation ---
        self._create_action_display(right_col)
        
        # --- Action Log ---
        self._create_action_log(right_col)
        
        # --- Status Bar ---
        self._create_status_bar(main_container)
    
    def _create_region_setup(self, parent):
        """Create region setup section"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x', pady=(0, 12))
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=15, pady=12)
        
        tk.Label(inner, text="REGION SETUP",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        tk.Label(inner, text="Select screen regions for detection",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w', pady=(2, 10))
        
        # Region buttons grid
        btn_frame = tk.Frame(inner, bg=COLORS['bg_card'])
        btn_frame.pack(fill='x')
        
        # Row 1 - Cards
        row1 = tk.Frame(btn_frame, bg=COLORS['bg_card'])
        row1.pack(fill='x', pady=(0, 8))
        
        self.hand_region_btn = ModernButton(row1, "Hand Cards", 
                                           lambda: self.select_region('hand'),
                                           'green', width=130, height=36, bg=COLORS['bg_card'])
        self.hand_region_btn.pack(side='left', padx=(0, 8))
        
        self.board_region_btn = ModernButton(row1, "Board Cards",
                                            lambda: self.select_region('board'),
                                            'red', width=130, height=36, bg=COLORS['bg_card'])
        self.board_region_btn.pack(side='left', padx=(0, 8))
        
        self.pot_region_btn = ModernButton(row1, "Pot Size",
                                          lambda: self.select_region('pot'),
                                          'gold', width=130, height=36, bg=COLORS['bg_card'])
        self.pot_region_btn.pack(side='left')
        
        # Row 2 - Buttons
        row2 = tk.Frame(btn_frame, bg=COLORS['bg_card'])
        row2.pack(fill='x')
        
        self.fold_btn_region = ModernButton(row2, "Fold Btn",
                                           lambda: self.select_region('fold'),
                                           'purple', width=100, height=36, bg=COLORS['bg_card'])
        self.fold_btn_region.pack(side='left', padx=(0, 6))
        
        self.call_btn_region = ModernButton(row2, "Call Btn",
                                           lambda: self.select_region('call'),
                                           'purple', width=100, height=36, bg=COLORS['bg_card'])
        self.call_btn_region.pack(side='left', padx=(0, 6))
        
        self.raise_btn_region = ModernButton(row2, "Raise Btn",
                                            lambda: self.select_region('raise'),
                                            'purple', width=100, height=36, bg=COLORS['bg_card'])
        self.raise_btn_region.pack(side='left', padx=(0, 6))
        
        self.bet_input_region = ModernButton(row2, "Bet Input",
                                            lambda: self.select_region('bet_input'),
                                            'cyan', width=100, height=36, bg=COLORS['bg_card'])
        self.bet_input_region.pack(side='left')
    
    def _create_play_style_section(self, parent):
        """Create play style selection section"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x', pady=(0, 12))
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=15, pady=12)
        
        tk.Label(inner, text="PLAY STYLE",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        # Style options
        self.style_var = tk.StringVar(value="BALANCED")
        
        styles = [
            ("STRICT", "Pure equity-based, no bluffs", COLORS['accent_green']),
            ("BLUFF", "Aggressive, exploitative play", COLORS['accent_red']),
            ("BALANCED", "GTO-approximation, mixed strategy", COLORS['accent_blue']),
        ]
        
        for style_name, desc, color in styles:
            frame = tk.Frame(inner, bg=COLORS['bg_card'])
            frame.pack(fill='x', pady=(8, 0))
            
            rb = tk.Radiobutton(frame, text=style_name, variable=self.style_var,
                               value=style_name, font=(FONT_FAMILY, 11, 'bold'),
                               bg=COLORS['bg_card'], fg=color,
                               selectcolor=COLORS['bg_elevated'],
                               activebackground=COLORS['bg_card'],
                               activeforeground=color,
                               command=self.on_style_change)
            rb.pack(anchor='w')
            
            tk.Label(frame, text=desc, font=(FONT_FAMILY, 9),
                    bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w', padx=(22, 0))
    
    def _create_execution_mode_section(self, parent):
        """Create execution mode section"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x', pady=(0, 12))
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=15, pady=12)
        
        tk.Label(inner, text="EXECUTION MODE",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        self.exec_mode_var = tk.StringVar(value="SEMI_AUTO")
        
        modes = [
            ("SEMI_AUTO", "Suggest action, confirm with SPACE", True),
            ("FULL_AUTO", "Fully automated execution", False),
        ]
        
        for mode_name, desc, safe in modes:
            frame = tk.Frame(inner, bg=COLORS['bg_card'])
            frame.pack(fill='x', pady=(8, 0))
            
            color = COLORS['accent_green'] if safe else COLORS['accent_red']
            rb = tk.Radiobutton(frame, text=mode_name.replace('_', '-'),
                               variable=self.exec_mode_var, value=mode_name,
                               font=(FONT_FAMILY, 11, 'bold'),
                               bg=COLORS['bg_card'], fg=color,
                               selectcolor=COLORS['bg_elevated'],
                               activebackground=COLORS['bg_card'],
                               activeforeground=color,
                               command=self.on_exec_mode_change)
            rb.pack(anchor='w')
            
            tk.Label(frame, text=desc, font=(FONT_FAMILY, 9),
                    bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w', padx=(22, 0))
    
    def _create_bet_sizing_section(self, parent):
        """Create bet sizing configuration section"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x', pady=(0, 12))
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=15, pady=12)
        
        tk.Label(inner, text="BET SIZING",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        # Preflop open raise
        pf_frame = tk.Frame(inner, bg=COLORS['bg_card'])
        pf_frame.pack(fill='x', pady=(10, 5))
        
        tk.Label(pf_frame, text="Preflop Open (BB):",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_secondary']).pack(side='left')
        
        self.open_raise_var = tk.StringVar(value="2.5")
        open_entry = tk.Entry(pf_frame, textvariable=self.open_raise_var,
                             width=6, font=(FONT_MONO, 10),
                             bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                             insertbackground=COLORS['text_primary'],
                             relief='flat', highlightthickness=1,
                             highlightbackground=COLORS['border'])
        open_entry.pack(side='right')
        
        # Postflop sizing
        post_frame = tk.Frame(inner, bg=COLORS['bg_card'])
        post_frame.pack(fill='x', pady=(5, 0))
        
        tk.Label(post_frame, text="Postflop (% pot):",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_secondary']).pack(side='left')
        
        self.cbet_var = tk.StringVar(value="50")
        cbet_entry = tk.Entry(post_frame, textvariable=self.cbet_var,
                             width=6, font=(FONT_MONO, 10),
                             bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                             insertbackground=COLORS['text_primary'],
                             relief='flat', highlightthickness=1,
                             highlightbackground=COLORS['border'])
        cbet_entry.pack(side='right')
    
    def _create_control_buttons(self, parent):
        """Create main control buttons"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x')
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=15, pady=15)
        
        btn_row = tk.Frame(inner, bg=COLORS['bg_card'])
        btn_row.pack()
        
        self.start_btn = ModernButton(btn_row, "Start Bot", self.toggle_bot,
                                     'green', width=180, height=48, icon="▶", bg=COLORS['bg_card'])
        self.start_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = ModernButton(btn_row, "Emergency Stop", self.emergency_stop,
                                    'red', width=180, height=48, icon="■", bg=COLORS['bg_card'])
        self.stop_btn.pack(side='left')
    
    def _create_game_state_display(self, parent):
        """Create game state display section"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x', pady=(0, 12))
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=20, pady=15)
        
        # Header
        header = tk.Frame(inner, bg=COLORS['bg_card'])
        header.pack(fill='x')
        
        tk.Label(header, text="CURRENT GAME STATE",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(side='left')
        
        self.phase_label = tk.Label(header, text="WAITING",
                                   font=(FONT_MONO, 10, 'bold'),
                                   bg=COLORS['bg_elevated'], fg=COLORS['accent_gold'])
        self.phase_label.pack(side='right', ipadx=8, ipady=2)
        
        # Cards row
        cards_row = tk.Frame(inner, bg=COLORS['bg_card'])
        cards_row.pack(fill='x', pady=(15, 0))
        
        # Hand cards
        hand_frame = tk.Frame(cards_row, bg=COLORS['bg_card'])
        hand_frame.pack(side='left', padx=(0, 30))
        
        tk.Label(hand_frame, text="YOUR HAND",
                font=(FONT_FAMILY, 9, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['accent_green']).pack(anchor='w')
        
        hand_cards_row = tk.Frame(hand_frame, bg=COLORS['bg_card'])
        hand_cards_row.pack(anchor='w', pady=(5, 0))
        
        for i in range(2):
            card = CardDisplay(hand_cards_row, bg=COLORS['bg_card'])
            card.pack(side='left', padx=(0, 4))
            self.hand_cards.append(card)
        
        # Board cards
        board_frame = tk.Frame(cards_row, bg=COLORS['bg_card'])
        board_frame.pack(side='left')
        
        tk.Label(board_frame, text="COMMUNITY CARDS",
                font=(FONT_FAMILY, 9, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['accent_red']).pack(anchor='w')
        
        board_cards_row = tk.Frame(board_frame, bg=COLORS['bg_card'])
        board_cards_row.pack(anchor='w', pady=(5, 0))
        
        for i in range(5):
            card = CardDisplay(board_cards_row, bg=COLORS['bg_card'])
            card.pack(side='left', padx=(0, 4))
            self.board_cards.append(card)
        
        # Stats row
        stats_row = tk.Frame(inner, bg=COLORS['bg_card'])
        stats_row.pack(fill='x', pady=(15, 0))
        
        # Equity
        equity_frame = tk.Frame(stats_row, bg=COLORS['bg_card'])
        equity_frame.pack(side='left', padx=(0, 40))
        
        tk.Label(equity_frame, text="EQUITY",
                font=(FONT_FAMILY, 9),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.equity_label = tk.Label(equity_frame, text="--.--%",
                                    font=(FONT_FAMILY, 24, 'bold'),
                                    bg=COLORS['bg_card'], fg=COLORS['text_primary'])
        self.equity_label.pack(anchor='w')
        
        self.equity_gauge = EquityGauge(equity_frame, width=150, height=20, bg=COLORS['bg_card'])
        self.equity_gauge.pack(anchor='w', pady=(5, 0))
        
        # Pot
        pot_frame = tk.Frame(stats_row, bg=COLORS['bg_card'])
        pot_frame.pack(side='left', padx=(0, 40))
        
        tk.Label(pot_frame, text="POT SIZE",
                font=(FONT_FAMILY, 9),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.pot_label = tk.Label(pot_frame, text="$0.00",
                                 font=(FONT_FAMILY, 18, 'bold'),
                                 bg=COLORS['bg_card'], fg=COLORS['accent_gold'])
        self.pot_label.pack(anchor='w')
        
        # To Call
        call_frame = tk.Frame(stats_row, bg=COLORS['bg_card'])
        call_frame.pack(side='left')
        
        tk.Label(call_frame, text="TO CALL",
                font=(FONT_FAMILY, 9),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.to_call_label = tk.Label(call_frame, text="$0.00",
                                     font=(FONT_FAMILY, 18, 'bold'),
                                     bg=COLORS['bg_card'], fg=COLORS['accent_red'])
        self.to_call_label.pack(anchor='w')
    
    def _create_action_display(self, parent):
        """Create action recommendation display"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='x', pady=(0, 12))
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=20, pady=15)
        
        tk.Label(inner, text="RECOMMENDED ACTION",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        action_row = tk.Frame(inner, bg=COLORS['bg_card'])
        action_row.pack(fill='x', pady=(10, 0))
        
        self.action_label = tk.Label(action_row, text="WAITING",
                                    font=(FONT_FAMILY, 28, 'bold'),
                                    bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        self.action_label.pack(side='left')
        
        self.action_amount_label = tk.Label(action_row, text="",
                                           font=(FONT_FAMILY, 20, 'bold'),
                                           bg=COLORS['bg_card'], fg=COLORS['accent_gold'])
        self.action_amount_label.pack(side='left', padx=(15, 0))
        
        self.action_reason_label = tk.Label(inner, text="Select regions and start bot to begin",
                                           font=(FONT_FAMILY, 10),
                                           bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        self.action_reason_label.pack(anchor='w', pady=(8, 0))
        
        # Confirm hint
        self.confirm_hint = tk.Label(inner, text="",
                                    font=(FONT_FAMILY, 10, 'bold'),
                                    bg=COLORS['bg_card'], fg=COLORS['accent_cyan'])
        self.confirm_hint.pack(anchor='w', pady=(5, 0))
    
    def _create_action_log(self, parent):
        """Create action log section"""
        section = tk.Frame(parent, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        section.pack(fill='both', expand=True)
        
        inner = tk.Frame(section, bg=COLORS['bg_card'])
        inner.pack(fill='both', expand=True, padx=20, pady=15)
        
        header = tk.Frame(inner, bg=COLORS['bg_card'])
        header.pack(fill='x')
        
        tk.Label(header, text="ACTION LOG",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(side='left')
        
        clear_btn = tk.Label(header, text="Clear",
                            font=(FONT_FAMILY, 10),
                            bg=COLORS['bg_card'], fg=COLORS['accent_blue'],
                            cursor='hand2')
        clear_btn.pack(side='right')
        clear_btn.bind('<Button-1>', lambda e: self.clear_log())
        
        # Log text
        log_frame = tk.Frame(inner, bg=COLORS['bg_elevated'])
        log_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=10,
                               font=(FONT_MONO, 10),
                               bg=COLORS['bg_elevated'], fg=COLORS['text_secondary'],
                               relief='flat', wrap='word', highlightthickness=0,
                               padx=10, pady=10)
        self.log_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.log("Bot initialized. Configure regions and start to begin.")
    
    def _create_status_bar(self, parent):
        """Create status bar"""
        status_bar = tk.Frame(parent, bg=COLORS['bg_card'],
                             highlightbackground=COLORS['border'], highlightthickness=1)
        status_bar.pack(fill='x', pady=(15, 0))
        
        inner = tk.Frame(status_bar, bg=COLORS['bg_card'])
        inner.pack(fill='x', padx=15, pady=8)
        
        self.status_dot = tk.Label(inner, text="●", font=('', 8),
                                  bg=COLORS['bg_card'], fg=COLORS['status_inactive'])
        self.status_dot.pack(side='left')
        
        self.status_label = tk.Label(inner, text="Ready — Configure regions to begin",
                                    font=(FONT_FAMILY, 10),
                                    bg=COLORS['bg_card'], fg=COLORS['text_secondary'])
        self.status_label.pack(side='left', padx=(8, 0))
        
        tk.Label(inner, text="ESC=Stop  SPACE=Confirm",
                font=(FONT_FAMILY, 9),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(side='right')
    
    # ═══════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════
    
    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete('1.0', tk.END)
        self.log("Log cleared.")
    
    def update_status(self, message, color='inactive'):
        """Update status bar"""
        color_map = {
            'active': COLORS['status_active'],
            'warning': COLORS['status_warning'],
            'inactive': COLORS['status_inactive'],
            'error': COLORS['status_error'],
        }
        self.status_dot.config(fg=color_map.get(color, COLORS['status_inactive']))
        self.status_label.config(text=message)
    
    def select_region(self, region_name):
        """Open region selection for a specific region"""
        self.log(f"Selecting {region_name} region...")
        self.update_status(f"Click and drag to select {region_name.upper()}", 'warning')
        
        def do_selection():
            region = self.screen_analyzer.select_region_interactive(region_name.upper())
            if region:
                self.screen_analyzer.set_region(region_name, region)
                
                # Also set in action executor if it's a button
                if region_name in ['fold', 'call', 'raise', 'bet_input']:
                    self.action_executor.set_button_region(region_name, region)
                
                self.window.after(0, lambda: self.log(f"{region_name.title()} region set: {region}"))
                self.window.after(0, lambda: self.update_status(f"{region_name.title()} region configured", 'active'))
            else:
                self.window.after(0, lambda: self.log(f"{region_name.title()} selection cancelled"))
                self.window.after(0, lambda: self.update_status("Selection cancelled", 'inactive'))
        
        threading.Thread(target=do_selection, daemon=True).start()
    
    def on_style_change(self):
        """Handle play style change"""
        style_name = self.style_var.get()
        self.play_style = PlayStyle[style_name]
        self.decision_engine.set_play_style(self.play_style)
        self.log(f"Play style changed to {style_name}")
    
    def on_exec_mode_change(self):
        """Handle execution mode change"""
        mode_name = self.exec_mode_var.get()
        self.execution_mode = ExecutionMode[mode_name]
        self.action_executor.set_mode(self.execution_mode)
        self.log(f"Execution mode changed to {mode_name.replace('_', '-')}")
        
        if self.execution_mode == ExecutionMode.FULL_AUTO:
            messagebox.showwarning(
                "Full Auto Mode",
                "Full auto mode will automatically click buttons!\n\n"
                "Press ESC at any time to emergency stop.\n\n"
                "Make sure all regions are correctly configured."
            )
    
    def toggle_bot(self):
        """Toggle bot on/off"""
        if self.bot_active:
            self.stop_bot()
        else:
            self.start_bot()
    
    def start_bot(self):
        """Start the bot"""
        # Validate regions
        if not self.screen_analyzer.config.hand_region:
            messagebox.showwarning("Setup Required", "Please select the hand cards region first!")
            return
        
        if not self.screen_analyzer.config.board_region:
            messagebox.showwarning("Setup Required", "Please select the board cards region first!")
            return
        
        self.bot_active = True
        self.action_executor.start()
        
        # Update UI
        self.start_btn.set_active(True, "Bot Running")
        self.bot_status_dot.config(fg=COLORS['status_active'])
        self.bot_status_label.config(text="Bot Active", fg=COLORS['status_active'])
        self.update_status("Bot is running...", 'active')
        self.log("Bot started!")
        
        # Start bot thread
        self.bot_thread = threading.Thread(target=self.bot_loop, daemon=True)
        self.bot_thread.start()
    
    def stop_bot(self):
        """Stop the bot normally"""
        self.bot_active = False
        self.action_executor.stop()
        
        # Update UI
        self.start_btn.set_active(False, "Start Bot")
        self.bot_status_dot.config(fg=COLORS['status_inactive'])
        self.bot_status_label.config(text="Bot Inactive", fg=COLORS['text_secondary'])
        self.update_status("Bot stopped", 'inactive')
        self.log("Bot stopped.")
    
    def emergency_stop(self):
        """Emergency stop the bot"""
        self.bot_active = False
        self.action_executor.emergency_halt()
        
        # Update UI
        self.start_btn.set_active(False, "Start Bot")
        self.bot_status_dot.config(fg=COLORS['status_error'])
        self.bot_status_label.config(text="EMERGENCY STOP", fg=COLORS['status_error'])
        self.update_status("Emergency stop activated!", 'error')
        self.log("EMERGENCY STOP!")
        
        # Reset action display
        self.action_label.config(text="STOPPED", fg=COLORS['status_error'])
        self.action_amount_label.config(text="")
        self.confirm_hint.config(text="")
    
    def confirm_action(self):
        """Confirm pending action in semi-auto mode"""
        if self.execution_mode == ExecutionMode.SEMI_AUTO:
            if self.action_executor.pending_action:
                self.log("Confirming action...")
                self.action_executor.confirm_pending_action()
    
    def bot_loop(self):
        """Main bot loop"""
        while self.bot_active:
            try:
                # Analyze screen
                analysis = self.screen_analyzer.analyze_table()
                
                # Update game state
                self.game_state.update_cards(
                    analysis.get('hand_cards', []),
                    analysis.get('board_cards', [])
                )
                self.game_state.pot_size = analysis.get('pot_size', 0)
                self.game_state.update_available_actions()
                
                # Update UI
                self.window.after(0, self.update_game_display)
                
                # Make decision if we have cards
                if len(self.game_state.hole_cards) == 2:
                    decision = self.decision_engine.make_decision(self.game_state)
                    self.window.after(0, lambda d=decision: self.display_decision(d))
                    
                    # Execute in auto mode
                    if self.execution_mode == ExecutionMode.FULL_AUTO:
                        self.execute_decision(decision)
                    else:
                        # Set up pending action for semi-auto
                        self.action_executor.suggest_action(
                            decision.action.value,
                            decision.amount
                        )
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self.window.after(0, lambda: self.log(f"Error: {e}"))
                time.sleep(1)
    
    def update_game_display(self):
        """Update the game state display"""
        # Update cards
        for i, widget in enumerate(self.hand_cards):
            if i < len(self.game_state.hole_cards):
                widget.set_card(self.game_state.hole_cards[i])
            else:
                widget.set_card("?")
        
        for i, widget in enumerate(self.board_cards):
            if i < len(self.game_state.community_cards):
                widget.set_card(self.game_state.community_cards[i])
            else:
                widget.set_card("?")
        
        # Update phase
        self.phase_label.config(text=self.game_state.phase.name)
        
        # Update stats
        self.pot_label.config(text=f"${self.game_state.pot_size:.2f}")
        self.to_call_label.config(text=f"${self.game_state.to_call:.2f}")
    
    def display_decision(self, decision):
        """Display the recommended decision"""
        action_colors = {
            Action.FOLD: COLORS['status_error'],
            Action.CHECK: COLORS['text_secondary'],
            Action.CALL: COLORS['status_warning'],
            Action.BET: COLORS['status_active'],
            Action.RAISE: COLORS['status_active'],
            Action.ALL_IN: COLORS['accent_purple'],
        }
        
        color = action_colors.get(decision.action, COLORS['text_primary'])
        self.action_label.config(text=decision.action.value.upper(), fg=color)
        
        if decision.amount > 0:
            self.action_amount_label.config(text=f"${decision.amount:.2f}")
        else:
            self.action_amount_label.config(text="")
        
        self.action_reason_label.config(text=decision.reasoning)
        
        # Update equity display
        self.equity_label.config(text=f"{decision.equity*100:.1f}%")
        self.equity_gauge.set_value(decision.equity)
        
        if decision.equity >= 0.6:
            self.equity_label.config(fg=COLORS['status_active'])
        elif decision.equity >= 0.4:
            self.equity_label.config(fg=COLORS['status_warning'])
        else:
            self.equity_label.config(fg=COLORS['status_error'])
        
        # Show confirm hint in semi-auto mode
        if self.execution_mode == ExecutionMode.SEMI_AUTO:
            self.confirm_hint.config(text="Press SPACE to confirm action")
        else:
            self.confirm_hint.config(text="")
    
    def execute_decision(self, decision):
        """Execute a decision (full auto mode)"""
        if decision.action == Action.FOLD:
            self.action_executor.execute_fold()
        elif decision.action == Action.CHECK:
            self.action_executor.execute_check()
        elif decision.action == Action.CALL:
            self.action_executor.execute_call()
        elif decision.action in (Action.BET, Action.RAISE):
            self.action_executor.execute_raise(decision.amount)
        elif decision.action == Action.ALL_IN:
            self.action_executor.execute_all_in()
        
        self.log(f"Executed: {decision.action.value.upper()} {f'${decision.amount:.2f}' if decision.amount else ''}")


def main():
    """Standalone test"""
    root = tk.Tk()
    root.withdraw()  # Hide root window
    
    app = PokerBotGUI(root)
    app.window.protocol("WM_DELETE_WINDOW", root.quit)
    
    root.mainloop()


if __name__ == "__main__":
    main()

