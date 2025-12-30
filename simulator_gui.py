#!/usr/bin/env python3
"""
Simulator GUI - Manual Card Input
Professional interface for manual card entry and win probability calculation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import threading
import time
import sys
import os
import platform

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'simulator'))
from poker_cli_session import tokenize_cards, to_treys, simulate, kelly_even_money

# ═══════════════════════════════════════════════════════════════════════════════
# FONT CONFIGURATION - Cross-platform compatible fonts
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
# COLOR PALETTE - Sophisticated dark poker theme
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
    
    'card_red': '#ef4444',
    'card_black': '#1a1a1a',
    'card_bg': '#ffffff',
    
    'equity_high': '#22c55e',
    'equity_medium': '#eab308',
    'equity_low': '#ef4444',
}


class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, color='blue', width=160, height=44, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        self.enabled = True
        
        color_map = {
            'blue': (COLORS['accent_blue'], COLORS['accent_blue_hover']),
            'green': (COLORS['accent_green'], COLORS['accent_green_hover']),
            'red': (COLORS['accent_red'], COLORS['accent_red_hover']),
            'gold': (COLORS['accent_gold'], '#e3b341'),
        }
        self.normal_color, self.hover_color = color_map.get(color, color_map['blue'])
        self.current_color = self.normal_color
        
        self.draw_button()
        
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        
    def draw_button(self):
        self.delete('all')
        color = self.current_color if self.enabled else COLORS['bg_elevated']
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=10, fill=color, outline='')
        text_color = COLORS['text_primary'] if self.enabled else COLORS['text_muted']
        self.create_text(self.width//2, self.height//2, text=self.text,
                        fill=text_color, font=(FONT_FAMILY, 12, 'bold'))
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, event):
        if self.enabled:
            self.current_color = self.hover_color
            self.draw_button()
            self.config(cursor='hand2')
        
    def on_leave(self, event):
        self.current_color = self.normal_color
        self.draw_button()
        
    def on_click(self, event):
        if self.enabled and self.command:
            self.command()
            
    def set_enabled(self, enabled):
        self.enabled = enabled
        self.draw_button()


class CardVisual(tk.Canvas):
    """Visual poker card widget"""
    def __init__(self, parent, card_text="?", width=55, height=78, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_card'], highlightthickness=0, **kwargs)
        self.card_text = card_text
        self.width = width
        self.height = height
        self.draw_card()
        
    def draw_card(self):
        self.delete('all')
        
        if not self.card_text or self.card_text == "?":
            # Empty card placeholder
            self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                    radius=6, fill=COLORS['bg_elevated'], 
                                    outline=COLORS['border'])
            self.create_text(self.width//2, self.height//2, text="?",
                           fill=COLORS['text_muted'], font=(FONT_FAMILY, 18, 'bold'))
        else:
            # Card with value
            self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                    radius=6, fill=COLORS['card_bg'], outline='')
            
            # Parse card
            suit = self.card_text[-1].lower() if len(self.card_text) > 0 else ''
            rank = self.card_text[:-1] if len(self.card_text) > 1 else self.card_text
            
            # Suit symbols and colors
            suit_symbols = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}
            color = COLORS['card_red'] if suit in ['h', 'd'] else COLORS['card_black']
            suit_symbol = suit_symbols.get(suit, '')
            
            # Draw rank
            self.create_text(self.width//2, self.height//2 - 8, text=rank,
                           fill=color, font=(FONT_FAMILY, 16, 'bold'))
            # Draw suit
            self.create_text(self.width//2, self.height//2 + 14, text=suit_symbol,
                           fill=color, font=(FONT_FAMILY, 18))
    
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
    def __init__(self, parent, width=300, height=30, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_card'], highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.value = 0
        self.draw_gauge()
        
    def draw_gauge(self):
        self.delete('all')
        
        # Background bar
        self.create_rounded_rect(0, 5, self.width, self.height-5,
                                radius=8, fill=COLORS['bg_elevated'], outline='')
        
        # Determine color based on value
        if self.value >= 0.6:
            color = COLORS['equity_high']
        elif self.value >= 0.4:
            color = COLORS['equity_medium']
        else:
            color = COLORS['equity_low']
        
        # Fill bar
        fill_width = max(8, int((self.width - 4) * self.value))
        if self.value > 0:
            self.create_rounded_rect(2, 7, fill_width, self.height-7,
                                    radius=6, fill=color, outline='')
    
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


class SimulatorGUI:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Poker Simulator Pro")
        self.window.geometry("950x750")
        self.window.configure(bg=COLORS['bg_dark'])
        
        # Center window
        self.window.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 475
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 375
        self.window.geometry(f'+{x}+{y}')
        
        # Simulation state
        self.simulation_running = False
        self.current_equity = None
        self.current_kelly = None
        
        # Card display widgets
        self.hand_cards = []
        self.board_cards = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        main_container = tk.Frame(self.window, bg=COLORS['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=30, pady=20)
        
        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        header_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
        header_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(header_frame, text="🎲  Poker Simulator",
                font=(FONT_FAMILY, 24, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['text_primary']).pack(anchor='w')
        
        tk.Label(header_frame, text="Calculate win probability with Monte Carlo simulation",
                font=(FONT_FAMILY, 12),
                bg=COLORS['bg_dark'], fg=COLORS['text_secondary']).pack(anchor='w')
        
        # ═══════════════════════════════════════════════════════════════════
        # CARDS INPUT SECTION
        # ═══════════════════════════════════════════════════════════════════
        cards_section = tk.Frame(main_container, bg=COLORS['bg_card'],
                                highlightbackground=COLORS['border'], highlightthickness=1)
        cards_section.pack(fill='x', pady=(0, 15))
        
        cards_inner = tk.Frame(cards_section, bg=COLORS['bg_card'])
        cards_inner.pack(fill='x', padx=25, pady=20)
        
        # HAND INPUT
        hand_frame = tk.Frame(cards_inner, bg=COLORS['bg_card'])
        hand_frame.pack(fill='x', pady=(0, 20))
        
        hand_label_frame = tk.Frame(hand_frame, bg=COLORS['bg_card'])
        hand_label_frame.pack(fill='x')
        
        tk.Label(hand_label_frame, text="YOUR HAND",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['accent_green']).pack(side='left')
        
        tk.Label(hand_label_frame, text="(2 cards)",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(side='left', padx=(8, 0))
        
        hand_input_frame = tk.Frame(hand_frame, bg=COLORS['bg_card'])
        hand_input_frame.pack(fill='x', pady=(10, 0))
        
        # Hand entry
        self.hand_entry = tk.Entry(hand_input_frame, font=(FONT_MONO, 14),
                                  bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                                  insertbackground=COLORS['text_primary'],
                                  relief='flat', width=20,
                                  highlightbackground=COLORS['border'],
                                  highlightcolor=COLORS['border_focus'],
                                  highlightthickness=1)
        self.hand_entry.pack(side='left', ipady=8, ipadx=10)
        self.hand_entry.insert(0, "As Kh")
        self.hand_entry.bind('<KeyRelease>', self.update_card_preview)
        
        # Hand card visuals
        self.hand_cards_frame = tk.Frame(hand_input_frame, bg=COLORS['bg_card'])
        self.hand_cards_frame.pack(side='left', padx=(20, 0))
        
        for i in range(2):
            card = CardVisual(self.hand_cards_frame)
            card.pack(side='left', padx=3)
            self.hand_cards.append(card)
        
        # BOARD INPUT
        board_frame = tk.Frame(cards_inner, bg=COLORS['bg_card'])
        board_frame.pack(fill='x')
        
        board_label_frame = tk.Frame(board_frame, bg=COLORS['bg_card'])
        board_label_frame.pack(fill='x')
        
        tk.Label(board_label_frame, text="COMMUNITY CARDS",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['accent_red']).pack(side='left')
        
        tk.Label(board_label_frame, text="(0-5 cards)",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(side='left', padx=(8, 0))
        
        board_input_frame = tk.Frame(board_frame, bg=COLORS['bg_card'])
        board_input_frame.pack(fill='x', pady=(10, 0))
        
        # Board entry
        self.board_entry = tk.Entry(board_input_frame, font=(FONT_MONO, 14),
                                   bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                                   insertbackground=COLORS['text_primary'],
                                   relief='flat', width=30,
                                   highlightbackground=COLORS['border'],
                                   highlightcolor=COLORS['border_focus'],
                                   highlightthickness=1)
        self.board_entry.pack(side='left', ipady=8, ipadx=10)
        self.board_entry.insert(0, "7h 2d 2s")
        self.board_entry.bind('<KeyRelease>', self.update_card_preview)
        
        # Board card visuals
        self.board_cards_frame = tk.Frame(board_input_frame, bg=COLORS['bg_card'])
        self.board_cards_frame.pack(side='left', padx=(20, 0))
        
        for i in range(5):
            card = CardVisual(self.board_cards_frame)
            card.pack(side='left', padx=3)
            self.board_cards.append(card)
        
        # Initialize card preview
        self.update_card_preview(None)
        
        # ═══════════════════════════════════════════════════════════════════
        # SETTINGS ROW
        # ═══════════════════════════════════════════════════════════════════
        settings_frame = tk.Frame(main_container, bg=COLORS['bg_card'],
                                 highlightbackground=COLORS['border'], highlightthickness=1)
        settings_frame.pack(fill='x', pady=(0, 15))
        
        settings_inner = tk.Frame(settings_frame, bg=COLORS['bg_card'])
        settings_inner.pack(fill='x', padx=25, pady=15)
        
        # Players setting
        players_frame = tk.Frame(settings_inner, bg=COLORS['bg_card'])
        players_frame.pack(side='left', padx=(0, 40))
        
        tk.Label(players_frame, text="PLAYERS",
                font=(FONT_FAMILY, 10, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_secondary']).pack(anchor='w')
        
        self.players_var = tk.StringVar(value="6")
        players_spin = tk.Spinbox(players_frame, from_=2, to=10, textvariable=self.players_var,
                                 font=(FONT_MONO, 12), width=6,
                                 bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                                 buttonbackground=COLORS['bg_elevated'],
                                 highlightbackground=COLORS['border'],
                                 highlightthickness=1, relief='flat')
        players_spin.pack(pady=(5, 0), ipady=5)
        
        # Trials setting
        trials_frame = tk.Frame(settings_inner, bg=COLORS['bg_card'])
        trials_frame.pack(side='left', padx=(0, 40))
        
        tk.Label(trials_frame, text="SIMULATIONS",
                font=(FONT_FAMILY, 10, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_secondary']).pack(anchor='w')
        
        self.trials_var = tk.StringVar(value="10000")
        trials_spin = tk.Spinbox(trials_frame, from_=1000, to=100000, increment=1000,
                                textvariable=self.trials_var,
                                font=(FONT_MONO, 12), width=10,
                                bg=COLORS['bg_input'], fg=COLORS['text_primary'],
                                buttonbackground=COLORS['bg_elevated'],
                                highlightbackground=COLORS['border'],
                                highlightthickness=1, relief='flat')
        trials_spin.pack(pady=(5, 0), ipady=5)
        
        # Control buttons
        controls_frame = tk.Frame(settings_inner, bg=COLORS['bg_card'])
        controls_frame.pack(side='right')
        
        self.calculate_btn = ModernButton(controls_frame, text="Calculate",
                                         command=self.calculate_probability,
                                         color='blue', width=140, height=42)
        self.calculate_btn.pack(side='left', padx=(0, 10))
        
        self.clear_btn = ModernButton(controls_frame, text="Clear",
                                     command=self.clear_inputs,
                                     color='gold', width=100, height=42)
        self.clear_btn.pack(side='left')
        
        # ═══════════════════════════════════════════════════════════════════
        # RESULTS SECTION
        # ═══════════════════════════════════════════════════════════════════
        results_section = tk.Frame(main_container, bg=COLORS['bg_card'],
                                  highlightbackground=COLORS['border'], highlightthickness=1)
        results_section.pack(fill='both', expand=True)
        
        results_inner = tk.Frame(results_section, bg=COLORS['bg_card'])
        results_inner.pack(fill='both', expand=True, padx=25, pady=20)
        
        # Results header
        results_header = tk.Frame(results_inner, bg=COLORS['bg_card'])
        results_header.pack(fill='x', pady=(0, 15))
        
        tk.Label(results_header, text="SIMULATION RESULTS",
                font=(FONT_FAMILY, 12, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(side='left')
        
        # Main results display
        results_content = tk.Frame(results_inner, bg=COLORS['bg_card'])
        results_content.pack(fill='both', expand=True)
        
        # Left side - Key metrics
        metrics_frame = tk.Frame(results_content, bg=COLORS['bg_card'])
        metrics_frame.pack(side='left', fill='y', padx=(0, 30))
        
        # Equity display
        equity_frame = tk.Frame(metrics_frame, bg=COLORS['bg_card'])
        equity_frame.pack(anchor='w', pady=(0, 20))
        
        tk.Label(equity_frame, text="EQUITY",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.equity_label = tk.Label(equity_frame, text="--.--%",
                                    font=(FONT_FAMILY, 36, 'bold'),
                                    bg=COLORS['bg_card'], fg=COLORS['text_primary'])
        self.equity_label.pack(anchor='w')
        
        self.equity_gauge = EquityGauge(equity_frame, width=250, height=24)
        self.equity_gauge.pack(anchor='w', pady=(5, 0))
        
        # Win/Tie display
        stats_frame = tk.Frame(metrics_frame, bg=COLORS['bg_card'])
        stats_frame.pack(anchor='w', pady=(0, 20))
        
        win_frame = tk.Frame(stats_frame, bg=COLORS['bg_card'])
        win_frame.pack(side='left', padx=(0, 30))
        
        tk.Label(win_frame, text="WIN RATE",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.win_label = tk.Label(win_frame, text="--.--%",
                                 font=(FONT_FAMILY, 20, 'bold'),
                                 bg=COLORS['bg_card'], fg=COLORS['equity_high'])
        self.win_label.pack(anchor='w')
        
        tie_frame = tk.Frame(stats_frame, bg=COLORS['bg_card'])
        tie_frame.pack(side='left')
        
        tk.Label(tie_frame, text="TIE RATE",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.tie_label = tk.Label(tie_frame, text="--.--%",
                                 font=(FONT_FAMILY, 20, 'bold'),
                                 bg=COLORS['bg_card'], fg=COLORS['equity_medium'])
        self.tie_label.pack(anchor='w')
        
        # Kelly criterion
        kelly_frame = tk.Frame(metrics_frame, bg=COLORS['bg_card'])
        kelly_frame.pack(anchor='w')
        
        tk.Label(kelly_frame, text="KELLY CRITERION (EVEN MONEY)",
                font=(FONT_FAMILY, 10),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(anchor='w')
        
        self.kelly_label = tk.Label(kelly_frame, text="--.--%",
                                   font=(FONT_FAMILY, 20, 'bold'),
                                   bg=COLORS['bg_card'], fg=COLORS['accent_purple'])
        self.kelly_label.pack(anchor='w')
        
        # Right side - Detailed analysis
        analysis_frame = tk.Frame(results_content, bg=COLORS['bg_elevated'],
                                 highlightbackground=COLORS['border'], highlightthickness=1)
        analysis_frame.pack(side='left', fill='both', expand=True)
        
        analysis_inner = tk.Frame(analysis_frame, bg=COLORS['bg_elevated'])
        analysis_inner.pack(fill='both', expand=True, padx=15, pady=15)
        
        tk.Label(analysis_inner, text="ANALYSIS",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_elevated'], fg=COLORS['text_primary']).pack(anchor='w')
        
        self.analysis_text = tk.Text(analysis_inner, height=12, width=40,
                                    font=(FONT_MONO, 11),
                                    bg=COLORS['bg_elevated'], fg=COLORS['text_secondary'],
                                    relief='flat', wrap='word',
                                    highlightthickness=0)
        self.analysis_text.pack(fill='both', expand=True, pady=(10, 0))
        self.analysis_text.insert('1.0', "Enter cards and click Calculate to run simulation...")
        self.analysis_text.config(state='disabled')
        
        # ═══════════════════════════════════════════════════════════════════
        # STATUS BAR
        # ═══════════════════════════════════════════════════════════════════
        status_frame = tk.Frame(main_container, bg=COLORS['bg_card'],
                               highlightbackground=COLORS['border'], highlightthickness=1)
        status_frame.pack(fill='x', pady=(15, 0))
        
        status_inner = tk.Frame(status_frame, bg=COLORS['bg_card'])
        status_inner.pack(fill='x', padx=15, pady=8)
        
        self.status_dot = tk.Label(status_inner, text="●",
                                  font=('', 8),
                                  bg=COLORS['bg_card'], fg=COLORS['accent_green'])
        self.status_dot.pack(side='left')
        
        self.status_label = tk.Label(status_inner, text="Ready — Enter your cards and click Calculate",
                                    font=(FONT_FAMILY, 10),
                                    bg=COLORS['bg_card'], fg=COLORS['text_secondary'])
        self.status_label.pack(side='left', padx=(8, 0))
        
    def update_card_preview(self, event):
        """Update the card preview displays"""
        # Update hand cards
        try:
            hand_text = self.hand_entry.get().strip()
            hand_cards = tokenize_cards(hand_text) if hand_text else []
        except:
            hand_cards = []
        
        for i, card_widget in enumerate(self.hand_cards):
            if i < len(hand_cards):
                card_widget.set_card(hand_cards[i])
            else:
                card_widget.set_card("?")
        
        # Update board cards
        try:
            board_text = self.board_entry.get().strip()
            board_cards = tokenize_cards(board_text) if board_text else []
        except:
            board_cards = []
        
        for i, card_widget in enumerate(self.board_cards):
            if i < len(board_cards):
                card_widget.set_card(board_cards[i])
            else:
                card_widget.set_card("?")
        
    def calculate_probability(self):
        """Calculate win probability"""
        try:
            # Get inputs
            hand_text = self.hand_entry.get().strip()
            board_text = self.board_entry.get().strip()
            players = int(self.players_var.get())
            trials = int(self.trials_var.get())
            
            if not hand_text:
                messagebox.showwarning("Warning", "Please enter your hand cards!")
                return
            
            # Parse cards
            hand_cards = tokenize_cards(hand_text)
            board_cards = tokenize_cards(board_text) if board_text else []
            
            if len(hand_cards) != 2:
                messagebox.showerror("Error", "Please enter exactly 2 hand cards!")
                return
            
            if len(board_cards) > 5:
                messagebox.showerror("Error", "Board cannot have more than 5 cards!")
                return
            
            # Check for duplicates
            all_cards = hand_cards + board_cards
            if len(set(all_cards)) != len(all_cards):
                messagebox.showerror("Error", "Duplicate cards detected!")
                return
            
            self.update_status("Running simulation...", 'gold')
            self.calculate_btn.set_enabled(False)
            
            # Start calculation in thread
            calc_thread = threading.Thread(target=self.run_simulation, 
                                         args=(hand_cards, board_cards, players, trials), 
                                         daemon=True)
            calc_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            self.update_status("Error in input", 'red')
            self.calculate_btn.set_enabled(True)
    
    def run_simulation(self, hand_cards, board_cards, players, trials):
        """Run the simulation in background"""
        try:
            # Convert to treys format
            my_cards = to_treys(hand_cards)
            board_cards_treys = to_treys(board_cards)
            
            # Run simulation
            win_p, tie_p, equity = simulate(players, my_cards, board_cards_treys, trials)
            kelly = kelly_even_money(equity)
            
            # Calculate confidence interval
            import math
            var = equity * (1 - equity) / max(1, trials)
            moe = 1.96 * math.sqrt(var)
            
            # Update GUI
            self.window.after(0, self.display_results, {
                'hand_cards': hand_cards,
                'board_cards': board_cards,
                'players': players,
                'trials': trials,
                'win_p': win_p,
                'tie_p': tie_p,
                'equity': equity,
                'kelly': kelly,
                'moe': moe
            })
            
        except Exception as e:
            self.window.after(0, self.show_error, f"Simulation error: {e}")
    
    def display_results(self, results):
        """Display simulation results"""
        # Update main metrics
        self.equity_label.config(text=f"{results['equity']*100:.1f}%")
        self.equity_gauge.set_value(results['equity'])
        
        # Color equity based on value
        if results['equity'] >= 0.6:
            self.equity_label.config(fg=COLORS['equity_high'])
        elif results['equity'] >= 0.4:
            self.equity_label.config(fg=COLORS['equity_medium'])
        else:
            self.equity_label.config(fg=COLORS['equity_low'])
        
        self.win_label.config(text=f"{results['win_p']*100:.1f}%")
        self.tie_label.config(text=f"{results['tie_p']*100:.1f}%")
        self.kelly_label.config(text=f"{results['kelly']*100:.1f}%")
        
        # Build analysis text
        analysis = []
        analysis.append(f"Hand: {' '.join(results['hand_cards'])}")
        analysis.append(f"Board: {' '.join(results['board_cards']) if results['board_cards'] else '(preflop)'}")
        analysis.append(f"Players: {results['players']} | Trials: {results['trials']:,}")
        analysis.append("")
        analysis.append(f"95% Confidence: ±{results['moe']*100:.2f}%")
        analysis.append("")
        analysis.append("─" * 35)
        analysis.append("")
        
        # Add interpretation
        if results['equity'] > 0.6:
            analysis.append("🔥 STRONG HAND")
            analysis.append("High equity position. Consider")
            analysis.append("betting aggressively for value.")
        elif results['equity'] > 0.5:
            analysis.append("✓ FAVORABLE POSITION")
            analysis.append("Positive equity. Play for value")
            analysis.append("but remain cautious.")
        elif results['equity'] > 0.4:
            analysis.append("⚠ MARGINAL HAND")
            analysis.append("Near break-even. Consider pot")
            analysis.append("odds and implied odds carefully.")
        else:
            analysis.append("✗ WEAK POSITION")
            analysis.append("Negative equity against range.")
            analysis.append("Consider folding or bluffing.")
        
        analysis.append("")
        analysis.append(f"Kelly suggests: {results['kelly']*100:.1f}% of bankroll")
        
        # Update analysis text
        self.analysis_text.config(state='normal')
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', '\n'.join(analysis))
        self.analysis_text.config(state='disabled')
        
        # Update status
        self.update_status(f"Simulation complete — Equity: {results['equity']*100:.1f}%", 'green')
        self.calculate_btn.set_enabled(True)
        
        # Save to state.json
        state = {
            "my_cards": results['hand_cards'],
            "board": results['board_cards'],
            "pot": None,
            "to_call": None,
            "stacks": {},
            "equity": results['equity'],
            "kelly": results['kelly']
        }
        
        with open("output/state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def show_error(self, error_msg):
        """Show error message"""
        self.analysis_text.config(state='normal')
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', f"ERROR:\n{error_msg}")
        self.analysis_text.config(state='disabled')
        self.update_status("Error occurred", 'red')
        self.calculate_btn.set_enabled(True)
    
    def clear_inputs(self):
        """Clear all inputs"""
        self.hand_entry.delete(0, tk.END)
        self.board_entry.delete(0, tk.END)
        
        # Reset to defaults
        self.hand_entry.insert(0, "As Kh")
        self.board_entry.insert(0, "7h 2d 2s")
        self.players_var.set("6")
        self.trials_var.set("10000")
        
        # Reset displays
        self.equity_label.config(text="--.--%", fg=COLORS['text_primary'])
        self.win_label.config(text="--.--%")
        self.tie_label.config(text="--.--%")
        self.kelly_label.config(text="--.--%")
        self.equity_gauge.set_value(0)
        
        self.analysis_text.config(state='normal')
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', "Enter cards and click Calculate to run simulation...")
        self.analysis_text.config(state='disabled')
        
        self.update_card_preview(None)
        self.update_status("Cleared — Ready for new calculation", 'green')
        
    def update_status(self, message, color='green'):
        """Update status label"""
        color_map = {
            'green': COLORS['accent_green'],
            'red': COLORS['accent_red'],
            'gold': COLORS['accent_gold'],
        }
        self.status_dot.config(fg=color_map.get(color, COLORS['accent_green']))
        self.status_label.config(text=message)
