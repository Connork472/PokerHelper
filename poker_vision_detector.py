#!/usr/bin/env python3
"""
Poker Vision Detector - GUI Version
Professional interface for real-time card detection with region selection
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import json
import time
import threading
from collections import deque
from mss import mss
import sys
import os
import platform

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'poker_vision'))

# Try to import onnx-based classifier, fallback to alternative if not available
try:
    from classify.infer_onnx_two import TwoHead
    print("Using ONNX-based classifier")
except ImportError:
    print("ONNX runtime not available, using fallback classifier")
    from classify.infer_onnx_fallback import TwoHead

from geometry.card_finder import find_cards

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
    'border': '#30363d',
    'border_light': '#484f58',
    'border_focus': '#1f6feb',
    
    'text_primary': '#f0f6fc',
    'text_secondary': '#8b949e',
    'text_muted': '#6e7681',
    
    'accent_green': '#238636',
    'accent_green_hover': '#2ea043',
    'accent_green_light': '#3fb950',
    'accent_red': '#da3633',
    'accent_red_hover': '#f85149',
    'accent_blue': '#1f6feb',
    'accent_blue_hover': '#388bfd',
    'accent_gold': '#d29922',
    'accent_purple': '#8957e5',
    
    'card_red': '#ef4444',
    'card_black': '#1a1a1a',
    'card_bg': '#ffffff',
    
    'status_active': '#22c55e',
    'status_pending': '#eab308',
    'status_inactive': '#6e7681',
}


class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, color='blue', width=160, height=44, icon=None, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.icon = icon
        self.width = width
        self.height = height
        self.enabled = True
        self.active = False
        
        color_map = {
            'blue': (COLORS['accent_blue'], COLORS['accent_blue_hover']),
            'green': (COLORS['accent_green'], COLORS['accent_green_hover']),
            'red': (COLORS['accent_red'], COLORS['accent_red_hover']),
            'gold': (COLORS['accent_gold'], '#e3b341'),
            'purple': (COLORS['accent_purple'], '#a371f7'),
        }
        self.normal_color, self.hover_color = color_map.get(color, color_map['blue'])
        self.current_color = self.normal_color
        self.base_color = color
        
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
        
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=10, fill=color, outline='')
        
        display_text = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(self.width//2, self.height//2, text=display_text,
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
        
    def set_text(self, text):
        self.text = text
        self.draw_button()


class CardDisplay(tk.Canvas):
    """Visual card display widget"""
    def __init__(self, parent, card_text="?", width=50, height=70, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_card'], highlightthickness=0, **kwargs)
        self.card_text = card_text
        self.width = width
        self.height = height
        self.draw_card()
        
    def draw_card(self):
        self.delete('all')
        
        if not self.card_text or self.card_text == "?":
            # Empty placeholder
            self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                    radius=5, fill=COLORS['bg_elevated'], 
                                    outline=COLORS['border'])
            self.create_text(self.width//2, self.height//2, text="?",
                           fill=COLORS['text_muted'], font=(FONT_FAMILY, 16, 'bold'))
        else:
            # Actual card
            self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                    radius=5, fill=COLORS['card_bg'], outline='')
            
            suit = self.card_text[-1].lower() if len(self.card_text) > 0 else ''
            rank = self.card_text[:-1] if len(self.card_text) > 1 else self.card_text
            
            suit_symbols = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}
            color = COLORS['card_red'] if suit in ['h', 'd'] else COLORS['card_black']
            suit_symbol = suit_symbols.get(suit, '')
            
            self.create_text(self.width//2, self.height//2 - 8, text=rank,
                           fill=color, font=(FONT_FAMILY, 14, 'bold'))
            self.create_text(self.width//2, self.height//2 + 12, text=suit_symbol,
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


class StatusIndicator(tk.Canvas):
    """Status indicator with label"""
    def __init__(self, parent, text, status='inactive', width=180, **kwargs):
        super().__init__(parent, width=width, height=28,
                        bg=COLORS['bg_card'], highlightthickness=0, **kwargs)
        self.text = text
        self.status = status
        self.width = width
        self.draw()
        
    def draw(self):
        self.delete('all')
        
        status_colors = {
            'active': COLORS['status_active'],
            'pending': COLORS['status_pending'],
            'inactive': COLORS['status_inactive'],
        }
        color = status_colors.get(self.status, COLORS['status_inactive'])
        
        # Dot
        self.create_oval(8, 10, 18, 20, fill=color, outline='')
        
        # Text
        self.create_text(26, 14, text=self.text, anchor='w',
                        fill=COLORS['text_secondary'], font=(FONT_FAMILY, 11))
    
    def set_status(self, status, text=None):
        self.status = status
        if text:
            self.text = text
        self.draw()


class PokerVisionDetector:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Poker Vision Pro")
        self.window.geometry("1100x800")
        self.window.configure(bg=COLORS['bg_dark'])
        
        # Center window
        self.window.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 550
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 400
        self.window.geometry(f'+{x}+{y}')
        
        # Detection components
        self.sct = mss()
        self.clf = TwoHead("config/rank.onnx", "config/suit.onnx")
        
        # State management
        self.hand_region = None
        self.board_region = None
        self.detection_active = False
        self.detection_thread = None
        
        # Card displays
        self.hand_card_widgets = []
        self.board_card_widgets = []
        
        # Temporal smoothing
        self.detection_history = deque(maxlen=5)
        self.stable_threshold = 3
        self.last_stable_detection = None
        
        # GUI setup
        self.setup_ui()
        
        # Start screen capture
        self.start_screen_capture()
        
    def setup_ui(self):
        """Setup the user interface"""
        main_container = tk.Frame(self.window, bg=COLORS['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=30, pady=20)
        
        # ═══════════════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════════════
        header_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
        header_frame.pack(fill='x', pady=(0, 20))
        
        header_left = tk.Frame(header_frame, bg=COLORS['bg_dark'])
        header_left.pack(side='left')
        
        tk.Label(header_left, text="🎯  Poker Vision",
                font=(FONT_FAMILY, 24, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['text_primary']).pack(anchor='w')
        
        tk.Label(header_left, text="Real-time card detection from your poker client",
                font=(FONT_FAMILY, 12),
                bg=COLORS['bg_dark'], fg=COLORS['text_secondary']).pack(anchor='w')
        
        # Status indicators in header
        status_frame = tk.Frame(header_frame, bg=COLORS['bg_dark'])
        status_frame.pack(side='right')
        
        self.hand_status = StatusIndicator(status_frame, "Hand Region", 'inactive')
        self.hand_status.pack(side='left', padx=(0, 15))
        
        self.board_status = StatusIndicator(status_frame, "Board Region", 'inactive')
        self.board_status.pack(side='left', padx=(0, 15))
        
        self.detection_status = StatusIndicator(status_frame, "Detection", 'inactive')
        self.detection_status.pack(side='left')
        
        # ═══════════════════════════════════════════════════════════════════
        # INSTRUCTIONS PANEL
        # ═══════════════════════════════════════════════════════════════════
        instructions_panel = tk.Frame(main_container, bg=COLORS['bg_card'],
                                     highlightbackground=COLORS['border'], highlightthickness=1)
        instructions_panel.pack(fill='x', pady=(0, 20))
        
        instructions_inner = tk.Frame(instructions_panel, bg=COLORS['bg_card'])
        instructions_inner.pack(fill='x', padx=20, pady=15)
        
        tk.Label(instructions_inner, text="SETUP WORKFLOW",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        steps_frame = tk.Frame(instructions_inner, bg=COLORS['bg_card'])
        steps_frame.pack(fill='x', pady=(10, 0))
        
        steps = [
            ("1", "H", "Select your HAND region (2 hole cards)", COLORS['accent_green']),
            ("2", "B", "Select the BOARD region (community cards)", COLORS['accent_red']),
            ("3", "D", "Start automatic detection", COLORS['accent_blue']),
            ("4", "R", "Restart the process if needed", COLORS['accent_gold']),
        ]
        
        for num, key, desc, color in steps:
            step = tk.Frame(steps_frame, bg=COLORS['bg_card'])
            step.pack(side='left', padx=(0, 30))
            
            key_label = tk.Label(step, text=key, width=2,
                               font=(FONT_MONO, 10, 'bold'),
                               bg=color, fg=COLORS['text_primary'])
            key_label.pack(side='left', padx=(0, 8), ipadx=4, ipady=2)
            
            tk.Label(step, text=desc,
                    font=(FONT_FAMILY, 11),
                    bg=COLORS['bg_card'], fg=COLORS['text_secondary']).pack(side='left')
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTROL BUTTONS
        # ═══════════════════════════════════════════════════════════════════
        controls_frame = tk.Frame(main_container, bg=COLORS['bg_card'],
                                 highlightbackground=COLORS['border'], highlightthickness=1)
        controls_frame.pack(fill='x', pady=(0, 20))
        
        controls_inner = tk.Frame(controls_frame, bg=COLORS['bg_card'])
        controls_inner.pack(fill='x', padx=20, pady=15)
        
        # Button row
        buttons_row = tk.Frame(controls_inner, bg=COLORS['bg_card'])
        buttons_row.pack()
        
        self.hand_btn = ModernButton(buttons_row, text="Select Hand", icon="H",
                                    command=self.select_hand, color='green',
                                    width=150, height=44)
        self.hand_btn.pack(side='left', padx=(0, 15))
        
        self.board_btn = ModernButton(buttons_row, text="Select Board", icon="B",
                                     command=self.select_board, color='red',
                                     width=150, height=44)
        self.board_btn.pack(side='left', padx=(0, 15))
        
        self.detect_btn = ModernButton(buttons_row, text="Start Detection", icon="D",
                                      command=self.toggle_detection, color='blue',
                                      width=160, height=44)
        self.detect_btn.pack(side='left', padx=(0, 15))
        
        self.restart_btn = ModernButton(buttons_row, text="Restart", icon="R",
                                       command=self.restart, color='gold',
                                       width=120, height=44)
        self.restart_btn.pack(side='left')
        
        # ═══════════════════════════════════════════════════════════════════
        # DETECTED CARDS DISPLAY
        # ═══════════════════════════════════════════════════════════════════
        cards_section = tk.Frame(main_container, bg=COLORS['bg_card'],
                                highlightbackground=COLORS['border'], highlightthickness=1)
        cards_section.pack(fill='x', pady=(0, 20))
        
        cards_inner = tk.Frame(cards_section, bg=COLORS['bg_card'])
        cards_inner.pack(fill='x', padx=25, pady=20)
        
        # Hand cards display
        hand_display = tk.Frame(cards_inner, bg=COLORS['bg_card'])
        hand_display.pack(side='left', padx=(0, 60))
        
        hand_header = tk.Frame(hand_display, bg=COLORS['bg_card'])
        hand_header.pack(anchor='w')
        
        tk.Label(hand_header, text="●", font=('', 10),
                bg=COLORS['bg_card'], fg=COLORS['accent_green']).pack(side='left')
        
        tk.Label(hand_header, text="YOUR HAND",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['accent_green']).pack(side='left', padx=(5, 0))
        
        hand_cards_frame = tk.Frame(hand_display, bg=COLORS['bg_card'])
        hand_cards_frame.pack(anchor='w', pady=(10, 0))
        
        for i in range(2):
            card = CardDisplay(hand_cards_frame)
            card.pack(side='left', padx=(0, 8))
            self.hand_card_widgets.append(card)
        
        self.hand_text_label = tk.Label(hand_display, text="Not detected",
                                       font=(FONT_MONO, 12),
                                       bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        self.hand_text_label.pack(anchor='w', pady=(8, 0))
        
        # Separator
        separator = tk.Frame(cards_inner, width=1, bg=COLORS['border'])
        separator.pack(side='left', fill='y', padx=30)
        
        # Board cards display
        board_display = tk.Frame(cards_inner, bg=COLORS['bg_card'])
        board_display.pack(side='left')
        
        board_header = tk.Frame(board_display, bg=COLORS['bg_card'])
        board_header.pack(anchor='w')
        
        tk.Label(board_header, text="●", font=('', 10),
                bg=COLORS['bg_card'], fg=COLORS['accent_red']).pack(side='left')
        
        tk.Label(board_header, text="COMMUNITY CARDS",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['accent_red']).pack(side='left', padx=(5, 0))
        
        board_cards_frame = tk.Frame(board_display, bg=COLORS['bg_card'])
        board_cards_frame.pack(anchor='w', pady=(10, 0))
        
        for i in range(5):
            card = CardDisplay(board_cards_frame)
            card.pack(side='left', padx=(0, 8))
            self.board_card_widgets.append(card)
        
        self.board_text_label = tk.Label(board_display, text="Not detected",
                                        font=(FONT_MONO, 12),
                                        bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        self.board_text_label.pack(anchor='w', pady=(8, 0))
        
        # ═══════════════════════════════════════════════════════════════════
        # DETECTION LOG
        # ═══════════════════════════════════════════════════════════════════
        log_section = tk.Frame(main_container, bg=COLORS['bg_card'],
                              highlightbackground=COLORS['border'], highlightthickness=1)
        log_section.pack(fill='both', expand=True)
        
        log_inner = tk.Frame(log_section, bg=COLORS['bg_card'])
        log_inner.pack(fill='both', expand=True, padx=20, pady=15)
        
        log_header = tk.Frame(log_inner, bg=COLORS['bg_card'])
        log_header.pack(fill='x')
        
        tk.Label(log_header, text="DETECTION LOG",
                font=(FONT_FAMILY, 11, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(side='left')
        
        self.clear_log_btn = tk.Label(log_header, text="Clear",
                                     font=(FONT_FAMILY, 10),
                                     bg=COLORS['bg_card'], fg=COLORS['accent_blue'],
                                     cursor='hand2')
        self.clear_log_btn.pack(side='right')
        self.clear_log_btn.bind('<Button-1>', lambda e: self.clear_log())
        
        # Log text area
        log_frame = tk.Frame(log_inner, bg=COLORS['bg_elevated'])
        log_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=10,
                               font=(FONT_MONO, 10),
                               bg=COLORS['bg_elevated'], fg=COLORS['text_secondary'],
                               relief='flat', wrap='word',
                               highlightthickness=0,
                               padx=10, pady=10)
        self.log_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.log("System initialized. Select hand and board regions to begin.")
        
        # ═══════════════════════════════════════════════════════════════════
        # STATUS BAR
        # ═══════════════════════════════════════════════════════════════════
        status_bar = tk.Frame(main_container, bg=COLORS['bg_card'],
                             highlightbackground=COLORS['border'], highlightthickness=1)
        status_bar.pack(fill='x', pady=(20, 0))
        
        status_inner = tk.Frame(status_bar, bg=COLORS['bg_card'])
        status_inner.pack(fill='x', padx=15, pady=8)
        
        self.status_dot = tk.Label(status_inner, text="●", font=('', 8),
                                  bg=COLORS['bg_card'], fg=COLORS['status_pending'])
        self.status_dot.pack(side='left')
        
        self.status_label = tk.Label(status_inner, text="Waiting for region selection...",
                                    font=(FONT_FAMILY, 10),
                                    bg=COLORS['bg_card'], fg=COLORS['text_secondary'])
        self.status_label.pack(side='left', padx=(8, 0))
        
        # Keyboard shortcuts hint
        tk.Label(status_inner, text="Shortcuts: H=Hand  B=Board  D=Detect  R=Restart  Q=Quit",
                font=(FONT_FAMILY, 9),
                bg=COLORS['bg_card'], fg=COLORS['text_muted']).pack(side='right')
        
        # Bind keyboard events
        self.window.bind('<KeyPress-h>', lambda e: self.select_hand())
        self.window.bind('<KeyPress-b>', lambda e: self.select_board())
        self.window.bind('<KeyPress-d>', lambda e: self.toggle_detection())
        self.window.bind('<KeyPress-r>', lambda e: self.restart())
        self.window.bind('<KeyPress-q>', lambda e: self.close())
        self.window.focus_set()
        
    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
    def clear_log(self):
        """Clear the log"""
        self.log_text.delete('1.0', tk.END)
        self.log("Log cleared.")
        
    def update_status(self, message, color='pending'):
        """Update status bar"""
        color_map = {
            'active': COLORS['status_active'],
            'pending': COLORS['status_pending'],
            'inactive': COLORS['status_inactive'],
            'error': COLORS['accent_red'],
        }
        self.status_dot.config(fg=color_map.get(color, COLORS['status_pending']))
        self.status_label.config(text=message)
        
    def start_screen_capture(self):
        """Start the screen capture thread"""
        self.capture_thread = threading.Thread(target=self.screen_capture_loop, daemon=True)
        self.capture_thread.start()
        
    def screen_capture_loop(self):
        """Main screen capture and detection loop"""
        while True:
            try:
                # Capture screen
                screen = self.capture_screen()
                
                # Show preview if detection is active
                if self.detection_active:
                    self.show_detection_preview(screen)
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"Screen capture error: {e}")
                time.sleep(1)
    
    def capture_screen(self):
        """Capture the full screen"""
        monitor = self.sct.monitors[1]
        return np.array(self.sct.grab(monitor))[:, :, :3]
    
    def show_detection_preview(self, screen):
        """Show detection preview with regions"""
        preview = screen.copy()
        
        # Draw hand region
        if self.hand_region:
            x, y, w, h = self.hand_region
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(preview, "HAND", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw board region
        if self.board_region:
            x, y, w, h = self.board_region
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(preview, "BOARD", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show preview
        cv2.imshow("Poker Vision Preview", preview)
        cv2.waitKey(1)
    
    def select_hand(self):
        """Select hand region"""
        self.log("Starting hand region selection...")
        self.update_status("Click and drag to select HAND region", 'pending')
        self.start_region_selection("hand")
        
    def select_board(self):
        """Select board region"""
        self.log("Starting board region selection...")
        self.update_status("Click and drag to select BOARD region", 'pending')
        self.start_region_selection("board")
        
    def start_region_selection(self, region_type):
        """Start region selection process"""
        # Start OpenCV selection in thread
        def start_selection():
            region = self.select_region_with_opencv(region_type)
            if region:
                if region_type == "hand":
                    self.hand_region = region
                    self.window.after(0, self.on_hand_selected)
                else:
                    self.board_region = region
                    self.window.after(0, self.on_board_selected)
            else:
                self.window.after(0, lambda: self.log(f"{region_type.title()} selection cancelled."))
                self.window.after(0, lambda: self.update_status("Selection cancelled", 'inactive'))
        
        selection_thread = threading.Thread(target=start_selection, daemon=True)
        selection_thread.start()
        
    def on_hand_selected(self):
        """Called when hand region is selected"""
        self.hand_btn.set_active(True, "Hand Selected ✓")
        self.hand_status.set_status('active', "Hand Region ✓")
        self.log(f"Hand region selected: {self.hand_region}")
        
        if self.board_region:
            self.update_status("Ready to detect. Press D to start.", 'pending')
        else:
            self.update_status("Now select the BOARD region.", 'pending')
    
    def on_board_selected(self):
        """Called when board region is selected"""
        self.board_btn.set_active(True, "Board Selected ✓")
        self.board_status.set_status('active', "Board Region ✓")
        self.log(f"Board region selected: {self.board_region}")
        
        if self.hand_region:
            self.update_status("Ready to detect. Press D to start.", 'pending')
        else:
            self.update_status("Now select the HAND region.", 'pending')
        
    def select_region_with_opencv(self, region_type):
        """Use OpenCV to select region"""
        try:
            # Capture screen
            screen = self.capture_screen()
            
            # Create selection window
            window_name = f"Select {region_type.upper()} - Press ENTER to confirm, ESC to cancel"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            # Selection variables
            drawing = False
            start_point = None
            end_point = None
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, start_point, end_point
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    start_point = (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    end_point = (x, y)
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    end_point = (x, y)
            
            cv2.setMouseCallback(window_name, mouse_callback)
            
            # Selection loop
            while True:
                display = screen.copy()
                
                # Draw instruction overlay
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
                
                color = (0, 255, 0) if region_type == "hand" else (0, 0, 255)
                cv2.putText(display, f"Select {region_type.upper()} region - Click and drag", 
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(display, "Press ENTER to confirm, ESC to cancel", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                
                if start_point and end_point:
                    cv2.rectangle(display, start_point, end_point, color, 3)
                
                cv2.imshow(window_name, display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter
                    if start_point and end_point:
                        x1, y1 = start_point
                        x2, y2 = end_point
                        x, y = min(x1, x2), min(y1, y2)
                        w, h = abs(x2 - x1), abs(y2 - y1)
                        
                        if w > 20 and h > 20:
                            cv2.destroyAllWindows()
                            return (x, y, w, h)
                elif key == 27:  # Escape
                    cv2.destroyAllWindows()
                    return None
                    
        except Exception as e:
            print(f"Region selection error: {e}")
            return None
    
    def toggle_detection(self):
        """Toggle card detection on/off"""
        if not self.hand_region or not self.board_region:
            messagebox.showwarning("Setup Required", 
                                  "Please select both hand and board regions first!")
            return
        
        if self.detection_active:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """Start card detection"""
        self.detection_active = True
        self.detect_btn.set_active(True, "Stop Detection")
        self.detection_status.set_status('active', "Detecting...")
        self.update_status("Detection active — analyzing cards...", 'active')
        self.log("Detection started.")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop card detection"""
        self.detection_active = False
        self.detect_btn.set_active(False, "Start Detection")
        self.detection_status.set_status('inactive', "Detection Paused")
        self.update_status("Detection paused. Press D to resume.", 'pending')
        self.log("Detection stopped.")
        cv2.destroyAllWindows()
        
    def detection_loop(self):
        """Main detection loop"""
        while self.detection_active:
            try:
                # Detect hand cards
                hand_cards = self.detect_cards_in_region(self.hand_region, 2, "hand")
                
                # Detect board cards
                board_cards = self.detect_cards_in_region(self.board_region, 5, "board")
                
                # Update GUI
                self.window.after(0, self.update_results, hand_cards, board_cards)
                
                # Update state.json
                result = {
                    "my_cards": hand_cards,
                    "board": board_cards,
                    "pot": None,
                    "to_call": None,
                    "stacks": {}
                }
                
                with open("output/state.json", "w") as f:
                    json.dump(result, f, indent=2)
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                self.window.after(0, lambda: self.log(f"Detection error: {e}"))
                time.sleep(1)
    
    def detect_cards_in_region(self, region, max_cards, region_name):
        """Detect cards in a region"""
        if not region:
            return []
        
        x, y, w, h = region
        roi = np.array(self.sct.grab({"top": y, "left": x, "width": w, "height": h}))[:, :, :3]
        
        # Save debug image
        cv2.imwrite(f"output/debug/{region_name}_roi_{int(time.time())}.png", roi)
        
        all_cards = []
        
        # Method 1: Contour detection
        try:
            cards = find_cards(roi, min_area=800, aspect_min=0.5, aspect_max=0.9)
            for i, (warped_card, quad) in enumerate(cards[:max_cards]):
                if warped_card.size == 0:
                    continue
                
                cv2.imwrite(f"output/debug/{region_name}_contour_{i}_{int(time.time())}.png", warped_card)
                
                # Detect card
                label, conf = self.clf.predict_with_conf(warped_card, 0.3)
                if conf > 0:
                    all_cards.append(label)
        except Exception as e:
            pass
        
        # Method 2: Grid detection
        try:
            grid_cards = self.detect_cards_grid(roi, max_cards, region_name)
            all_cards.extend(grid_cards)
        except Exception as e:
            pass
        
        # Remove duplicates
        unique_cards = []
        seen = set()
        for card in all_cards:
            if card and card not in seen:
                unique_cards.append(card)
                seen.add(card)
        
        return unique_cards[:max_cards]
    
    def detect_cards_grid(self, roi, max_cards, region_name):
        """Grid-based card detection"""
        h, w = roi.shape[:2]
        cards = []
        
        if max_cards == 2:
            cols = 2
        else:
            cols = 5
        
        card_width = w // cols
        
        for i in range(cols):
            x_start = i * card_width
            x_end = min((i + 1) * card_width, w)
            
            if x_end - x_start < 30:
                continue
            
            card_roi = roi[:, x_start:x_end]
            cv2.imwrite(f"output/debug/{region_name}_grid_{i}_{int(time.time())}.png", card_roi)
            
            label, conf = self.clf.predict_with_conf(card_roi, 0.3)
            if conf > 0:
                cards.append(label)
        
        return cards
    
    def update_results(self, hand_cards, board_cards):
        """Update the results display"""
        # Update hand card widgets
        for i, widget in enumerate(self.hand_card_widgets):
            if i < len(hand_cards):
                widget.set_card(hand_cards[i])
            else:
                widget.set_card("?")
        
        # Update board card widgets
        for i, widget in enumerate(self.board_card_widgets):
            if i < len(board_cards):
                widget.set_card(board_cards[i])
            else:
                widget.set_card("?")
        
        # Update text labels
        hand_str = " ".join(hand_cards) if hand_cards else "Not detected"
        board_str = " ".join(board_cards) if board_cards else "Not detected"
        
        self.hand_text_label.config(text=hand_str,
                                   fg=COLORS['text_primary'] if hand_cards else COLORS['text_muted'])
        self.board_text_label.config(text=board_str,
                                    fg=COLORS['text_primary'] if board_cards else COLORS['text_muted'])
        
        # Update status
        self.update_status(f"Detected: Hand={len(hand_cards)} Board={len(board_cards)}", 'active')
    
    def restart(self):
        """Restart the detection process"""
        self.detection_active = False
        self.hand_region = None
        self.board_region = None
        
        # Reset buttons
        self.hand_btn.set_active(False, "Select Hand")
        self.board_btn.set_active(False, "Select Board")
        self.detect_btn.set_active(False, "Start Detection")
        
        # Reset status indicators
        self.hand_status.set_status('inactive', "Hand Region")
        self.board_status.set_status('inactive', "Board Region")
        self.detection_status.set_status('inactive', "Detection")
        
        # Reset card displays
        for widget in self.hand_card_widgets:
            widget.set_card("?")
        for widget in self.board_card_widgets:
            widget.set_card("?")
        
        self.hand_text_label.config(text="Not detected", fg=COLORS['text_muted'])
        self.board_text_label.config(text="Not detected", fg=COLORS['text_muted'])
        
        self.update_status("Restarted — Select regions to begin again.", 'pending')
        self.log("System restarted. Select hand and board regions to begin.")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
    
    def close(self):
        """Close the detector window"""
        self.detection_active = False
        cv2.destroyAllWindows()
        self.window.destroy()
