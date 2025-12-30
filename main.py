#!/usr/bin/env python3
"""
PokerHelper - Main Application
Professional GUI for poker card detection and win probability calculation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
import threading
import time
import platform

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'poker_vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'simulator'))

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

# Fallback to system default if font not available
# Note: We can't test fonts before Tk is initialized, so we'll handle this in configure_styles

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE - Sophisticated dark poker theme
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'bg_dark': '#0d1117',           # Deep background
    'bg_card': '#161b22',           # Card/panel background
    'bg_elevated': '#21262d',       # Elevated surfaces
    'bg_hover': '#30363d',          # Hover states
    'border': '#30363d',            # Borders
    'border_light': '#484f58',      # Light borders
    
    'text_primary': '#f0f6fc',      # Primary text
    'text_secondary': '#8b949e',    # Secondary text
    'text_muted': '#6e7681',        # Muted text
    
    'accent_green': '#238636',      # Success/Hand selection
    'accent_green_hover': '#2ea043',
    'accent_red': '#da3633',        # Danger/Board selection
    'accent_red_hover': '#f85149',
    'accent_blue': '#1f6feb',       # Primary action
    'accent_blue_hover': '#388bfd',
    'accent_gold': '#d29922',       # Warning/Highlights
    'accent_gold_hover': '#e3b341',
    'accent_purple': '#8957e5',     # Special
    
    'card_red': '#ef4444',          # Red suits (hearts/diamonds)
    'card_black': '#f0f6fc',        # Black suits (spades/clubs)
}

class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, color='blue', width=200, height=50, icon=None, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.icon = icon
        self.width = width
        self.height = height
        
        # Color mapping
        color_map = {
            'blue': (COLORS['accent_blue'], COLORS['accent_blue_hover']),
            'green': (COLORS['accent_green'], COLORS['accent_green_hover']),
            'red': (COLORS['accent_red'], COLORS['accent_red_hover']),
            'gold': (COLORS['accent_gold'], COLORS['accent_gold_hover']),
            'purple': (COLORS['accent_purple'], '#a371f7'),
        }
        self.normal_color, self.hover_color = color_map.get(color, color_map['blue'])
        self.current_color = self.normal_color
        
        self.draw_button()
        
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        
    def draw_button(self):
        self.delete('all')
        # Rounded rectangle
        self.create_rounded_rect(2, 2, self.width-2, self.height-2, 
                                radius=12, fill=self.current_color, outline='')
        
        # Text with optional icon
        display_text = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(self.width//2, self.height//2, text=display_text,
                        fill=COLORS['text_primary'], font=(FONT_FAMILY, 13, 'bold'))
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def on_enter(self, event):
        self.current_color = self.hover_color
        self.draw_button()
        self.config(cursor='hand2')
        
    def on_leave(self, event):
        self.current_color = self.normal_color
        self.draw_button()
        
    def on_click(self, event):
        if self.command:
            self.command()


class CardDisplay(tk.Canvas):
    """Visual card display widget"""
    def __init__(self, parent, card_text="?", width=60, height=85, **kwargs):
        super().__init__(parent, width=width, height=height,
                        bg=COLORS['bg_dark'], highlightthickness=0, **kwargs)
        self.card_text = card_text
        self.width = width
        self.height = height
        self.draw_card()
        
    def draw_card(self):
        self.delete('all')
        
        if self.card_text == "?":
            # Empty card placeholder
            self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                    radius=8, fill=COLORS['bg_elevated'], 
                                    outline=COLORS['border'])
            self.create_text(self.width//2, self.height//2, text="?",
                           fill=COLORS['text_muted'], font=(FONT_FAMILY, 24, 'bold'))
        else:
            # Actual card
            self.create_rounded_rect(2, 2, self.width-2, self.height-2,
                                    radius=8, fill='#ffffff', outline=COLORS['border'])
            
            # Determine suit color
            suit = self.card_text[-1].lower() if len(self.card_text) > 1 else ''
            color = COLORS['card_red'] if suit in ['h', 'd'] else '#1a1a1a'
            
            # Suit symbols
            suit_symbols = {'h': '♥', 'd': '♦', 's': '♠', 'c': '♣'}
            rank = self.card_text[:-1] if len(self.card_text) > 1 else self.card_text
            suit_symbol = suit_symbols.get(suit, '')
            
            # Draw rank and suit
            self.create_text(self.width//2, self.height//2 - 10, text=rank,
                           fill=color, font=(FONT_FAMILY, 20, 'bold'))
            self.create_text(self.width//2, self.height//2 + 18, text=suit_symbol,
                           fill=color, font=(FONT_FAMILY, 22))
    
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


class PokerHelperApp:
    def __init__(self, root):
        try:
            self.root = root
            self.root.title("PokerHelper Pro")
            self.root.geometry("1100x700")
            self.root.configure(bg=COLORS['bg_dark'])
            self.root.resizable(True, True)
            
            # Configure ttk styles first
            self.configure_styles()
            
            # Check window still exists after style configuration
            if not self.root.winfo_exists():
                raise RuntimeError("Window was destroyed during style configuration")
            
            # Setup UI before centering
            self.setup_ui()
            
            # Check window still exists after UI setup
            if not self.root.winfo_exists():
                raise RuntimeError("Window was destroyed during UI setup")
            
            # Center window on screen after UI is set up
            self.center_window()
            
            # Force initial update
            if self.root.winfo_exists():
                try:
                    self.root.update_idletasks()
                    self.root.update()
                except tk.TclError as e:
                    # Window was destroyed, can't update
                    print(f"Warning: Could not update window: {e}")
                    raise RuntimeError("Window was destroyed during update") from e
        except Exception as e:
            import traceback
            error_msg = f"Error initializing application: {e}\n\n{traceback.format_exc()}"
            print(error_msg)
            # Only show error dialog if window still exists
            try:
                if self.root.winfo_exists():
                    messagebox.showerror("Initialization Error", str(e))
            except:
                pass
            # Don't raise - let the program continue if possible
            # If window was destroyed, there's nothing we can do
        
    def center_window(self):
        """Center the window on screen"""
        try:
            # Check if window still exists
            if not self.root.winfo_exists():
                return
            self.root.update_idletasks()
            width = 1100
            height = 700
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
        except (tk.TclError, AttributeError) as e:
            # Window was destroyed or doesn't exist
            print(f"Warning: Could not center window: {e}")
            return
        except Exception as e:
            # Other errors - try default geometry if window still exists
            print(f"Warning: Could not center window: {e}")
            try:
                if self.root.winfo_exists():
                    self.root.geometry('1100x700')
            except:
                pass
        
    def configure_styles(self):
        """Configure ttk widget styles"""
        try:
            style = ttk.Style()
            style.theme_use('clam')
        except Exception as e:
            print(f"Warning: Could not configure styles: {e}")
            # Continue without custom styles
            return
        
        # Configure various styles
        style.configure('TFrame', background=COLORS['bg_dark'])
        style.configure('Card.TFrame', background=COLORS['bg_card'])
        
        style.configure('Title.TLabel', 
                       font=(FONT_FAMILY, 32, 'bold'),
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text_primary'])
        
        style.configure('Subtitle.TLabel',
                       font=(FONT_FAMILY, 14),
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text_secondary'])
        
        style.configure('Info.TLabel',
                       font=(FONT_FAMILY, 11),
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text_muted'])
        
        style.configure('Status.TLabel',
                       font=(FONT_FAMILY, 11),
                       background=COLORS['bg_card'],
                       foreground=COLORS['accent_green'])
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Check if window still exists before starting
        if not self.root.winfo_exists():
            raise RuntimeError("Window was destroyed before setup_ui could complete")
        
        try:
            # Main container with padding
            main_container = tk.Frame(self.root, bg=COLORS['bg_dark'])
            main_container.pack(fill='both', expand=True, padx=40, pady=30)
            
            # ═══════════════════════════════════════════════════════════════════
            # HEADER SECTION
            # ═══════════════════════════════════════════════════════════════════
            header_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
            header_frame.pack(fill='x', pady=(0, 30))
        
            # Logo/Title area
            title_area = tk.Frame(header_frame, bg=COLORS['bg_dark'])
            title_area.pack()
            
            # Poker chip icon (using unicode)
            icon_label = tk.Label(title_area, text="🎰", font=('', 48),
                                 bg=COLORS['bg_dark'])
            icon_label.pack()
            
            # Main title
            title_label = tk.Label(title_area, text="PokerHelper Pro",
                                  font=(FONT_FAMILY, 36, 'bold'),
                                  bg=COLORS['bg_dark'], fg=COLORS['text_primary'])
            title_label.pack(pady=(10, 5))
            
            # Subtitle
            subtitle_label = tk.Label(title_area, 
                                     text="Real-time Card Detection & Win Probability Analysis",
                                     font=(FONT_FAMILY, 14),
                                     bg=COLORS['bg_dark'], fg=COLORS['text_secondary'])
            subtitle_label.pack()
            
            # Divider line
            divider = tk.Frame(main_container, height=1, bg=COLORS['border'])
            divider.pack(fill='x', pady=20)
            
            # ═══════════════════════════════════════════════════════════════════
            # MAIN ACTIONS SECTION
            # ═══════════════════════════════════════════════════════════════════
            actions_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
            actions_frame.pack(fill='x', pady=20)
            
            # Feature cards container
            cards_frame = tk.Frame(actions_frame, bg=COLORS['bg_dark'])
            cards_frame.pack()
            
            # Poker Vision Card
            vision_card = self.create_feature_card(
                cards_frame,
                icon="🎯",
                title="Poker Vision",
                description="Automatically detect cards from your screen in real-time",
                button_text="Launch Vision",
                button_color='green',
                command=self.open_poker_vision
            )
            vision_card.pack(side='left', padx=15)
            
            # Manual Simulator Card
            simulator_card = self.create_feature_card(
                cards_frame,
                icon="🎲",
                title="Manual Simulator",
                description="Enter cards manually to calculate win probability",
                button_text="Open Simulator",
                button_color='blue',
                command=self.open_simulator
            )
            simulator_card.pack(side='left', padx=15)
            
            # Auto Bot Card
            bot_card = self.create_feature_card(
                cards_frame,
                icon="🤖",
                title="Auto Bot",
                description="Automated poker assistant with configurable play styles",
                button_text="Launch Bot",
                button_color='purple',
                command=self.open_poker_bot
            )
            bot_card.pack(side='left', padx=15)
            
            # Settings Card
            settings_card = self.create_feature_card(
                cards_frame,
                icon="⚙️",
                title="Settings",
                description="Configure detection parameters and preferences",
                button_text="Open Settings",
                button_color='gold',
                command=self.open_settings
            )
            settings_card.pack(side='left', padx=15)
            
            # ═══════════════════════════════════════════════════════════════════
            # INFO SECTION
            # ═══════════════════════════════════════════════════════════════════
            info_frame = tk.Frame(main_container, bg=COLORS['bg_card'], 
                                 highlightbackground=COLORS['border'], highlightthickness=1)
            info_frame.pack(fill='x', pady=30, ipady=20, ipadx=20)
            
            info_title = tk.Label(info_frame, text="Quick Start Guide",
                                 font=(FONT_FAMILY, 14, 'bold'),
                                 bg=COLORS['bg_card'], fg=COLORS['text_primary'])
            info_title.pack(pady=(10, 15))
            
            steps = [
                ("1", "Launch Poker Vision to auto-detect cards from your poker client"),
                ("2", "Select the hand and board regions on your screen"),
                ("3", "View real-time equity calculations as cards are detected"),
            ]
            
            for num, text in steps:
                step_frame = tk.Frame(info_frame, bg=COLORS['bg_card'])
                step_frame.pack(fill='x', padx=20, pady=5)
                
                num_label = tk.Label(step_frame, text=num, width=3,
                                    font=(FONT_FAMILY, 12, 'bold'),
                                    bg=COLORS['accent_blue'], fg=COLORS['text_primary'])
                num_label.pack(side='left', padx=(0, 15))
                
                text_label = tk.Label(step_frame, text=text,
                                     font=(FONT_FAMILY, 12),
                                     bg=COLORS['bg_card'], fg=COLORS['text_secondary'])
                text_label.pack(side='left')
            
            # ═══════════════════════════════════════════════════════════════════
            # STATUS BAR
            # ═══════════════════════════════════════════════════════════════════
            status_frame = tk.Frame(main_container, bg=COLORS['bg_card'],
                                   highlightbackground=COLORS['border'], highlightthickness=1)
            status_frame.pack(side='bottom', fill='x', pady=(20, 0))
            
            status_inner = tk.Frame(status_frame, bg=COLORS['bg_card'])
            status_inner.pack(fill='x', padx=15, pady=10)
            
            # Status indicator
            self.status_dot = tk.Label(status_inner, text="●", 
                                      font=('', 10),
                                      bg=COLORS['bg_card'], fg=COLORS['accent_green'])
            self.status_dot.pack(side='left')
            
            self.status_label = tk.Label(status_inner, text="System Ready",
                                        font=(FONT_FAMILY, 11),
                                        bg=COLORS['bg_card'], fg=COLORS['text_secondary'])
            self.status_label.pack(side='left', padx=(8, 0))
            
            # Version info
            version_label = tk.Label(status_inner, text="v2.0",
                                    font=(FONT_FAMILY, 10),
                                    bg=COLORS['bg_card'], fg=COLORS['text_muted'])
            version_label.pack(side='right')
            
            # Force update to ensure widgets are displayed
            if self.root.winfo_exists():
                self.root.update_idletasks()
                self.root.update()
        except Exception as e:
            import traceback
            error_msg = f"Error setting up UI: {e}\n\n{traceback.format_exc()}"
            print(error_msg)
            # Create a simple error display if window still exists
            try:
                if self.root.winfo_exists():
                    error_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
                    error_frame.pack(fill='both', expand=True, padx=20, pady=20)
                    error_label = tk.Label(error_frame, text=f"Error: {e}\n\nSee console for details", 
                                         font=('TkDefaultFont', 12),
                                         bg=COLORS['bg_dark'], fg=COLORS['accent_red'],
                                         wraplength=1000, justify='left')
                    error_label.pack()
                    self.root.update()
            except:
                pass
            # Re-raise the exception so __init__ can handle it
            raise
        
    def create_feature_card(self, parent, icon, title, description, button_text, button_color, command):
        """Create a feature card widget"""
        card = tk.Frame(parent, bg=COLORS['bg_card'], width=240, height=280,
                       highlightbackground=COLORS['border'], highlightthickness=1)
        card.pack_propagate(False)
        
        # Icon
        icon_label = tk.Label(card, text=icon, font=('', 40),
                             bg=COLORS['bg_card'])
        icon_label.pack(pady=(25, 15))
        
        # Title
        title_label = tk.Label(card, text=title,
                              font=(FONT_FAMILY, 16, 'bold'),
                              bg=COLORS['bg_card'], fg=COLORS['text_primary'])
        title_label.pack(pady=(0, 10))
        
        # Description
        desc_label = tk.Label(card, text=description,
                             font=(FONT_FAMILY, 11),
                             bg=COLORS['bg_card'], fg=COLORS['text_secondary'],
                             wraplength=200, justify='center')
        desc_label.pack(pady=(0, 20))
        
        # Button
        btn = ModernButton(card, text=button_text, command=command,
                          color=button_color, width=180, height=44)
        btn.pack(pady=(0, 20))
        
        # Hover effect for card
        def on_enter(e):
            card.configure(highlightbackground=COLORS['border_light'])
        def on_leave(e):
            card.configure(highlightbackground=COLORS['border'])
            
        card.bind('<Enter>', on_enter)
        card.bind('<Leave>', on_leave)
        
        return card
        
    def open_poker_vision(self):
        """Open the poker vision detection window"""
        try:
            from poker_vision_detector import PokerVisionDetector
            self.poker_vision_window = PokerVisionDetector(self.root)
            self.update_status("Poker Vision Detection launched", 'green')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Poker Vision: {e}")
            self.update_status("Error launching Poker Vision", 'red')
            
    def open_simulator(self):
        """Open the manual simulator window"""
        try:
            from simulator_gui import SimulatorGUI
            self.simulator_window = SimulatorGUI(self.root)
            self.update_status("Manual Simulator launched", 'green')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Simulator: {e}")
            self.update_status("Error launching Simulator", 'red')
    
    def open_poker_bot(self):
        """Open the automated poker bot window"""
        try:
            from poker_bot_gui import PokerBotGUI
            self.poker_bot_window = PokerBotGUI(self.root)
            self.update_status("Poker Bot launched", 'green')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Poker Bot: {e}")
            self.update_status("Error launching Poker Bot", 'red')
            
    def open_settings(self):
        """Open settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.configure(bg=COLORS['bg_dark'])
        settings_window.transient(self.root)
        
        # Center settings window
        settings_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 200
        settings_window.geometry(f'+{x}+{y}')
        
        # Header
        header = tk.Frame(settings_window, bg=COLORS['bg_dark'])
        header.pack(fill='x', padx=30, pady=20)
        
        tk.Label(header, text="⚙️  Settings",
                font=(FONT_FAMILY, 20, 'bold'),
                bg=COLORS['bg_dark'], fg=COLORS['text_primary']).pack(anchor='w')
        
        tk.Label(header, text="Configure application preferences",
                font=(FONT_FAMILY, 12),
                bg=COLORS['bg_dark'], fg=COLORS['text_secondary']).pack(anchor='w', pady=(5, 0))
        
        # Settings content
        content = tk.Frame(settings_window, bg=COLORS['bg_card'],
                          highlightbackground=COLORS['border'], highlightthickness=1)
        content.pack(fill='both', expand=True, padx=30, pady=(0, 20))
        
        # Debug mode option
        self.debug_var = tk.BooleanVar()
        debug_frame = tk.Frame(content, bg=COLORS['bg_card'])
        debug_frame.pack(fill='x', padx=20, pady=15)
        
        debug_check = tk.Checkbutton(debug_frame, text="Enable Debug Mode",
                                    variable=self.debug_var,
                                    font=(FONT_FAMILY, 12),
                                    bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                                    selectcolor=COLORS['bg_elevated'],
                                    activebackground=COLORS['bg_card'],
                                    activeforeground=COLORS['text_primary'])
        debug_check.pack(anchor='w')
        
        debug_desc = tk.Label(debug_frame, text="Show additional logging and debug information",
                             font=(FONT_FAMILY, 10),
                             bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        debug_desc.pack(anchor='w', padx=(24, 0))
        
        # Save debug images option
        self.save_images_var = tk.BooleanVar()
        images_frame = tk.Frame(content, bg=COLORS['bg_card'])
        images_frame.pack(fill='x', padx=20, pady=15)
        
        images_check = tk.Checkbutton(images_frame, text="Save Debug Images",
                                     variable=self.save_images_var,
                                     font=(FONT_FAMILY, 12),
                                     bg=COLORS['bg_card'], fg=COLORS['text_primary'],
                                     selectcolor=COLORS['bg_elevated'],
                                     activebackground=COLORS['bg_card'],
                                     activeforeground=COLORS['text_primary'])
        images_check.pack(anchor='w')
        
        images_desc = tk.Label(images_frame, text="Save captured card images to output/debug folder",
                              font=(FONT_FAMILY, 10),
                              bg=COLORS['bg_card'], fg=COLORS['text_muted'])
        images_desc.pack(anchor='w', padx=(24, 0))
        
        # Divider
        tk.Frame(content, height=1, bg=COLORS['border']).pack(fill='x', padx=20, pady=15)
        
        # Detection settings label
        tk.Label(content, text="Detection Settings",
                font=(FONT_FAMILY, 12, 'bold'),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w', padx=20)
        
        # Confidence threshold
        thresh_frame = tk.Frame(content, bg=COLORS['bg_card'])
        thresh_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(thresh_frame, text="Confidence Threshold",
                font=(FONT_FAMILY, 12),
                bg=COLORS['bg_card'], fg=COLORS['text_primary']).pack(anchor='w')
        
        self.threshold_var = tk.DoubleVar(value=0.3)
        threshold_scale = ttk.Scale(thresh_frame, from_=0.1, to=0.9,
                                   variable=self.threshold_var, orient='horizontal')
        threshold_scale.pack(fill='x', pady=(5, 0))
        
        # Save button
        btn_frame = tk.Frame(settings_window, bg=COLORS['bg_dark'])
        btn_frame.pack(fill='x', padx=30, pady=(0, 20))
        
        save_btn = ModernButton(btn_frame, text="Save Settings", 
                               command=lambda: self.save_settings(settings_window),
                               color='green', width=150, height=40)
        save_btn.pack(side='right')
        
    def save_settings(self, window):
        """Save settings to config file"""
        settings = {
            'debug_mode': self.debug_var.get(),
            'save_debug_images': self.save_images_var.get(),
            'confidence_threshold': self.threshold_var.get()
        }
        
        with open('config.json', 'w') as f:
            json.dump(settings, f, indent=2)
            
        self.update_status("Settings saved successfully", 'green')
        window.destroy()
        
    def update_status(self, message, color='green'):
        """Update status label"""
        color_map = {
            'green': COLORS['accent_green'],
            'red': COLORS['accent_red'],
            'gold': COLORS['accent_gold'],
        }
        self.status_dot.config(fg=color_map.get(color, COLORS['accent_green']))
        self.status_label.config(text=message)
        self.root.update()


def main():
    try:
        root = tk.Tk()
        
        # Try to set window icon (if available)
        try:
            # For macOS, we can't easily set icons, so we skip
            pass
        except:
            pass
        
        app = PokerHelperApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        error_msg = f"Fatal error: {e}\n\n{traceback.format_exc()}"
        print(error_msg)
        # Try to show error in a message box if possible
        try:
            import tkinter.messagebox as mb
            mb.showerror("Fatal Error", error_msg)
        except:
            pass
        raise


if __name__ == "__main__":
    main()
