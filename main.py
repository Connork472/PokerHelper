#!/usr/bin/env python3
"""
PokerHelper - Main Application
Clean GUI for poker card detection and win probability calculation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
import threading
import time

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'poker_vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'simulator'))

class PokerHelperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PokerHelper - Card Detection & Win Probability")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='white')
        self.style.configure('Info.TLabel', font=('Arial', 10), background='#2c3e50', foreground='#ecf0f1')

        # Button styling to improve visibility
        button_base = {
            'font': ('Arial', 12, 'bold'),
            'padding': (16, 10),
            'foreground': 'white',
            'background': '#3498db',
            'borderwidth': 0,
            'relief': 'flat'
        }
        self.style.configure('TButton', **button_base)
        self.style.configure('Button.TButton', **button_base)
        button_state_map = {
            'background': [
                ('pressed', '#2471a3'),
                ('active', '#2980b9'),
                ('disabled', '#95a5a6')
            ],
            'foreground': [
                ('disabled', '#ecf0f1')
            ]
        }
        self.style.map('TButton', **button_state_map)
        self.style.map('Button.TButton', **button_state_map)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=20)
        
        title_label = ttk.Label(title_frame, text="PokerHelper", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Card Detection & Win Probability Calculator", 
                                 style='Info.TLabel')
        subtitle_label.pack()
        
        # Main buttons frame
        buttons_frame = tk.Frame(self.root, bg='#2c3e50')
        buttons_frame.pack(pady=40)
        
        # Poker Vision Detection Button
        vision_btn = ttk.Button(buttons_frame, text="🎯 Poker Vision Detection", 
                               command=self.open_poker_vision, style='Button.TButton')
        vision_btn.pack(pady=10, padx=20, fill='x')
        
        # Manual Simulator Button
        simulator_btn = ttk.Button(buttons_frame, text="🎲 Manual Simulator", 
                                   command=self.open_simulator, style='Button.TButton')
        simulator_btn.pack(pady=10, padx=20, fill='x')
        
        # Settings Button
        settings_btn = ttk.Button(buttons_frame, text="⚙️ Settings", 
                                 command=self.open_settings, style='Button.TButton')
        settings_btn.pack(pady=10, padx=20, fill='x')
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#2c3e50')
        status_frame.pack(side='bottom', fill='x', padx=20, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Info.TLabel')
        self.status_label.pack()
        
        # Info text
        info_frame = tk.Frame(self.root, bg='#2c3e50')
        info_frame.pack(pady=20)
        
        info_text = """
Poker Vision: Automatically detect cards from screen capture
Manual Simulator: Enter cards manually for win probability calculation
        """
        
        info_label = ttk.Label(info_frame, text=info_text.strip(), style='Info.TLabel')
        info_label.pack()
        
    def open_poker_vision(self):
        """Open the poker vision detection window"""
        try:
            from poker_vision_detector import PokerVisionDetector
            self.poker_vision_window = PokerVisionDetector(self.root)
            self.update_status("Poker Vision Detection opened")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Poker Vision: {e}")
            
    def open_simulator(self):
        """Open the manual simulator window"""
        try:
            from simulator_gui import SimulatorGUI
            self.simulator_window = SimulatorGUI(self.root)
            self.update_status("Manual Simulator opened")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Simulator: {e}")
            
    def open_settings(self):
        """Open settings window"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#2c3e50')
        
        # Settings content
        ttk.Label(settings_window, text="Settings", style='Title.TLabel').pack(pady=20)
        
        # Debug mode checkbox
        self.debug_var = tk.BooleanVar()
        debug_check = ttk.Checkbutton(settings_window, text="Debug Mode", 
                                     variable=self.debug_var)
        debug_check.pack(pady=10)
        
        # Save debug images checkbox
        self.save_images_var = tk.BooleanVar()
        save_check = ttk.Checkbutton(settings_window, text="Save Debug Images", 
                                    variable=self.save_images_var)
        save_check.pack(pady=10)
        
        # Save button
        save_btn = ttk.Button(settings_window, text="Save Settings", 
                             command=lambda: self.save_settings(settings_window))
        save_btn.pack(pady=20)
        
    def save_settings(self, window):
        """Save settings to config file"""
        settings = {
            'debug_mode': self.debug_var.get(),
            'save_debug_images': self.save_images_var.get()
        }
        
        with open('config.json', 'w') as f:
            json.dump(settings, f, indent=2)
            
        messagebox.showinfo("Settings", "Settings saved successfully!")
        window.destroy()
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()

def main():
    root = tk.Tk()
    app = PokerHelperApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
