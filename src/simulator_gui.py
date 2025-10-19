#!/usr/bin/env python3
"""
Simulator GUI - Manual Card Input
Clean interface for manual card entry and win probability calculation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import threading
import time
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulator'))
from poker_cli_session import tokenize_cards, to_treys, simulate, kelly_even_money

class SimulatorGUI:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Poker Simulator - Manual Input")
        self.window.geometry("800x600")
        self.window.configure(bg='#2c3e50')
        
        # Simulation state
        self.simulation_running = False
        self.current_equity = None
        self.current_kelly = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.window, bg='#2c3e50')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="Poker Simulator - Manual Input", 
                              font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack()
        
        # Card input frame
        input_frame = tk.Frame(self.window, bg='#2c3e50')
        input_frame.pack(pady=20, fill='x', padx=20)
        
        # Hand input
        hand_frame = tk.Frame(input_frame, bg='#2c3e50')
        hand_frame.pack(fill='x', pady=10)
        
        tk.Label(hand_frame, text="Your Hand (2 cards):", 
                font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#27ae60').pack(side='left')
        
        self.hand_entry = tk.Entry(hand_frame, font=('Arial', 12), width=20)
        self.hand_entry.pack(side='left', padx=10)
        self.hand_entry.insert(0, "As Kh")
        
        # Board input
        board_frame = tk.Frame(input_frame, bg='#2c3e50')
        board_frame.pack(fill='x', pady=10)
        
        tk.Label(board_frame, text="Board Cards (0-5):", 
                font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#e74c3c').pack(side='left')
        
        self.board_entry = tk.Entry(board_frame, font=('Arial', 12), width=30)
        self.board_entry.pack(side='left', padx=10)
        self.board_entry.insert(0, "7h 2d 2s")
        
        # Settings frame
        settings_frame = tk.Frame(input_frame, bg='#2c3e50')
        settings_frame.pack(fill='x', pady=10)
        
        # Players setting
        players_frame = tk.Frame(settings_frame, bg='#2c3e50')
        players_frame.pack(side='left', padx=10)
        
        tk.Label(players_frame, text="Players:", 
                font=('Arial', 10), bg='#2c3e50', fg='white').pack(side='left')
        
        self.players_var = tk.StringVar(value="6")
        players_spin = tk.Spinbox(players_frame, from_=2, to=10, textvariable=self.players_var, 
                                font=('Arial', 10), width=5)
        players_spin.pack(side='left', padx=5)
        
        # Trials setting
        trials_frame = tk.Frame(settings_frame, bg='#2c3e50')
        trials_frame.pack(side='left', padx=10)
        
        tk.Label(trials_frame, text="Trials:", 
                font=('Arial', 10), bg='#2c3e50', fg='white').pack(side='left')
        
        self.trials_var = tk.StringVar(value="10000")
        trials_spin = tk.Spinbox(trials_frame, from_=1000, to=100000, textvariable=self.trials_var, 
                                font=('Arial', 10), width=8)
        trials_spin.pack(side='left', padx=5)
        
        # Control buttons
        controls_frame = tk.Frame(self.window, bg='#2c3e50')
        controls_frame.pack(pady=20)
        
        self.calculate_btn = tk.Button(controls_frame, text="Calculate Win Probability", 
                                     command=self.calculate_probability, font=('Arial', 12),
                                     bg='#3498db', fg='white', padx=20, pady=10)
        self.calculate_btn.pack(side='left', padx=10)
        
        self.clear_btn = tk.Button(controls_frame, text="Clear", 
                                  command=self.clear_inputs, font=('Arial', 12),
                                  bg='#f39c12', fg='white', padx=20, pady=10)
        self.clear_btn.pack(side='left', padx=10)
        
        # Results frame
        results_frame = tk.Frame(self.window, bg='#2c3e50')
        results_frame.pack(pady=20, fill='both', expand=True, padx=20)
        
        # Results title
        results_title = tk.Label(results_frame, text="Simulation Results", 
                                font=('Arial', 14, 'bold'), bg='#2c3e50', fg='white')
        results_title.pack(pady=10)
        
        # Results display
        self.results_text = tk.Text(results_frame, height=15, width=70, 
                                   font=('Courier', 10), bg='#34495e', fg='white')
        self.results_text.pack(fill='both', expand=True)
        
        # Status
        status_frame = tk.Frame(self.window, bg='#2c3e50')
        status_frame.pack(side='bottom', fill='x', padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready - Enter your cards and click Calculate", 
                                   font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack()
        
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
            
            self.status_label.config(text="Calculating... Please wait")
            self.calculate_btn.config(state='disabled')
            
            # Start calculation in thread
            calc_thread = threading.Thread(target=self.run_simulation, 
                                         args=(hand_cards, board_cards, players, trials), 
                                         daemon=True)
            calc_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            self.status_label.config(text="Error in input")
            self.calculate_btn.config(state='normal')
    
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
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        result_text = f"""
=== POKER SIMULATION RESULTS ===

Hand: {' '.join(results['hand_cards'])}
Board: {' '.join(results['board_cards']) if results['board_cards'] else '(none)'}
Players: {results['players']}
Trials: {results['trials']:,}

=== PROBABILITY ANALYSIS ===
Win%: {results['win_p']*100:.2f}%
Tie%: {results['tie_p']*100:.2f}%
Equity: {results['equity']*100:.2f}%

=== KELLY CRITERION ===
Kelly (even-money): {results['kelly']*100:.2f}% of bankroll

=== CONFIDENCE INTERVAL ===
95% CI (equity): ±{results['moe']*100:.2f}%

=== INTERPRETATION ===
"""
        
        # Add interpretation
        if results['equity'] > 0.6:
            result_text += "🔥 STRONG HAND - High equity, consider betting aggressively"
        elif results['equity'] > 0.5:
            result_text += "✅ GOOD HAND - Positive equity, play for value"
        elif results['equity'] > 0.4:
            result_text += "⚠️  MARGINAL HAND - Close to break-even, play carefully"
        else:
            result_text += "❌ WEAK HAND - Negative equity, consider folding"
        
        result_text += f"\n\nKelly suggests betting {results['kelly']*100:.1f}% of your bankroll"
        
        if results['kelly'] > 0.1:
            result_text += " (HIGH CONFIDENCE)"
        elif results['kelly'] > 0.05:
            result_text += " (MODERATE CONFIDENCE)"
        else:
            result_text += " (LOW CONFIDENCE)"
        
        # Display results
        self.results_text.insert(tk.END, result_text)
        
        # Update status
        self.status_label.config(text=f"Calculation complete - Equity: {results['equity']*100:.1f}%")
        self.calculate_btn.config(state='normal')
        
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
        
        with open("state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def show_error(self, error_msg):
        """Show error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"ERROR: {error_msg}")
        self.status_label.config(text="Error occurred")
        self.calculate_btn.config(state='normal')
    
    def clear_inputs(self):
        """Clear all inputs"""
        self.hand_entry.delete(0, tk.END)
        self.board_entry.delete(0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.status_label.config(text="Inputs cleared")
        
        # Reset to defaults
        self.hand_entry.insert(0, "As Kh")
        self.board_entry.insert(0, "7h 2d 2s")
        self.players_var.set("6")
        self.trials_var.set("10000")
