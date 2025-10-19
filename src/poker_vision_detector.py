#!/usr/bin/env python3
"""
Poker Vision Detector - GUI Version
Clean interface for card detection with h/b key controls
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

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'poker_vision'))
from classify.infer_onnx_two import TwoHead
from geometry.card_finder import find_cards

class PokerVisionDetector:
    def __init__(self, parent):
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Poker Vision Detector")
        self.window.geometry("1000x700")
        self.window.configure(bg='#2c3e50')
        
        # Detection components
        self.sct = mss()
        self.clf = TwoHead("rank.onnx", "suit.onnx")
        
        # State management
        self.hand_region = None
        self.board_region = None
        self.detection_active = False
        self.detection_thread = None
        
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
        # Title
        title_frame = tk.Frame(self.window, bg='#2c3e50')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="Poker Vision Detector", 
                              font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack()
        
        # Instructions
        instructions_frame = tk.Frame(self.window, bg='#2c3e50')
        instructions_frame.pack(pady=10)
        
        instructions = """
Instructions:
1. Press 'h' to select HAND region (2 cards)
2. Press 'b' to select BOARD region (5 cards)  
3. Press 'd' to start detection
4. Press 'r' to restart
5. Press 'q' to quit
        """
        
        instructions_label = tk.Label(instructions_frame, text=instructions.strip(), 
                                    font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1')
        instructions_label.pack()
        
        # Control buttons
        controls_frame = tk.Frame(self.window, bg='#2c3e50')
        controls_frame.pack(pady=20)
        
        self.hand_btn = tk.Button(controls_frame, text="Select HAND (h)", 
                                 command=self.select_hand, font=('Arial', 12),
                                 bg='#27ae60', fg='white', padx=20, pady=10)
        self.hand_btn.pack(side='left', padx=10)
        
        self.board_btn = tk.Button(controls_frame, text="Select BOARD (b)", 
                                  command=self.select_board, font=('Arial', 12),
                                  bg='#e74c3c', fg='white', padx=20, pady=10)
        self.board_btn.pack(side='left', padx=10)
        
        self.detect_btn = tk.Button(controls_frame, text="Start Detection (d)", 
                                   command=self.start_detection, font=('Arial', 12),
                                   bg='#3498db', fg='white', padx=20, pady=10)
        self.detect_btn.pack(side='left', padx=10)
        
        self.restart_btn = tk.Button(controls_frame, text="Restart (r)", 
                                    command=self.restart, font=('Arial', 12),
                                    bg='#f39c12', fg='white', padx=20, pady=10)
        self.restart_btn.pack(side='left', padx=10)
        
        # Results frame
        results_frame = tk.Frame(self.window, bg='#2c3e50')
        results_frame.pack(pady=20, fill='both', expand=True)
        
        # Hand results
        hand_frame = tk.Frame(results_frame, bg='#2c3e50')
        hand_frame.pack(fill='x', pady=10)
        
        hand_label = tk.Label(hand_frame, text="HAND CARDS:", 
                             font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#27ae60')
        hand_label.pack(side='left')
        
        self.hand_result = tk.Label(hand_frame, text="Not detected", 
                                   font=('Arial', 12), bg='#2c3e50', fg='white')
        self.hand_result.pack(side='left', padx=10)
        
        # Board results
        board_frame = tk.Frame(results_frame, bg='#2c3e50')
        board_frame.pack(fill='x', pady=10)
        
        board_label = tk.Label(board_frame, text="BOARD CARDS:", 
                              font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#e74c3c')
        board_label.pack(side='left')
        
        self.board_result = tk.Label(board_frame, text="Not detected", 
                                   font=('Arial', 12), bg='#2c3e50', fg='white')
        self.board_result.pack(side='left', padx=10)
        
        # Status
        status_frame = tk.Frame(self.window, bg='#2c3e50')
        status_frame.pack(side='bottom', fill='x', padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready - Press 'h' to select hand", 
                                   font=('Arial', 10), bg='#2c3e50', fg='#ecf0f1')
        self.status_label.pack()
        
        # Bind keyboard events
        self.window.bind('<KeyPress-h>', lambda e: self.select_hand())
        self.window.bind('<KeyPress-b>', lambda e: self.select_board())
        self.window.bind('<KeyPress-d>', lambda e: self.start_detection())
        self.window.bind('<KeyPress-r>', lambda e: self.restart())
        self.window.bind('<KeyPress-q>', lambda e: self.close())
        self.window.focus_set()
        
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
        # Create preview window
        preview = screen.copy()
        
        # Draw hand region
        if self.hand_region:
            x, y, w, h = self.hand_region
            cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(preview, "HAND", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw board region
        if self.board_region:
            x, y, w, h = self.board_region
            cv2.rectangle(preview, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(preview, "BOARD", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Show preview
        cv2.imshow("Poker Vision Preview", preview)
        cv2.waitKey(1)
    
    def select_hand(self):
        """Select hand region"""
        self.status_label.config(text="Click and drag to select HAND region (2 cards)")
        self.start_region_selection("hand")
        
    def select_board(self):
        """Select board region"""
        self.status_label.config(text="Click and drag to select BOARD region (5 cards)")
        self.start_region_selection("board")
        
    def start_region_selection(self, region_type):
        """Start region selection process"""
        # Create selection window
        selection_window = tk.Toplevel(self.window)
        selection_window.title(f"Select {region_type.upper()} Region")
        selection_window.geometry("400x200")
        selection_window.configure(bg='#2c3e50')
        
        # Instructions
        instructions = f"""
Click and drag to select your {region_type.upper()} region.
This should contain your {region_type} cards.

Press Enter when done, or Escape to cancel.
        """
        
        tk.Label(selection_window, text=instructions.strip(), 
                font=('Arial', 12), bg='#2c3e50', fg='white').pack(pady=20)
        
        # Start OpenCV selection
        def start_selection():
            region = self.select_region_with_opencv(region_type)
            if region:
                if region_type == "hand":
                    self.hand_region = region
                    self.hand_btn.config(bg='#2ecc71', text="HAND Selected ✓")
                    self.status_label.config(text="Hand region selected. Press 'b' to select board.")
                else:
                    self.board_region = region
                    self.board_btn.config(bg='#e67e22', text="BOARD Selected ✓")
                    self.status_label.config(text="Board region selected. Press 'd' to start detection.")
            selection_window.destroy()
        
        # Start selection in thread
        selection_thread = threading.Thread(target=start_selection, daemon=True)
        selection_thread.start()
        
    def select_region_with_opencv(self, region_type):
        """Use OpenCV to select region"""
        try:
            # Capture screen
            screen = self.capture_screen()
            
            # Create selection window
            cv2.namedWindow(f"Select {region_type.upper()}", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(f"Select {region_type.upper()}", cv2.WND_PROP_TOPMOST, 1)
            
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
            
            cv2.setMouseCallback(f"Select {region_type.upper()}", mouse_callback)
            
            # Selection loop
            while True:
                display = screen.copy()
                
                if start_point and end_point:
                    cv2.rectangle(display, start_point, end_point, (0, 255, 255), 2)
                
                cv2.putText(display, f"Select {region_type.upper()} region", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display, "Press Enter to confirm, Escape to cancel", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                cv2.imshow(f"Select {region_type.upper()}", display)
                
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
    
    def start_detection(self):
        """Start card detection"""
        if not self.hand_region or not self.board_region:
            messagebox.showwarning("Warning", "Please select both hand and board regions first!")
            return
        
        self.detection_active = True
        self.detect_btn.config(bg='#e74c3c', text="Detection Active ✓")
        self.status_label.config(text="Detection active - detecting cards...")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
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
                
                with open("state.json", "w") as f:
                    json.dump(result, f, indent=2)
                
                time.sleep(0.5)  # Update every 500ms
                
            except Exception as e:
                print(f"Detection error: {e}")
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
            print(f"Contour detection failed: {e}")
        
        # Method 2: Grid detection
        try:
            grid_cards = self.detect_cards_grid(roi, max_cards, region_name)
            all_cards.extend(grid_cards)
        except Exception as e:
            print(f"Grid detection failed: {e}")
        
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
        hand_str = ", ".join(hand_cards) if hand_cards else "None detected"
        board_str = ", ".join(board_cards) if board_cards else "None detected"
        
        self.hand_result.config(text=hand_str)
        self.board_result.config(text=board_str)
        
        self.status_label.config(text=f"Detected: Hand={len(hand_cards)}, Board={len(board_cards)}")
    
    def restart(self):
        """Restart the detection process"""
        self.detection_active = False
        self.hand_region = None
        self.board_region = None
        
        # Reset buttons
        self.hand_btn.config(bg='#27ae60', text="Select HAND (h)")
        self.board_btn.config(bg='#e74c3c', text="Select BOARD (b)")
        self.detect_btn.config(bg='#3498db', text="Start Detection (d)")
        
        # Reset results
        self.hand_result.config(text="Not detected")
        self.board_result.config(text="Not detected")
        
        self.status_label.config(text="Restarted - Press 'h' to select hand")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
    
    def close(self):
        """Close the detector window"""
        self.detection_active = False
        cv2.destroyAllWindows()
        self.window.destroy()
