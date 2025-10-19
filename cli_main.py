#!/usr/bin/env python3
"""
PokerHelper - Command Line Interface
Clean CLI for poker card detection and win probability calculation
"""

import sys
import os
import json
import time
import cv2
import numpy as np
from collections import deque
from mss import mss

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'poker_vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'simulator'))

# Try to import onnx-based classifier, fallback to alternative if not available
try:
    from classify.infer_onnx_two import TwoHead
    print("Using ONNX-based classifier")
except ImportError:
    print("ONNX runtime not available, using fallback classifier")
    from classify.infer_onnx_fallback import TwoHead

try:
    from geometry.card_finder import find_cards
    from poker_cli_session import tokenize_cards, to_treys, simulate, kelly_even_money
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install opencv-python mss numpy treys")
    sys.exit(1)

class PokerHelperCLI:
    def __init__(self):
        self.sct = mss()
        self.clf = TwoHead("config/rank.onnx", "config/suit.onnx")
        
        # Detection state
        self.hand_region = None
        self.board_region = None
        self.detection_active = False
        
        # Temporal smoothing
        self.detection_history = deque(maxlen=5)
        self.stable_threshold = 3
        self.last_stable_detection = None
        
    def main_menu(self):
        """Display main menu and handle user choice"""
        while True:
            print("\n" + "="*60)
            print("🎯 PokerHelper - Card Detection & Win Probability")
            print("="*60)
            print("1. 🎯 Poker Vision Detection (Automatic)")
            print("2. 🎲 Manual Simulator (Manual Input)")
            print("3. ⚙️  Settings & Configuration")
            print("4. 📊 View Current State")
            print("5. ❌ Exit")
            print("="*60)
            
            choice = input("Select an option (1-5): ").strip()
            
            if choice == "1":
                self.poker_vision_mode()
            elif choice == "2":
                self.manual_simulator_mode()
            elif choice == "3":
                self.settings_mode()
            elif choice == "4":
                self.view_state()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-5.")
    
    def poker_vision_mode(self):
        """Poker vision detection mode"""
        print("\n🎯 Poker Vision Detection Mode")
        print("="*40)
        print("Instructions:")
        print("1. Press 'h' to select HAND region (2 cards)")
        print("2. Press 'b' to select BOARD region (5 cards)")
        print("3. Press 'd' to start detection")
        print("4. Press 'r' to restart")
        print("5. Press 'q' to quit")
        print("\nStarting screen capture...")
        
        # Start detection interface
        self.detection_interface()
    
    def detection_interface(self):
        """Interactive detection interface"""
        cv2.namedWindow("Poker Vision Detector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Poker Vision Detector", self.mouse_callback)
        
        print("Screen capture started. Use mouse to select regions.")
        
        while True:
            # Capture screen
            screen = self.capture_screen()
            display_frame = screen.copy()
            
            # Draw regions
            if self.hand_region:
                x, y, w, h = self.hand_region
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(display_frame, "HAND", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if self.board_region:
                x, y, w, h = self.board_region
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                cv2.putText(display_frame, "BOARD", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Status text
            if not self.hand_region:
                status = "Press 'h' to select HAND region"
            elif not self.board_region:
                status = "Press 'b' to select BOARD region"
            elif not self.detection_active:
                status = "Press 'd' to start detection"
            else:
                status = "Detection active - Press 'r' to restart"
            
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Poker Vision Detector", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.select_region("hand")
            elif key == ord('b'):
                self.select_region("board")
            elif key == ord('d') and self.hand_region and self.board_region:
                self.start_detection()
            elif key == ord('r'):
                self.restart_detection()
        
        cv2.destroyAllWindows()
    
    def select_region(self, region_type):
        """Select region using OpenCV"""
        print(f"Selecting {region_type} region... Click and drag to select area.")
        
        # Capture screen
        screen = self.capture_screen()
        
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
        
        cv2.namedWindow(f"Select {region_type.upper()}", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(f"Select {region_type.upper()}", mouse_callback)
        
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
                        region = (x, y, w, h)
                        if region_type == "hand":
                            self.hand_region = region
                            print(f"✓ Hand region selected: {region}")
                        else:
                            self.board_region = region
                            print(f"✓ Board region selected: {region}")
                        cv2.destroyAllWindows()
                        return
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                return
    
    def start_detection(self):
        """Start card detection"""
        print("Starting card detection...")
        self.detection_active = True
        
        # Detection loop
        while self.detection_active:
            try:
                # Detect hand cards
                hand_cards = self.detect_cards_in_region(self.hand_region, 2, "hand")
                
                # Detect board cards
                board_cards = self.detect_cards_in_region(self.board_region, 5, "board")
                
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
                
                # Display results
                hand_str = ", ".join(hand_cards) if hand_cards else "None"
                board_str = ", ".join(board_cards) if board_cards else "None"
                
                print(f"\rHand: {hand_str} | Board: {board_str}", end="", flush=True)
                
                time.sleep(0.5)  # Update every 500ms
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nDetection error: {e}")
                time.sleep(1)
    
    def detect_cards_in_region(self, region, max_cards, region_name):
        """Detect cards in a region"""
        if not region:
            return []
        
        x, y, w, h = region
        roi = np.array(self.sct.grab({"top": y, "left": x, "width": w, "height": h}))[:, :, :3]
        
        all_cards = []
        
        # Method 1: Contour detection
        try:
            cards = find_cards(roi, min_area=800, aspect_min=0.5, aspect_max=0.9)
            for warped_card, quad in cards[:max_cards]:
                if warped_card.size == 0:
                    continue
                
                label, conf = self.clf.predict_with_conf(warped_card, 0.3)
                if conf > 0:
                    all_cards.append(label)
        except Exception as e:
            pass
        
        # Method 2: Grid detection
        try:
            grid_cards = self.detect_cards_grid(roi, max_cards)
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
    
    def detect_cards_grid(self, roi, max_cards):
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
            label, conf = self.clf.predict_with_conf(card_roi, 0.3)
            if conf > 0:
                cards.append(label)
        
        return cards
    
    def restart_detection(self):
        """Restart detection"""
        self.detection_active = False
        self.hand_region = None
        self.board_region = None
        print("\nDetection restarted. Press 'h' to select hand region.")
    
    def manual_simulator_mode(self):
        """Manual simulator mode"""
        print("\n🎲 Manual Simulator Mode")
        print("="*40)
        
        while True:
            print("\nEnter your cards:")
            hand_input = input("Hand (2 cards, e.g., 'As Kh'): ").strip()
            
            if not hand_input:
                print("Please enter your hand cards.")
                continue
            
            try:
                hand_cards = tokenize_cards(hand_input)
                if len(hand_cards) != 2:
                    print("Please enter exactly 2 hand cards.")
                    continue
                break
            except Exception as e:
                print(f"Invalid hand format: {e}")
        
        while True:
            board_input = input("Board (0-5 cards, e.g., '7h 2d 2s'): ").strip()
            
            try:
                board_cards = tokenize_cards(board_input) if board_input else []
                if len(board_cards) > 5:
                    print("Board cannot have more than 5 cards.")
                    continue
                break
            except Exception as e:
                print(f"Invalid board format: {e}")
        
        # Get settings
        try:
            players = int(input("Number of players (2-10, default 6): ") or "6")
            trials = int(input("Number of trials (1000-100000, default 10000): ") or "10000")
        except ValueError:
            players = 6
            trials = 10000
        
        # Run simulation
        print(f"\nRunning simulation with {players} players, {trials:,} trials...")
        
        try:
            my_cards = to_treys(hand_cards)
            board_cards_treys = to_treys(board_cards)
            
            win_p, tie_p, equity = simulate(players, my_cards, board_cards_treys, trials)
            kelly = kelly_even_money(equity)
            
            # Calculate confidence interval
            import math
            var = equity * (1 - equity) / max(1, trials)
            moe = 1.96 * math.sqrt(var)
            
            # Display results
            print("\n" + "="*60)
            print("🎯 SIMULATION RESULTS")
            print("="*60)
            print(f"Hand: {' '.join(hand_cards)}")
            print(f"Board: {' '.join(board_cards) if board_cards else '(none)'}")
            print(f"Players: {players} | Trials: {trials:,}")
            print()
            print(f"Win%: {win_p*100:.2f}%")
            print(f"Tie%: {tie_p*100:.2f}%")
            print(f"Equity: {equity*100:.2f}%")
            print()
            print(f"Kelly (even-money): {kelly*100:.2f}% of bankroll")
            print(f"95% CI (equity): ±{moe*100:.2f}%")
            print("="*60)
            
            # Save to state.json
            state = {
                "my_cards": hand_cards,
                "board": board_cards,
                "pot": None,
                "to_call": None,
                "stacks": {},
                "equity": equity,
                "kelly": kelly
            }
            
            with open("output/state.json", "w") as f:
                json.dump(state, f, indent=2)
            
            print("Results saved to output/state.json")
            
        except Exception as e:
            print(f"Simulation error: {e}")
    
    def settings_mode(self):
        """Settings and configuration"""
        print("\n⚙️ Settings & Configuration")
        print("="*40)
        print("1. View current configuration")
        print("2. Reset configuration")
        print("3. Test models")
        print("4. Back to main menu")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            self.view_config()
        elif choice == "2":
            self.reset_config()
        elif choice == "3":
            self.test_models()
        elif choice == "4":
            return
        else:
            print("Invalid choice.")
    
    def view_config(self):
        """View current configuration"""
        print("\nCurrent Configuration:")
        print("-" * 30)
        
        # Check if models exist
        rank_model = "config/rank.onnx"
        suit_model = "config/suit.onnx"
        
        print(f"Rank model: {'✓ Found' if os.path.exists(rank_model) else '✗ Missing'}")
        print(f"Suit model: {'✓ Found' if os.path.exists(suit_model) else '✗ Missing'}")
        
        # Check state.json
        state_file = "output/state.json"
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            print(f"Current state: {state}")
        else:
            print("No current state file")
    
    def reset_config(self):
        """Reset configuration"""
        print("Resetting configuration...")
        
        # Remove state file
        if os.path.exists("output/state.json"):
            os.remove("output/state.json")
            print("✓ State file removed")
        
        # Clear debug images
        debug_dir = "output/debug"
        if os.path.exists(debug_dir):
            for file in os.listdir(debug_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(debug_dir, file))
            print("✓ Debug images cleared")
        
        print("Configuration reset complete.")
    
    def test_models(self):
        """Test model loading"""
        print("Testing models...")
        
        try:
            clf = TwoHead("config/rank.onnx", "config/suit.onnx")
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
    
    def view_state(self):
        """View current state"""
        state_file = "output/state.json"
        
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            
            print("\n📊 Current State:")
            print("-" * 30)
            print(f"Hand: {state.get('my_cards', 'None')}")
            print(f"Board: {state.get('board', 'None')}")
            if 'equity' in state:
                print(f"Equity: {state['equity']*100:.1f}%")
            if 'kelly' in state:
                print(f"Kelly: {state['kelly']*100:.1f}%")
        else:
            print("No current state file found.")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for region selection"""
        pass  # Handled in select_region method
    
    def capture_screen(self):
        """Capture the full screen"""
        monitor = self.sct.monitors[1]
        return np.array(self.sct.grab(monitor))[:, :, :3]

def main():
    """Main entry point"""
    print("🎯 PokerHelper - Command Line Interface")
    print("Loading...")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/debug", exist_ok=True)
    
    try:
        app = PokerHelperCLI()
        app.main_menu()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
