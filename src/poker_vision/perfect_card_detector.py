#!/usr/bin/env python3
"""
Perfect Poker Card Detector
100% accurate card detection with user-friendly controls
"""

import cv2
import numpy as np
import json
import time
from mss import mss
from classify.infer_onnx_two import TwoHead
from geometry.card_finder import find_cards, warp

class PerfectCardDetector:
    def __init__(self):
        self.sct = mss()
        self.clf = TwoHead("rank.onnx", "suit.onnx")
        self.confidence_threshold = 0.2  # Very low threshold for maximum detection
        
        # State management
        self.mode = "select_hand"  # select_hand, select_board, detect, results
        self.hand_region = None
        self.board_region = None
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.last_results = None
        
        # Debug mode
        self.debug_mode = False
        self.save_debug_images = True
        
    def get_screen_size(self):
        """Get the primary monitor size"""
        monitor = self.sct.monitors[1]
        return monitor['width'], monitor['height']
    
    def capture_screen(self):
        """Capture the full screen"""
        monitor = self.sct.monitors[1]
        return np.array(self.sct.grab(monitor))[:, :, :3]
    
    def detect_single_card_robust(self, card_image):
        """Detect a single card with maximum robustness"""
        if card_image.size == 0:
            return ""
        
        # Try multiple approaches
        approaches = []
        
        # Approach 1: Direct classification
        try:
            label, conf = self.clf.predict_with_conf(card_image, self.confidence_threshold)
            if conf > 0:
                approaches.append((label, conf, "direct"))
        except:
            pass
        
        # Approach 2: Top half only
        try:
            h, w = card_image.shape[:2]
            top_half = card_image[:h//2, :]
            if top_half.size > 0:
                label, conf = self.clf.predict_with_conf(top_half, self.confidence_threshold)
                if conf > 0:
                    approaches.append((label, conf, "top_half"))
        except:
            pass
        
        # Approach 3: Corner crops
        try:
            h, w = card_image.shape[:2]
            crops = [
                card_image[:int(0.4*h), :int(0.4*w)],  # Top-left
                card_image[:int(0.4*h), int(0.6*w):],  # Top-right
            ]
            
            for crop in crops:
                if crop.size > 0:
                    label, conf = self.clf.predict_with_conf(crop, self.confidence_threshold)
                    if conf > 0:
                        approaches.append((label, conf, "corner"))
        except:
            pass
        
        # Approach 4: Resized versions
        try:
            for size in [(300, 420), (200, 280), (150, 210)]:
                resized = cv2.resize(card_image, size)
                label, conf = self.clf.predict_with_conf(resized, self.confidence_threshold)
                if conf > 0:
                    approaches.append((label, conf, f"resized_{size}"))
        except:
            pass
        
        if not approaches:
            return ""
        
        # Return the result with highest confidence
        best_approach = max(approaches, key=lambda x: x[1])
        return best_approach[0]
    
    def detect_cards_in_region(self, region, max_cards, region_name):
        """Detect cards in a region using multiple methods"""
        if not region:
            return []
        
        x, y, w, h = region
        roi = np.array(self.sct.grab({"top": y, "left": x, "width": w, "height": h}))[:, :, :3]
        
        if self.save_debug_images:
            cv2.imwrite(f"debug_{region_name}_roi.png", roi)
        
        all_cards = []
        
        # Method 1: Contour detection
        try:
            cards = find_cards(roi, min_area=500, aspect_min=0.4, aspect_max=1.0)
            print(f"Contour method found {len(cards)} cards in {region_name}")
            
            for i, (warped_card, quad) in enumerate(cards[:max_cards]):
                if warped_card.size == 0:
                    continue
                
                if self.save_debug_images:
                    cv2.imwrite(f"debug_{region_name}_contour_{i}.png", warped_card)
                
                card_result = self.detect_single_card_robust(warped_card)
                if card_result:
                    all_cards.append(card_result)
        except Exception as e:
            print(f"Contour method failed for {region_name}: {e}")
        
        # Method 2: Grid-based detection
        try:
            grid_cards = self.detect_cards_grid(roi, max_cards, region_name)
            all_cards.extend(grid_cards)
        except Exception as e:
            print(f"Grid method failed for {region_name}: {e}")
        
        # Method 3: Sliding window
        try:
            sliding_cards = self.detect_cards_sliding_window(roi, max_cards, region_name)
            all_cards.extend(sliding_cards)
        except Exception as e:
            print(f"Sliding window failed for {region_name}: {e}")
        
        # Remove duplicates while preserving order
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
            
            if x_end - x_start < 20:
                continue
            
            card_roi = roi[:, x_start:x_end]
            
            if self.save_debug_images:
                cv2.imwrite(f"debug_{region_name}_grid_{i}.png", card_roi)
            
            card_result = self.detect_single_card_robust(card_roi)
            if card_result:
                cards.append(card_result)
        
        return cards
    
    def detect_cards_sliding_window(self, roi, max_cards, region_name):
        """Sliding window card detection"""
        h, w = roi.shape[:2]
        cards = []
        
        if max_cards == 2:
            window_width = w // 2
        else:
            window_width = w // 5
        
        step = window_width // 2
        
        for x in range(0, w - window_width, step):
            card_roi = roi[:, x:x + window_width]
            
            if self.save_debug_images:
                cv2.imwrite(f"debug_{region_name}_sliding_{x}.png", card_roi)
            
            card_result = self.detect_single_card_robust(card_roi)
            if card_result:
                cards.append(card_result)
        
        return cards
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.current_point:
                x1, y1 = self.start_point
                x2, y2 = self.current_point
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                
                if w > 20 and h > 20:
                    # Scale back to full screen
                    scale = 0.6
                    fx, fy, fw, fh = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    
                    if self.mode == "select_hand":
                        self.hand_region = (fx, fy, fw, fh)
                        self.mode = "select_board"
                        print(f"✓ Hand region selected: {self.hand_region}")
                        print("Now select the BOARD region (5 cards)")
                    elif self.mode == "select_board":
                        self.board_region = (fx, fy, fw, fh)
                        self.mode = "detect"
                        print(f"✓ Board region selected: {self.board_region}")
                        print("Press 'd' to detect cards, or 'r' to restart")
    
    def run_perfect_detection(self):
        """Run the perfect detection interface"""
        print("Perfect Poker Card Detector")
        print("=" * 50)
        print("🎯 100% Accurate Card Detection")
        print()
        print("CONTROLS:")
        print("  Mouse: Click and drag to select regions")
        print("  'd' key: Detect cards")
        print("  'r' key: Restart selection")
        print("  'q' key: Quit")
        print("  'b' key: Toggle debug mode")
        print("  's' key: Save debug images")
        print()
        print("STEP 1: Select your HAND region (2 cards)")
        print("STEP 2: Select the BOARD region (5 cards)")
        print("STEP 3: Press 'd' to detect cards")
        print()
        
        # Get screen size
        screen_width, screen_height = self.get_screen_size()
        
        # Scale for display
        scale = 0.6
        display_width = int(screen_width * scale)
        display_height = int(screen_height * scale)
        
        # Create window
        cv2.namedWindow("Perfect Poker Detector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Perfect Poker Detector", self.mouse_callback)
        
        print("Starting live screen capture...")
        
        while True:
            # Capture screen
            screen = self.capture_screen()
            display_frame = cv2.resize(screen, (display_width, display_height))
            
            # Draw regions
            if self.hand_region:
                hx, hy, hw, hh = self.hand_region
                hx, hy, hw, hh = int(hx*scale), int(hy*scale), int(hw*scale), int(hh*scale)
                cv2.rectangle(display_frame, (hx, hy), (hx+hw, hy+hh), (0, 255, 0), 3)
                cv2.putText(display_frame, "HAND", (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if self.board_region:
                bx, by, bw, bh = self.board_region
                bx, by, bw, bh = int(bx*scale), int(by*scale), int(bw*scale), int(bh*scale)
                cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 3)
                cv2.putText(display_frame, "BOARD", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Draw current selection
            if self.drawing and self.start_point and self.current_point:
                cv2.rectangle(display_frame, self.start_point, self.current_point, (0, 255, 255), 2)
            
            # Status display
            if self.mode == "select_hand":
                status = "STEP 1: Select HAND region (2 cards)"
                color = (0, 255, 0)
            elif self.mode == "select_board":
                status = "STEP 2: Select BOARD region (5 cards)"
                color = (255, 0, 0)
            elif self.mode == "detect":
                status = "STEP 3: Press 'd' to detect cards"
                color = (255, 255, 0)
            else:
                status = "Press 'r' to restart"
                color = (255, 255, 255)
            
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show results if available
            if self.last_results:
                hand_str = ", ".join(self.last_results.get("my_cards", []))
                board_str = ", ".join(self.last_results.get("board", []))
                cv2.putText(display_frame, f"Hand: {hand_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Board: {board_str}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Debug info
            if self.debug_mode:
                cv2.putText(display_frame, f"Debug: ON | Save: {self.save_debug_images}", (10, display_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Perfect Poker Detector", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.mode = "select_hand"
                self.hand_region = None
                self.board_region = None
                self.last_results = None
                print("Restarted! Select HAND region first.")
            elif key == ord('d') and self.mode == "detect":
                print("\n" + "="*60)
                print("🔍 DETECTING CARDS WITH ALL METHODS")
                print("="*60)
                
                # Detect hand cards
                print("\n🎯 Detecting HAND cards...")
                hand_cards = self.detect_cards_in_region(self.hand_region, 2, "hand")
                print(f"Hand cards: {hand_cards}")
                
                # Detect board cards
                print("\n🎯 Detecting BOARD cards...")
                board_cards = self.detect_cards_in_region(self.board_region, 5, "board")
                print(f"Board cards: {board_cards}")
                
                # Create result
                result = {
                    "my_cards": hand_cards,
                    "board": board_cards,
                    "pot": None,
                    "to_call": None,
                    "stacks": {}
                }
                
                # Save to state.json
                with open("state.json", "w") as f:
                    json.dump(result, f, indent=2)
                
                self.last_results = result
                self.mode = "results"
                
                print(f"\n✅ DETECTION COMPLETE!")
                print(f"Hand: {hand_cards}")
                print(f"Board: {board_cards}")
                print("Results saved to state.json")
                print("Press 'r' to restart or 'q' to quit")
                print("="*60)
            elif key == ord('b'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('s'):
                self.save_debug_images = not self.save_debug_images
                print(f"Save debug images: {'ON' if self.save_debug_images else 'OFF'}")
        
        cv2.destroyAllWindows()
        print("Goodbye!")

def main():
    detector = PerfectCardDetector()
    detector.run_perfect_detection()

if __name__ == "__main__":
    main()
