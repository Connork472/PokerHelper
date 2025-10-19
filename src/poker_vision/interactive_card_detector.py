#!/usr/bin/env python3
"""
Interactive Poker Card Detector
Shows live screen stream, allows region selection, then detects cards
"""

import cv2
import numpy as np
import json
import time
from mss import mss
from classify.infer_onnx_two import TwoHead

class InteractiveCardDetector:
    def __init__(self):
        self.sct = mss()
        self.clf = TwoHead("rank.onnx", "suit.onnx")
        self.screen_width = 1920  # Default, will be updated
        self.screen_height = 1080
        self.hand_region = None
        self.board_region = None
        self.detection_mode = False
        self.preview_scale = 0.5  # Scale down for preview
        
    def get_screen_size(self):
        """Get the primary monitor size"""
        monitor = self.sct.monitors[1]  # Primary monitor
        self.screen_width = monitor['width']
        self.screen_height = monitor['height']
        return self.screen_width, self.screen_height
    
    def capture_screen(self):
        """Capture the full screen"""
        monitor = self.sct.monitors[1]
        return np.array(self.sct.grab(monitor))[:, :, :3]
    
    def split_into_slots(self, roi_image, num_slots):
        """Split ROI into equal horizontal slots"""
        h, w = roi_image.shape[:2]
        slot_width = w // num_slots
        slots = []
        
        for i in range(num_slots):
            x_start = i * slot_width
            x_end = (i + 1) * slot_width if i < num_slots - 1 else w
            slot = roi_image[:, x_start:x_end]
            slots.append(slot)
        
        return slots
    
    def detect_cards_in_slots(self, slots, confidence_threshold=0.55):
        """Detect cards in each slot using the classifier"""
        results = []
        for slot in slots:
            if slot.size == 0:
                results.append("")
                continue
                
            label, conf = self.clf.predict_with_conf(slot, confidence_threshold)
            results.append(label if conf > 0 else "")
        
        return results
    
    def draw_region_overlay(self, frame, region, color, label):
        """Draw region overlay on frame"""
        if region is None:
            return frame
        
        x, y, w, h = region
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return overlay
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if hasattr(self, 'start_point') and hasattr(self, 'current_point'):
                # Calculate region
                x1, y1 = self.start_point
                x2, y2 = self.current_point
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                
                if w > 10 and h > 10:  # Minimum size
                    if not self.hand_region:
                        self.hand_region = (x, y, w, h)
                        print(f"Hand region selected: {self.hand_region}")
                    elif not self.board_region:
                        self.board_region = (x, y, w, h)
                        print(f"Board region selected: {self.board_region}")
                        print("Both regions selected! Press 'd' to detect cards.")
    
    def run_interactive_mode(self):
        """Run the interactive region selection mode"""
        print("Interactive Poker Card Detector")
        print("=" * 40)
        print("Instructions:")
        print("1. Click and drag to select your HAND region (2 cards)")
        print("2. Click and drag to select the BOARD region (5 cards)")
        print("3. Press 'd' to detect cards")
        print("4. Press 'q' to quit")
        print("5. Press 'r' to reset regions")
        print()
        
        # Get screen size
        self.get_screen_size()
        
        # Create window
        cv2.namedWindow("Poker Card Detector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Poker Card Detector", self.mouse_callback)
        
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        while True:
            # Capture screen
            screen = self.capture_screen()
            
            # Scale down for display
            display_height = int(self.screen_height * self.preview_scale)
            display_width = int(self.screen_width * self.preview_scale)
            display_frame = cv2.resize(screen, (display_width, display_height))
            
            # Draw regions
            if self.hand_region:
                # Scale region coordinates for display
                hx, hy, hw, hh = self.hand_region
                hx, hy, hw, hh = int(hx * self.preview_scale), int(hy * self.preview_scale), int(hw * self.preview_scale), int(hh * self.preview_scale)
                display_frame = self.draw_region_overlay(display_frame, (hx, hy, hw, hh), (0, 255, 0), "HAND")
            
            if self.board_region:
                # Scale region coordinates for display
                bx, by, bw, bh = self.board_region
                bx, by, bw, bh = int(bx * self.preview_scale), int(by * self.preview_scale), int(bw * self.preview_scale), int(bh * self.preview_scale)
                display_frame = self.draw_region_overlay(display_frame, (bx, by, bw, bh), (255, 0, 0), "BOARD")
            
            # Draw current selection
            if self.drawing and hasattr(self, 'start_point') and hasattr(self, 'current_point'):
                x1, y1 = self.start_point
                x2, y2 = self.current_point
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add status text
            status = "Select HAND region" if not self.hand_region else "Select BOARD region" if not self.board_region else "Press 'd' to detect"
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Poker Card Detector", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.hand_region = None
                self.board_region = None
                print("Regions reset")
            elif key == ord('d') and self.hand_region and self.board_region:
                self.detect_cards()
        
        cv2.destroyAllWindows()
    
    def detect_cards(self):
        """Detect cards in the selected regions"""
        print("\nDetecting cards...")
        
        # Scale regions back to full screen coordinates
        scale_factor = 1.0 / self.preview_scale
        
        # Process hand region (2 slots)
        if self.hand_region:
            hx, hy, hw, hh = self.hand_region
            # Scale back to full screen
            hx, hy, hw, hh = int(hx * scale_factor), int(hy * scale_factor), int(hw * scale_factor), int(hh * scale_factor)
            
            hand_roi = self.capture_region(hx, hy, hw, hh)
            hand_slots = self.split_into_slots(hand_roi, 2)
            hand_results = self.detect_cards_in_slots(hand_slots)
            hand_cards = [card for card in hand_results if card]
            print(f"Hand cards: {hand_cards}")
        else:
            hand_cards = []
        
        # Process board region (5 slots)
        if self.board_region:
            bx, by, bw, bh = self.board_region
            # Scale back to full screen
            bx, by, bw, bh = int(bx * scale_factor), int(by * scale_factor), int(bw * scale_factor), int(bh * scale_factor)
            
            board_roi = self.capture_region(bx, by, bw, bh)
            board_slots = self.split_into_slots(board_roi, 5)
            board_results = self.detect_cards_in_slots(board_slots)
            board_cards = [card for card in board_results if card]
            print(f"Board cards: {board_cards}")
        else:
            board_cards = []
        
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
        
        print(f"\nDetection complete!")
        print(f"Hand: {hand_cards}")
        print(f"Board: {board_cards}")
        print("Results saved to state.json")
        
        return result
    
    def capture_region(self, x, y, w, h):
        """Capture a specific region of the screen"""
        monitor = {
            "top": y,
            "left": x,
            "width": w,
            "height": h
        }
        return np.array(self.sct.grab(monitor))[:, :, :3]

def main():
    detector = InteractiveCardDetector()
    detector.run_interactive_mode()

if __name__ == "__main__":
    main()
