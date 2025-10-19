#!/usr/bin/env python3
"""
Robust Poker Card Detector
Uses multiple detection methods for accurate card recognition
"""

import cv2
import numpy as np
import json
from mss import mss
from classify.infer_onnx_two import TwoHead
from geometry.card_finder import find_cards, warp

class RobustCardDetector:
    def __init__(self):
        self.sct = mss()
        self.clf = TwoHead("rank.onnx", "suit.onnx")
        self.confidence_threshold = 0.4  # Lower threshold for better detection
        
    def detect_cards_contour_method(self, roi_image, max_cards=5):
        """Use contour detection to find individual cards"""
        try:
            cards = find_cards(roi_image, min_area=1000, aspect_min=0.5, aspect_max=0.9)
            results = []
            
            for i, (warped_card, quad) in enumerate(cards[:max_cards]):
                if warped_card.size == 0:
                    continue
                    
                # Try both corners and pick the best
                label, conf = self.clf.predict_with_conf(warped_card, self.confidence_threshold)
                if conf > 0:
                    results.append((label, conf, warped_card))
            
            # Sort by confidence and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return [r[0] for r in results]
            
        except Exception as e:
            print(f"Contour method error: {e}")
            return []
    
    def detect_cards_sliding_window(self, roi_image, max_cards=5):
        """Use sliding window approach to find cards"""
        try:
            h, w = roi_image.shape[:2]
            results = []
            
            # Calculate card dimensions based on ROI
            if max_cards == 2:  # Hand
                card_width = w // 2
                card_height = h
            else:  # Board
                card_width = w // 5
                card_height = h
            
            # Slide across the image
            for i in range(max_cards):
                x_start = i * card_width
                x_end = min((i + 1) * card_width, w)
                
                if x_end - x_start < 20:  # Too small
                    continue
                    
                card_roi = roi_image[:, x_start:x_end]
                
                # Try different crops within the card area
                card_crops = self.get_card_crops(card_roi)
                
                best_label = ""
                best_conf = 0
                
                for crop in card_crops:
                    if crop.size == 0:
                        continue
                    label, conf = self.clf.predict_with_conf(crop, self.confidence_threshold)
                    if conf > best_conf:
                        best_label = label
                        best_conf = conf
                
                if best_conf > 0:
                    results.append((best_label, best_conf))
            
            # Sort by confidence
            results.sort(key=lambda x: x[1], reverse=True)
            return [r[0] for r in results]
            
        except Exception as e:
            print(f"Sliding window error: {e}")
            return []
    
    def get_card_crops(self, card_roi):
        """Get multiple crops from a card region to improve detection"""
        h, w = card_roi.shape[:2]
        crops = []
        
        # Full card
        crops.append(card_roi)
        
        # Top half (where rank/suit usually are)
        crops.append(card_roi[:h//2, :])
        
        # Top-left corner
        crops.append(card_roi[:int(0.4*h), :int(0.4*w)])
        
        # Top-right corner  
        crops.append(card_roi[:int(0.4*h), int(0.6*w):])
        
        # Center area
        center_h, center_w = h//4, w//4
        crops.append(card_roi[center_h:3*center_h, center_w:3*center_w])
        
        return crops
    
    def detect_cards_hybrid(self, roi_image, max_cards=5):
        """Combine multiple detection methods for best results"""
        methods = [
            ("contour", self.detect_cards_contour_method),
            ("sliding", self.detect_cards_sliding_window)
        ]
        
        all_results = []
        
        for method_name, method_func in methods:
            try:
                results = method_func(roi_image, max_cards)
                print(f"{method_name} method found: {results}")
                all_results.extend([(r, method_name) for r in results])
            except Exception as e:
                print(f"{method_name} method failed: {e}")
        
        # Remove duplicates and return unique cards
        unique_cards = []
        seen = set()
        
        for card, method in all_results:
            if card and card not in seen:
                unique_cards.append(card)
                seen.add(card)
        
        return unique_cards[:max_cards]
    
    def run_interactive_detection(self):
        """Run the interactive detection interface"""
        print("Robust Poker Card Detector")
        print("=" * 40)
        print("Instructions:")
        print("1. Click and drag to select your HAND region (2 cards)")
        print("2. Click and drag to select the BOARD region (5 cards)")
        print("3. Press 'd' to detect cards using multiple methods")
        print("4. Press 'q' to quit")
        print("5. Press 'r' to reset regions")
        print()
        
        # Get screen size
        monitor = self.sct.monitors[1]
        screen_width = monitor['width']
        screen_height = monitor['height']
        
        # Scale for display
        scale = 0.6
        display_width = int(screen_width * scale)
        display_height = int(screen_height * scale)
        
        # Region selection variables
        hand_region = None
        board_region = None
        drawing = False
        start_point = None
        current_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, current_point, hand_region, board_region
            
            if event == cv2.EVENT_LBUTTONDOWN:
                start_point = (x, y)
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                current_point = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if start_point and current_point:
                    x1, y1 = start_point
                    x2, y2 = current_point
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    
                    if w > 20 and h > 20:
                        # Scale back to full screen
                        fx, fy, fw, fh = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                        
                        if not hand_region:
                            hand_region = (fx, fy, fw, fh)
                            print(f"✓ Hand region selected: {hand_region}")
                        elif not board_region:
                            board_region = (fx, fy, fw, fh)
                            print(f"✓ Board region selected: {board_region}")
                            print("✓ Both regions selected! Press 'd' to detect cards.")
        
        # Create window
        cv2.namedWindow("Robust Poker Detector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Robust Poker Detector", mouse_callback)
        
        print("Starting live screen capture...")
        
        while True:
            # Capture screen
            screen = np.array(self.sct.grab(monitor))[:, :, :3]
            display_frame = cv2.resize(screen, (display_width, display_height))
            
            # Draw regions
            if hand_region:
                hx, hy, hw, hh = hand_region
                hx, hy, hw, hh = int(hx*scale), int(hy*scale), int(hw*scale), int(hh*scale)
                cv2.rectangle(display_frame, (hx, hy), (hx+hw, hy+hh), (0, 255, 0), 2)
                cv2.putText(display_frame, "HAND", (hx, hy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if board_region:
                bx, by, bw, bh = board_region
                bx, by, bw, bh = int(bx*scale), int(by*scale), int(bw*scale), int(bh*scale)
                cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
                cv2.putText(display_frame, "BOARD", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw current selection
            if drawing and start_point and current_point:
                cv2.rectangle(display_frame, start_point, current_point, (0, 255, 255), 2)
            
            # Status
            if not hand_region:
                status = "Select HAND region (2 cards)"
            elif not board_region:
                status = "Select BOARD region (5 cards)"
            else:
                status = "Press 'd' to detect cards"
            
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, "Robust detection with multiple methods", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Robust Poker Detector", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                hand_region = None
                board_region = None
                print("Regions reset")
            elif key == ord('d') and hand_region and board_region:
                print("\n" + "="*50)
                print("DETECTING CARDS WITH MULTIPLE METHODS")
                print("="*50)
                
                # Detect hand cards
                print("\nDetecting HAND cards...")
                hx, hy, hw, hh = hand_region
                hand_roi = np.array(self.sct.grab({"top": hy, "left": hx, "width": hw, "height": hh}))[:, :, :3]
                hand_cards = self.detect_cards_hybrid(hand_roi, 2)
                print(f"Hand cards detected: {hand_cards}")
                
                # Detect board cards
                print("\nDetecting BOARD cards...")
                bx, by, bw, bh = board_region
                board_roi = np.array(self.sct.grab({"top": by, "left": bx, "width": bw, "height": bh}))[:, :, :3]
                board_cards = self.detect_cards_hybrid(board_roi, 5)
                print(f"Board cards detected: {board_cards}")
                
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
                
                print(f"\n✓ DETECTION COMPLETE!")
                print(f"Hand: {hand_cards}")
                print(f"Board: {board_cards}")
                print("Results saved to state.json")
                print("="*50)
        
        cv2.destroyAllWindows()

def main():
    detector = RobustCardDetector()
    detector.run_interactive_detection()

if __name__ == "__main__":
    main()
