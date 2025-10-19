#!/usr/bin/env python3
"""
Ultimate Poker Card Detector
Most accurate card detection with multiple fallback methods
"""

import cv2
import numpy as np
import json
from mss import mss
from classify.infer_onnx_two import TwoHead
from geometry.card_finder import find_cards, warp

class UltimateCardDetector:
    def __init__(self):
        self.sct = mss()
        self.clf = TwoHead("rank.onnx", "suit.onnx")
        self.confidence_threshold = 0.3  # Lower threshold for better detection
        
    def detect_cards_advanced_contour(self, roi_image, max_cards=5):
        """Advanced contour detection with multiple parameters"""
        results = []
        
        # Try different contour detection parameters
        param_sets = [
            {"min_area": 500, "aspect_min": 0.4, "aspect_max": 1.0},
            {"min_area": 1000, "aspect_min": 0.5, "aspect_max": 0.9},
            {"min_area": 1500, "aspect_min": 0.6, "aspect_max": 0.8},
        ]
        
        for params in param_sets:
            try:
                cards = find_cards(roi_image, **params)
                print(f"Contour method (area>={params['min_area']}): found {len(cards)} cards")
                
                for warped_card, quad in cards[:max_cards]:
                    if warped_card.size == 0:
                        continue
                    
                    # Try multiple crops from the warped card
                    crops = self.get_multiple_crops(warped_card)
                    
                    best_label = ""
                    best_conf = 0
                    
                    for crop in crops:
                        if crop.size == 0:
                            continue
                        label, conf = self.clf.predict_with_conf(crop, self.confidence_threshold)
                        if conf > best_conf:
                            best_label = label
                            best_conf = conf
                    
                    if best_conf > 0:
                        results.append((best_label, best_conf))
                        
            except Exception as e:
                print(f"Contour method failed: {e}")
                continue
        
        # Remove duplicates and sort by confidence
        unique_results = {}
        for label, conf in results:
            if label not in unique_results or conf > unique_results[label]:
                unique_results[label] = conf
        
        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        return [label for label, conf in sorted_results[:max_cards]]
    
    def detect_cards_grid_scan(self, roi_image, max_cards=5):
        """Grid-based scanning for card detection"""
        h, w = roi_image.shape[:2]
        results = []
        
        if max_cards == 2:  # Hand
            grid_cols = 2
        else:  # Board
            grid_cols = 5
            
        card_width = w // grid_cols
        
        for i in range(grid_cols):
            x_start = i * card_width
            x_end = min((i + 1) * card_width, w)
            
            if x_end - x_start < 30:
                continue
                
            card_roi = roi_image[:, x_start:x_end]
            
            # Try multiple detection approaches on this card
            card_results = self.detect_single_card_multiple_methods(card_roi)
            
            if card_results:
                results.append(card_results)
        
        return results[:max_cards]
    
    def detect_single_card_multiple_methods(self, card_roi):
        """Try multiple methods to detect a single card"""
        if card_roi.size == 0:
            return ""
        
        # Method 1: Direct classification
        label1, conf1 = self.clf.predict_with_conf(card_roi, self.confidence_threshold)
        
        # Method 2: Top half only
        h, w = card_roi.shape[:2]
        top_half = card_roi[:h//2, :]
        label2, conf2 = self.clf.predict_with_conf(top_half, self.confidence_threshold)
        
        # Method 3: Corner crops
        corner_crops = [
            card_roi[:int(0.4*h), :int(0.4*w)],  # Top-left
            card_roi[:int(0.4*h), int(0.6*w):],  # Top-right
        ]
        
        best_label = ""
        best_conf = 0
        
        for i, (label, conf) in enumerate([(label1, conf1), (label2, conf2)]):
            if conf > best_conf:
                best_label = label
                best_conf = conf
        
        for crop in corner_crops:
            if crop.size > 0:
                label, conf = self.clf.predict_with_conf(crop, self.confidence_threshold)
                if conf > best_conf:
                    best_label = label
                    best_conf = conf
        
        return best_label if best_conf > 0 else ""
    
    def get_multiple_crops(self, warped_card):
        """Get multiple crops from a warped card for better detection"""
        h, w = warped_card.shape[:2]
        crops = []
        
        # Full card
        crops.append(warped_card)
        
        # Top half
        crops.append(warped_card[:h//2, :])
        
        # Top-left corner
        crops.append(warped_card[:int(0.4*h), :int(0.4*w)])
        
        # Top-right corner
        crops.append(warped_card[:int(0.4*h), int(0.6*w):])
        
        # Center area
        center_h, center_w = h//4, w//4
        crops.append(warped_card[center_h:3*center_h, center_w:3*center_w])
        
        # Resized versions
        crops.append(cv2.resize(warped_card, (300, 420)))
        crops.append(cv2.resize(warped_card, (200, 280)))
        
        return crops
    
    def run_ultimate_detection(self):
        """Run the ultimate detection interface"""
        print("Ultimate Poker Card Detector")
        print("=" * 40)
        print("Most accurate card detection with multiple methods")
        print("Instructions:")
        print("1. Click and drag to select your HAND region (2 cards)")
        print("2. Click and drag to select the BOARD region (5 cards)")
        print("3. Press 'd' to detect cards with all methods")
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
        cv2.namedWindow("Ultimate Poker Detector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Ultimate Poker Detector", mouse_callback)
        
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
                status = "Press 'd' to detect cards with ALL methods"
            
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, "Ultimate detection: Contour + Grid + Multiple crops", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Ultimate Poker Detector", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                hand_region = None
                board_region = None
                print("Regions reset")
            elif key == ord('d') and hand_region and board_region:
                print("\n" + "="*60)
                print("ULTIMATE CARD DETECTION - ALL METHODS")
                print("="*60)
                
                # Detect hand cards with all methods
                print("\n🔍 Detecting HAND cards...")
                hx, hy, hw, hh = hand_region
                hand_roi = np.array(self.sct.grab({"top": hy, "left": hx, "width": hw, "height": hh}))[:, :, :3]
                
                # Try advanced contour detection
                hand_contour = self.detect_cards_advanced_contour(hand_roi, 2)
                print(f"Contour method: {hand_contour}")
                
                # Try grid scan
                hand_grid = self.detect_cards_grid_scan(hand_roi, 2)
                print(f"Grid method: {hand_grid}")
                
                # Combine results
                hand_cards = list(set(hand_contour + hand_grid))[:2]
                print(f"Final hand cards: {hand_cards}")
                
                # Detect board cards with all methods
                print("\n🔍 Detecting BOARD cards...")
                bx, by, bw, bh = board_region
                board_roi = np.array(self.sct.grab({"top": by, "left": bx, "width": bw, "height": bh}))[:, :, :3]
                
                # Try advanced contour detection
                board_contour = self.detect_cards_advanced_contour(board_roi, 5)
                print(f"Contour method: {board_contour}")
                
                # Try grid scan
                board_grid = self.detect_cards_grid_scan(board_roi, 5)
                print(f"Grid method: {board_grid}")
                
                # Combine results
                board_cards = list(set(board_contour + board_grid))[:5]
                print(f"Final board cards: {board_cards}")
                
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
                
                print(f"\n✅ ULTIMATE DETECTION COMPLETE!")
                print(f"Hand: {hand_cards}")
                print(f"Board: {board_cards}")
                print("Results saved to state.json")
                print("="*60)
        
        cv2.destroyAllWindows()

def main():
    detector = UltimateCardDetector()
    detector.run_ultimate_detection()

if __name__ == "__main__":
    main()
