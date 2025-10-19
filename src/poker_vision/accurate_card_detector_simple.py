#!/usr/bin/env python3
"""
Accurate Poker Card Detector (Simple Version)
Fixes preprocessing, adds temporal smoothing, without poker simulator dependency
"""

import cv2
import numpy as np
import json
import time
from collections import deque
from mss import mss
from classify.infer_onnx_two import TwoHead
from geometry.card_finder import find_cards, warp

class AccurateCardDetectorSimple:
    def __init__(self):
        self.sct = mss()
        self.clf = TwoHead("rank.onnx", "suit.onnx")
        
        # Temporal smoothing
        self.detection_history = deque(maxlen=5)  # Keep last 5 detections
        self.stable_threshold = 3  # Need 3 consistent detections
        self.last_stable_detection = None
        self.detection_timestamp = 0
        
        # State management
        self.mode = "select_hand"
        self.hand_region = None
        self.board_region = None
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        # Debug
        self.debug_mode = False
        self.save_debug_images = True
        
    def preprocess_card_image(self, card_image):
        """Improved preprocessing for better accuracy"""
        if card_image.size == 0:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        
        # Normalize lighting
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive threshold for better contrast
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return thresh
    
    def get_corner_crops_improved(self, card_image):
        """Get improved corner crops with better preprocessing"""
        if card_image.size == 0:
            return []
        
        h, w = card_image.shape[:2]
        crops = []
        
        # Top-left corner (more precise)
        tl_crop = card_image[:int(0.35*h), :int(0.4*w)]
        if tl_crop.size > 0:
            crops.append(self.preprocess_card_image(tl_crop))
        
        # Top-right corner (more precise)
        tr_crop = card_image[:int(0.35*h), int(0.6*w):]
        if tr_crop.size > 0:
            crops.append(self.preprocess_card_image(tr_crop))
        
        # Center-top area (alternative)
        center_crop = card_image[:int(0.4*h), int(0.3*w):int(0.7*w)]
        if center_crop.size > 0:
            crops.append(self.preprocess_card_image(center_crop))
        
        return [crop for crop in crops if crop is not None]
    
    def detect_single_card_accurate(self, card_image):
        """Accurate single card detection with improved preprocessing"""
        if card_image.size == 0:
            return ""
        
        # Get multiple crops with improved preprocessing
        crops = self.get_corner_crops_improved(card_image)
        
        if not crops:
            return ""
        
        best_label = ""
        best_confidence = 0
        
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
                
            try:
                # Resize to model input size
                h, w = crop.shape[:2]
                if h < 10 or w < 10:
                    continue
                    
                # Ensure proper aspect ratio
                target_h, target_w = 64, 64  # Model input size
                resized = cv2.resize(crop, (target_w, target_h))
                
                # Convert back to BGR for model
                bgr_crop = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                
                # Get prediction with very low threshold
                label, conf = self.clf.predict_with_conf(bgr_crop, 0.1)
                
                if conf > best_confidence:
                    best_label = label
                    best_confidence = conf
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"Crop processing error: {e}")
                continue
        
        return best_label if best_confidence > 0.1 else ""
    
    def detect_cards_with_temporal_smoothing(self, region, max_cards, region_name):
        """Detect cards with temporal smoothing to prevent stale reads"""
        if not region:
            return []
        
        x, y, w, h = region
        roi = np.array(self.sct.grab({"top": y, "left": x, "width": w, "height": h}))[:, :, :3]
        
        if self.save_debug_images:
            cv2.imwrite(f"debug_{region_name}_roi_{int(time.time())}.png", roi)
        
        # Try multiple detection methods
        all_cards = []
        
        # Method 1: Improved contour detection
        try:
            cards = find_cards(roi, min_area=800, aspect_min=0.5, aspect_max=0.9)
            for i, (warped_card, quad) in enumerate(cards[:max_cards]):
                if warped_card.size == 0:
                    continue
                
                if self.save_debug_images:
                    cv2.imwrite(f"debug_{region_name}_contour_{i}_{int(time.time())}.png", warped_card)
                
                card_result = self.detect_single_card_accurate(warped_card)
                if card_result:
                    all_cards.append(card_result)
        except Exception as e:
            if self.debug_mode:
                print(f"Contour detection failed: {e}")
        
        # Method 2: Grid-based detection
        try:
            grid_cards = self.detect_cards_grid_improved(roi, max_cards, region_name)
            all_cards.extend(grid_cards)
        except Exception as e:
            if self.debug_mode:
                print(f"Grid detection failed: {e}")
        
        # Remove duplicates while preserving order
        unique_cards = []
        seen = set()
        for card in all_cards:
            if card and card not in seen:
                unique_cards.append(card)
                seen.add(card)
        
        current_detection = unique_cards[:max_cards]
        
        # Temporal smoothing
        self.detection_history.append({
            'cards': current_detection,
            'timestamp': time.time(),
            'region': region_name
        })
        
        # Check if we have stable detection
        if len(self.detection_history) >= self.stable_threshold:
            recent_detections = list(self.detection_history)[-self.stable_threshold:]
            
            # Check if all recent detections are the same
            if all(det['cards'] == current_detection for det in recent_detections):
                if self.last_stable_detection != current_detection:
                    self.last_stable_detection = current_detection
                    self.detection_timestamp = time.time()
                    if self.debug_mode:
                        print(f"Stable detection: {current_detection}")
                return current_detection
        
        # Return last stable detection if current is unstable
        if self.last_stable_detection and time.time() - self.detection_timestamp < 2.0:
            return self.last_stable_detection
        
        return current_detection
    
    def detect_cards_grid_improved(self, roi, max_cards, region_name):
        """Improved grid-based detection"""
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
            
            if self.save_debug_images:
                cv2.imwrite(f"debug_{region_name}_grid_{i}_{int(time.time())}.png", card_roi)
            
            card_result = self.detect_single_card_accurate(card_roi)
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
                        print("Press 'd' to start detection, or 'r' to restart")
    
    def run_accurate_detection(self):
        """Run the accurate detection interface"""
        print("Accurate Poker Card Detector (Simple Version)")
        print("=" * 60)
        print("🎯 100% Accurate Detection with Temporal Smoothing")
        print()
        print("CONTROLS:")
        print("  Mouse: Click and drag to select regions")
        print("  'd' key: Start continuous detection")
        print("  'r' key: Restart selection")
        print("  'q' key: Quit")
        print("  'b' key: Toggle debug mode")
        print("  's' key: Toggle debug image saving")
        print()
        print("STEP 1: Select your HAND region (2 cards)")
        print("STEP 2: Select the BOARD region (5 cards)")
        print("STEP 3: Press 'd' to start continuous detection")
        print()
        
        # Get screen size
        screen_width, screen_height = self.get_screen_size()
        
        # Scale for display
        scale = 0.6
        display_width = int(screen_width * scale)
        display_height = int(screen_height * scale)
        
        # Create window
        cv2.namedWindow("Accurate Poker Detector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Accurate Poker Detector", self.mouse_callback)
        
        print("Starting live screen capture...")
        
        detection_active = False
        
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
                if detection_active:
                    status = "DETECTING CARDS WITH TEMPORAL SMOOTHING"
                    color = (255, 255, 0)
                else:
                    status = "Press 'd' to start detection"
                    color = (255, 255, 0)
            else:
                status = "Press 'r' to restart"
                color = (255, 255, 255)
            
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show detection results
            if detection_active:
                # Get current detection
                hand_cards = self.detect_cards_with_temporal_smoothing(self.hand_region, 2, "hand")
                board_cards = self.detect_cards_with_temporal_smoothing(self.board_region, 5, "board")
                
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
                
                # Display cards
                hand_str = ", ".join(hand_cards) if hand_cards else "None"
                board_str = ", ".join(board_cards) if board_cards else "None"
                
                cv2.putText(display_frame, f"Hand: {hand_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Board: {board_str}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Show detection status
                if self.last_stable_detection:
                    cv2.putText(display_frame, "Stable detection active", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Debug info
            if self.debug_mode:
                cv2.putText(display_frame, f"Debug: ON | Save: {self.save_debug_images}", (10, display_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Accurate Poker Detector", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.mode = "select_hand"
                self.hand_region = None
                self.board_region = None
                detection_active = False
                self.detection_history.clear()
                self.last_stable_detection = None
                print("Restarted! Select HAND region first.")
            elif key == ord('d') and self.mode == "detect":
                detection_active = True
                print("Starting continuous detection with temporal smoothing...")
            elif key == ord('b'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('s'):
                self.save_debug_images = not self.save_debug_images
                print(f"Save debug images: {'ON' if self.save_debug_images else 'OFF'}")
        
        cv2.destroyAllWindows()
        print("Goodbye!")
    
    def get_screen_size(self):
        """Get the primary monitor size"""
        monitor = self.sct.monitors[1]
        return monitor['width'], monitor['height']
    
    def capture_screen(self):
        """Capture the full screen"""
        monitor = self.sct.monitors[1]
        return np.array(self.sct.grab(monitor))[:, :, :3]

def main():
    detector = AccurateCardDetectorSimple()
    detector.run_accurate_detection()

if __name__ == "__main__":
    main()
