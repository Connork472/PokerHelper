#!/usr/bin/env python3
"""
Simple Poker Card Detector
Shows live screen, allows region selection, detects cards
"""

import cv2
import numpy as np
import json
from mss import mss
from classify.infer_onnx_two import TwoHead

def main():
    print("Simple Poker Card Detector")
    print("=" * 30)
    print("Instructions:")
    print("1. You'll see your screen live")
    print("2. Click and drag to select your HAND (2 cards)")
    print("3. Click and drag to select the BOARD (5 cards)")
    print("4. Press 'd' to detect cards")
    print("5. Press 'q' to quit")
    print()
    
    # Initialize
    sct = mss()
    clf = TwoHead("rank.onnx", "suit.onnx")
    
    # Get screen size
    monitor = sct.monitors[1]
    screen_width = monitor['width']
    screen_height = monitor['height']
    
    # Scale for display (make it smaller)
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
                
                if w > 20 and h > 20:  # Minimum size
                    # Scale back to full screen coordinates
                    fx, fy, fw, fh = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    
                    if not hand_region:
                        hand_region = (fx, fy, fw, fh)
                        print(f"✓ Hand region selected: {hand_region}")
                    elif not board_region:
                        board_region = (fx, fy, fw, fh)
                        print(f"✓ Board region selected: {board_region}")
                        print("✓ Both regions selected! Press 'd' to detect cards.")
    
    # Create window
    cv2.namedWindow("Poker Detector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Poker Detector", mouse_callback)
    
    print("Starting live screen capture...")
    print("Select your regions by clicking and dragging!")
    
    while True:
        # Capture screen
        screen = np.array(sct.grab(monitor))[:, :, :3]
        
        # Scale down for display
        display_frame = cv2.resize(screen, (display_width, display_height))
        
        # Draw selected regions
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
        
        # Add status
        if not hand_region:
            status = "Select HAND region (2 cards)"
        elif not board_region:
            status = "Select BOARD region (5 cards)"
        else:
            status = "Press 'd' to detect cards"
        
        cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'd' to detect, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow("Poker Detector", display_frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') and hand_region and board_region:
            print("\nDetecting cards...")
            
            # Split into slots and detect
            def split_into_slots(roi, num_slots):
                h, w = roi.shape[:2]
                slot_width = w // num_slots
                slots = []
                for i in range(num_slots):
                    x_start = i * slot_width
                    x_end = (i + 1) * slot_width if i < num_slots - 1 else w
                    slots.append(roi[:, x_start:x_end])
                return slots
            
            def detect_cards_in_slots(slots):
                results = []
                for slot in slots:
                    if slot.size == 0:
                        results.append("")
                        continue
                    label, conf = clf.predict_with_conf(slot, 0.55)
                    results.append(label if conf > 0 else "")
                return results
            
            # Process hand
            hx, hy, hw, hh = hand_region
            hand_roi = np.array(sct.grab({"top": hy, "left": hx, "width": hw, "height": hh}))[:, :, :3]
            hand_slots = split_into_slots(hand_roi, 2)
            hand_results = detect_cards_in_slots(hand_slots)
            hand_cards = [card for card in hand_results if card]
            
            # Process board
            bx, by, bw, bh = board_region
            board_roi = np.array(sct.grab({"top": by, "left": bx, "width": bw, "height": bh}))[:, :, :3]
            board_slots = split_into_slots(board_roi, 5)
            board_results = detect_cards_in_slots(board_slots)
            board_cards = [card for card in board_results if card]
            
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
            
            print(f"✓ Detection complete!")
            print(f"Hand: {hand_cards}")
            print(f"Board: {board_cards}")
            print("Results saved to state.json")
            print("Press 'q' to quit or select new regions")
    
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main()
