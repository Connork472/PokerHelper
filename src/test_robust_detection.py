#!/usr/bin/env python3
"""
Test script for the robust card detection system
"""

import json
import cv2
import numpy as np
from mss import mss
from poker_vision.classify.infer_onnx_two import TwoHead
from poker_vision.geometry.card_finder import find_cards

def test_improved_classifier():
    """Test the improved classifier with better confidence calculation"""
    print("Testing improved classifier...")
    
    clf = TwoHead("rank.onnx", "suit.onnx")
    
    # Test with a simple image
    sct = mss()
    
    # Load existing ROI config
    try:
        with open("roi_config.json") as f:
            cfg = json.load(f)
    except:
        print("No roi_config.json found. Please run the interactive detector first.")
        return False
    
    # Test hand region
    if "my_hand_region" in cfg:
        print("Testing hand region...")
        hand_roi = np.array(sct.grab(cfg["my_hand_region"]))[:, :, :3]
        
        # Try contour detection
        cards = find_cards(hand_roi, min_area=800)
        print(f"Found {len(cards)} cards via contour detection")
        
        for i, (warped_card, quad) in enumerate(cards[:2]):
            if warped_card.size > 0:
                label, conf = clf.predict_with_conf(warped_card, 0.3)
                print(f"Card {i+1}: {label} (confidence: {conf:.3f})")
    
    # Test board region  
    if "board_region" in cfg:
        print("Testing board region...")
        board_roi = np.array(sct.grab(cfg["board_region"]))[:, :, :3]
        
        # Try contour detection
        cards = find_cards(board_roi, min_area=700)
        print(f"Found {len(cards)} cards via contour detection")
        
        for i, (warped_card, quad) in enumerate(cards[:5]):
            if warped_card.size > 0:
                label, conf = clf.predict_with_conf(warped_card, 0.3)
                print(f"Card {i+1}: {label} (confidence: {conf:.3f})")
    
    return True

def main():
    print("Robust Card Detection Test")
    print("=" * 30)
    
    success = test_improved_classifier()
    
    if success:
        print("\n✓ Test completed successfully!")
        print("\nTo use the robust detector:")
        print("  python poker_vision/robust_card_detector.py")
    else:
        print("\n❌ Test failed. Please check your setup.")

if __name__ == "__main__":
    main()
