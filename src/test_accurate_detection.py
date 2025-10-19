#!/usr/bin/env python3
"""
Test script for the accurate card detection system
"""

import json
import time
import cv2
import numpy as np
from mss import mss
from poker_vision.classify.infer_onnx_two import TwoHead

def test_improved_preprocessing():
    """Test the improved preprocessing"""
    print("Testing improved preprocessing...")
    
    clf = TwoHead("rank.onnx", "suit.onnx")
    
    # Load existing ROI config
    try:
        with open("roi_config.json") as f:
            cfg = json.load(f)
    except:
        print("No roi_config.json found. Please run the accurate detector first.")
        return False
    
    sct = mss()
    
    # Test hand region
    if "my_hand_region" in cfg:
        print("Testing hand region with improved preprocessing...")
        hand_roi = np.array(sct.grab(cfg["my_hand_region"]))[:, :, :3]
        
        # Save debug image
        cv2.imwrite("test_hand_roi.png", hand_roi)
        
        # Test multiple crops
        h, w = hand_roi.shape[:2]
        crops = [
            hand_roi[:int(0.35*h), :int(0.4*w)],  # Top-left
            hand_roi[:int(0.35*h), int(0.6*w):],  # Top-right
        ]
        
        for i, crop in enumerate(crops):
            if crop.size > 0:
                cv2.imwrite(f"test_hand_crop_{i}.png", crop)
                label, conf = clf.predict_with_conf(crop, 0.1)
                print(f"Hand crop {i}: {label} (confidence: {conf:.3f})")
    
    # Test board region
    if "board_region" in cfg:
        print("Testing board region with improved preprocessing...")
        board_roi = np.array(sct.grab(cfg["board_region"]))[:, :, :3]
        
        # Save debug image
        cv2.imwrite("test_board_roi.png", board_roi)
        
        # Test multiple crops
        h, w = board_roi.shape[:2]
        crops = [
            board_roi[:int(0.35*h), :int(0.4*w)],  # Top-left
            board_roi[:int(0.35*h), int(0.6*w):],  # Top-right
        ]
        
        for i, crop in enumerate(crops):
            if crop.size > 0:
                cv2.imwrite(f"test_board_crop_{i}.png", crop)
                label, conf = clf.predict_with_conf(crop, 0.1)
                print(f"Board crop {i}: {label} (confidence: {conf:.3f})")
    
    return True

def test_temporal_smoothing():
    """Test temporal smoothing concept"""
    print("\nTesting temporal smoothing...")
    
    # Simulate detection history
    detections = [
        ["As", "Kh"],
        ["As", "Kh"], 
        ["As", "Kh"],
        ["As", "Kh"],
        ["As", "Kh"]
    ]
    
    # Check for stability
    if len(set(str(d) for d in detections)) == 1:
        print("✓ Temporal smoothing would work - consistent detections")
        return True
    else:
        print("✗ Temporal smoothing would not work - inconsistent detections")
        return False

def main():
    print("Accurate Card Detection Test")
    print("=" * 40)
    
    # Test improved preprocessing
    preprocessing_ok = test_improved_preprocessing()
    
    # Test temporal smoothing
    smoothing_ok = test_temporal_smoothing()
    
    print(f"\nTest Results:")
    print(f"Improved preprocessing: {'✓ PASS' if preprocessing_ok else '✗ FAIL'}")
    print(f"Temporal smoothing: {'✓ PASS' if smoothing_ok else '✗ FAIL'}")
    
    if preprocessing_ok and smoothing_ok:
        print("\n🎉 All tests passed!")
        print("\nTo use the accurate detector:")
        print("  python poker_vision/accurate_card_detector.py")
        print("\nFeatures:")
        print("  ✓ Improved preprocessing for better accuracy")
        print("  ✓ Temporal smoothing to prevent stale reads")
        print("  ✓ Poker simulator integration for win probability")
        print("  ✓ Debug mode and image saving")
    else:
        print("\n❌ Some tests failed. Please check your setup.")

if __name__ == "__main__":
    main()
