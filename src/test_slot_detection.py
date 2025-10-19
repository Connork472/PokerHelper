#!/usr/bin/env python3
"""
Test script for the new slot-based poker detection system.
This script runs a quick test to verify the detection is working.
"""

import json
import time
import subprocess
import sys
import os

def test_debug_snapshot():
    """Test the debug snapshot functionality"""
    print("Testing debug snapshot...")
    try:
        result = subprocess.run([
            sys.executable, "poker_vision/debug_snapshot.py"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Debug snapshot test passed")
            print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ Debug snapshot test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Debug snapshot test failed: {e}")
        return False

def test_state_json():
    """Test that state.json is being written correctly"""
    print("Testing state.json output...")
    try:
        if os.path.exists("state.json"):
            with open("state.json") as f:
                state = json.load(f)
            
            # Check required fields
            required_fields = ["my_cards", "board", "pot", "to_call", "stacks"]
            for field in required_fields:
                if field not in state:
                    print(f"✗ Missing field: {field}")
                    return False
            
            print("✓ State.json has correct structure")
            print(f"Current state: {state}")
            return True
        else:
            print("✗ state.json not found")
            return False
    except Exception as e:
        print(f"✗ State.json test failed: {e}")
        return False

def main():
    print("PokerHelper Slot-Based Detection Test")
    print("=" * 40)
    
    # Change to the project directory
    os.chdir("/Users/connor/PokerHelper")
    
    # Test debug snapshot
    debug_ok = test_debug_snapshot()
    
    # Test state.json
    state_ok = test_state_json()
    
    print("\nTest Results:")
    print(f"Debug snapshot: {'✓ PASS' if debug_ok else '✗ FAIL'}")
    print(f"State.json: {'✓ PASS' if state_ok else '✗ FAIL'}")
    
    if debug_ok and state_ok:
        print("\n🎉 All tests passed! The slot-based detection is working correctly.")
        print("\nTo run the live detector:")
        print("  python poker_vision/main_slot_mode.py")
        print("\nKeyboard controls:")
        print("  q/esc: quit")
        print("  p: pause/resume") 
        print("  v: toggle preview")
        print("  r: reload config")
        print("  [/]: adjust speed")
    else:
        print("\n❌ Some tests failed. Please check the configuration and models.")

if __name__ == "__main__":
    main()
