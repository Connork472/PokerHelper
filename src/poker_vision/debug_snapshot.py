import json, cv2, numpy as np
from mss import mss
from classify.infer_onnx_two import TwoHead

def grab(sct, r):
    return np.array(sct.grab(r))[:, :, :3]

def split_into_slots(roi_image, num_slots):
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

def detect_cards_in_slots(slots, classifier, confidence_threshold=0.55):
    """Detect cards in each slot using the classifier"""
    results = []
    for i, slot in enumerate(slots):
        if slot.size == 0:
            results.append("")
            continue
            
        # Save slot image for debugging
        cv2.imwrite(f"debug_slot_{i}.png", slot)
        
        label, conf = classifier.predict_with_conf(slot, confidence_threshold)
        results.append(label if conf > 0 else "")
        print(f"Slot {i}: {label} (conf: {conf:.3f})")
    
    return results

def main():
    with open("roi_config.json") as f:
        cfg = json.load(f)
    sct = mss()
    clf = TwoHead("rank.onnx", "suit.onnx")
    out = {"my_cards": [], "board": []}

    # Process hand region (2 slots)
    if "my_hand_region" in cfg:
        hand = grab(sct, cfg["my_hand_region"])
        cv2.imwrite("debug_hand.png", hand)
        hand_slots = split_into_slots(hand, 2)
        hand_results = detect_cards_in_slots(hand_slots, clf)
        out["my_cards"] = [card for card in hand_results if card]

    # Process board region (5 slots)
    if "board_region" in cfg:
        board = grab(sct, cfg["board_region"])
        cv2.imwrite("debug_board.png", board)
        board_slots = split_into_slots(board, 5)
        board_results = detect_cards_in_slots(board_slots, clf)
        out["board"] = [card for card in board_results if card]

    print(json.dumps(out))

if __name__ == "__main__":
  main()
