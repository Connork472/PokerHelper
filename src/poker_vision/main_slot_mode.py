import json, time, cv2, numpy as np, signal
from mss import mss
from classify.infer_onnx_two import TwoHead

def grab(sct, rect):
    return np.array(sct.grab(rect))[:, :, :3]

def write_state(p, path="state.json"):
    with open(path, "w") as f: 
        json.dump(p, f, indent=2)

def load_cfg(path="roi_config.json"):
    with open(path) as f: 
        return json.load(f)

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
    for slot in slots:
        if slot.size == 0:
            results.append("")
            continue
            
        label, conf = classifier.predict_with_conf(slot, confidence_threshold)
        results.append(label if conf > 0 else "")
    
    return results

def create_small_preview(hand_slots, board_slots, hand_results, board_results):
    """Create a small preview showing only slot crops, not full screen"""
    preview_h, preview_w = 200, 400
    canvas = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
    
    # Show hand slots (2 slots)
    if hand_slots:
        slot_h, slot_w = 80, 60
        for i, (slot, result) in enumerate(zip(hand_slots[:2], hand_results[:2])):
            if slot.size > 0:
                resized = cv2.resize(slot, (slot_w, slot_h))
                y_start = 10
                x_start = 10 + i * (slot_w + 10)
                canvas[y_start:y_start+slot_h, x_start:x_start+slot_w] = resized
                cv2.putText(canvas, result or "?", (x_start, y_start+slot_h+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Show board slots (5 slots)
    if board_slots:
        slot_h, slot_w = 60, 50
        for i, (slot, result) in enumerate(zip(board_slots[:5], board_results[:5])):
            if slot.size > 0:
                resized = cv2.resize(slot, (slot_w, slot_h))
                y_start = 120
                x_start = 10 + i * (slot_w + 5)
                canvas[y_start:y_start+slot_h, x_start:x_start+slot_w] = resized
                cv2.putText(canvas, result or "?", (x_start, y_start+slot_h+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    return canvas

def main():
    sct = mss()
    clf = TwoHead("rank.onnx", "suit.onnx")
    cfg = load_cfg()

    preview = False
    paused = False
    interval = 0.35
    win = "PokerVision"
    confidence_threshold = 0.55

    running = True
    def handle_sigint(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_sigint)

    while running:
        t0 = time.time()

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('v'):
            preview = not preview
            if preview:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
            else:
                try: 
                    cv2.destroyWindow(win)
                except: 
                    pass
        elif key == ord('r'):
            try: 
                cfg = load_cfg()
                print("Reloaded config")
            except Exception as e: 
                print(f"Failed to reload config: {e}")
        elif key == ord('['):
            interval = max(0.05, interval - 0.05)
        elif key == ord(']'):
            interval = min(1.00, interval + 0.05)

        if paused:
            time.sleep(0.05)
            continue

        # Initialize state
        state = {"my_cards": [], "board": [], "pot": None, "to_call": None, "stacks": {}}
        hand_slots = []
        board_slots = []

        # Process hand region (2 slots)
        if "my_hand_region" in cfg:
            try:
                hand_roi = grab(sct, cfg["my_hand_region"])
                hand_slots = split_into_slots(hand_roi, 2)
                hand_results = detect_cards_in_slots(hand_slots, clf, confidence_threshold)
                state["my_cards"] = [card for card in hand_results if card]
            except Exception as e:
                print(f"Hand detection error: {e}")
                state["my_cards"] = []

        # Process board region (5 slots)
        if "board_region" in cfg:
            try:
                board_roi = grab(sct, cfg["board_region"])
                board_slots = split_into_slots(board_roi, 5)
                board_results = detect_cards_in_slots(board_slots, clf, confidence_threshold)
                state["board"] = [card for card in board_results if card]
            except Exception as e:
                print(f"Board detection error: {e}")
                state["board"] = []

        # Write state
        write_state(state)
        print(f"Hand: {state['my_cards']}, Board: {state['board']}")

        # Show small preview (only slot crops, no full screen mirroring)
        if preview:
            try:
                preview_img = create_small_preview(hand_slots, board_slots, 
                                                 state["my_cards"], state["board"])
                cv2.putText(preview_img, f"{'PAUSED' if paused else 'RUN'} {interval:.2f}s", 
                           (10, preview_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.imshow(win, preview_img)
            except Exception as e:
                print(f"Preview error: {e}")

        # Maintain timing
        dt = time.time() - t0
        if interval - dt > 0:
            time.sleep(interval - dt)

    try: 
        cv2.destroyAllWindows()
    except: 
        pass

if __name__ == "__main__":
    main()
