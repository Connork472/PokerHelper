import json, time, cv2, numpy as np, signal
from mss import mss
from geometry.card_finder import find_cards
from classify.infer_onnx_two import TwoHead

def grab(sct, rect):
  return np.array(sct.grab(rect))[:, :, :3]

def write_state(p, path="state.json"):
  with open(path, "w") as f: json.dump(p, f, indent=2)

def load_cfg(path="roi_config.json"):
  with open(path) as f: return json.load(f)

def mean_x(quad):
  return float(quad[:,0].mean())

def main():
  sct = mss()
  clf = TwoHead("rank.onnx", "suit.onnx")
  cfg = load_cfg()

  preview = False
  paused = False
  interval = 0.35
  win = "PokerVision"

  if preview:
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)

  running = True
  def handle_sigint(sig, frame):
    nonlocal running
    running = False
  signal.signal(signal.SIGINT, handle_sigint)

  canvas_w, canvas_h = 900, 360

  while running:
    t0 = time.time()

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
        try: cv2.destroyWindow(win)
        except: pass
    elif key == ord('r'):
      try: cfg = load_cfg()
      except: pass
    elif key == ord('['):
      interval = max(0.05, interval - 0.05)
    elif key == ord(']'):
      interval = min(1.00, interval + 0.05)

    if paused:
      time.sleep(0.05)
      continue

    state = {"my_cards": [], "board": [], "pot": None, "to_call": None, "stacks": {}}

    if "my_hand_region" in cfg:
      hand = grab(sct, cfg["my_hand_region"])
      hcards = find_cards(hand, min_area=800)
      hcards = sorted(hcards, key=lambda c: mean_x(c[1]))
      state["my_cards"] = [clf.predict(c[0]) for c in hcards[:2]]

    if "board_region" in cfg:
      board = grab(sct, cfg["board_region"])
      bcards = find_cards(board, min_area=700)
      bcards = sorted(bcards, key=lambda c: mean_x(c[1]))
      state["board"] = [clf.predict(c[0]) for c in bcards[:5]]

    write_state(state)
    print(state)

    if preview:
      canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
      x = 20
      if "my_hand_region" in cfg:
        h = cv2.resize(hand, (260, 180))
        canvas[20:200, x:x+260] = h
        cv2.putText(canvas, f"HAND: {','.join(state['my_cards'])}", (x, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        x += 300
      if "board_region" in cfg:
        b = cv2.resize(board, (520, 180))
        canvas[20:200, x:x+520] = b
        cv2.putText(canvas, f"BOARD: {','.join(state['board'])}", (x, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
      cv2.putText(canvas, f"{'PAUSED' if paused else 'RUN'} {interval:.2f}s  q/esc quit  p pause  v preview  r reload  [ ] fps", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
      cv2.imshow(win, canvas)

    dt = time.time() - t0
    if interval - dt > 0:
      time.sleep(interval - dt)

  try: cv2.destroyAllWindows()
  except: pass

if __name__ == "__main__":
  main()
