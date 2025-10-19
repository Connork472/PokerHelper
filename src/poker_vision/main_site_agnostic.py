"""Site-agnostic Poker Vision runtime loop.

This module streams card detections from user-defined regions of the screen,
performs rank/suit classification via the ONNX models, smooths noisy
predictions, and writes the results to ``output/state.json``.  It is designed to
be resilient for non-technical players:

* Runtime preprocessing faithfully mirrors the synthetic training pipeline so
  the rank/suit models see the same grayscale glyphs they were trained on.
* Temporal smoothing provides a stable readout even when single frames misfire.
* Contextual console/overlay help makes the experience approachable.
* Every detection is logged (with confidences) for later analysis.
* Whenever both hole cards are known the Poker CLI simulator is invoked to
  estimate live win probability and Kelly fraction.
"""

from __future__ import annotations

import csv
import json
import signal
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mss import mss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from classify.infer_onnx_two import PredictionResult, TwoHead
from geometry.card_finder import find_cards
from simulator.poker_cli_session import kelly_even_money, simulate, to_treys

STATE_PATH = PROJECT_ROOT / "output" / "state.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "roi_config.json"
LOG_PATH = PROJECT_ROOT / "logs" / "detections.csv"
MODEL_RANK = PROJECT_ROOT / "rank.onnx"
MODEL_SUIT = PROJECT_ROOT / "suit.onnx"

SUIT_TO_SYMBOL = {
    "s": ("♠", (210, 210, 210)),
    "c": ("♣", (210, 210, 210)),
    "h": ("♥", (60, 60, 230)),
    "d": ("♦", (60, 60, 230)),
}


def _mean_x(quad: np.ndarray) -> float:
    return float(quad[:, 0].mean())


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Dict[str, int]]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"Warning: {path} is invalid JSON. Starting with an empty config.")
    else:
        print(f"ROI config missing at {path}. Launch ui/roi_select.py to create it.")
    return {}


def write_state(state: Dict[str, object], path: Path = STATE_PATH) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(state, indent=2))


def make_blank_prediction(height: int, width: int) -> PredictionResult:
    return PredictionResult(
        label="",
        confidence=0.0,
        margin=0.0,
        rank_label="",
        suit_label="",
        rank_confidence=0.0,
        suit_confidence=0.0,
        rank_margin=0.0,
        suit_margin=0.0,
        corner="",
        processed_tile=np.zeros((height, width), dtype=np.uint8),
    )


@dataclass
class SmoothedPrediction:
    label: str
    confidence: float
    margin: float
    stable: bool
    display_label: str
    instability_reason: str
    is_stale: bool
    latest: PredictionResult
    history_size: int


class TemporalSmoother:
    """Maintain per-slot detection history to stabilise predictions."""

    def __init__(
        self,
        slots: int,
        history: int = 6,
        stability_threshold: int = 3,
        conf_floor: float = 0.55,
        stale_after: float = 2.5,
    ) -> None:
        self.history_len = history
        self.stability_threshold = stability_threshold
        self.conf_floor = conf_floor
        self.stale_after = stale_after
        self._init_slots(slots)

    def _init_slots(self, slots: int) -> None:
        self.slots = slots
        self.history: List[Deque[Tuple[float, PredictionResult]]] = [
            deque(maxlen=self.history_len) for _ in range(slots)
        ]
        self.last_confirmed: List[str] = [""] * slots
        self.last_confirmed_time: List[float] = [0.0] * slots

    def update(self, slot_index: int, prediction: PredictionResult) -> SmoothedPrediction:
        now = time.time()
        queue = self.history[slot_index]
        queue.append((now, prediction))

        labels = [p.label for _, p in queue if p.label]
        label_counts = Counter(labels)
        top_label = label_counts.most_common(1)[0][0] if label_counts else ""
        occurrences = label_counts[top_label] if top_label else 0
        confidences = [p.confidence for _, p in queue if p.label == top_label]
        margins = [p.margin for _, p in queue if p.label == top_label]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        avg_margin = float(np.mean(margins)) if margins else 0.0

        stable = bool(top_label) and occurrences >= self.stability_threshold and avg_conf >= self.conf_floor
        reason = ""
        if not top_label:
            reason = "empty"
        elif not stable:
            reason = "changing" if occurrences < self.stability_threshold else "low_conf"

        display_label = self.last_confirmed[slot_index]
        stale = False
        if stable and top_label:
            display_label = top_label
            self.last_confirmed[slot_index] = top_label
            self.last_confirmed_time[slot_index] = now
        else:
            last_time = self.last_confirmed_time[slot_index]
            if display_label and last_time and now - last_time > self.stale_after:
                display_label = ""
                self.last_confirmed[slot_index] = ""
                self.last_confirmed_time[slot_index] = 0.0
                stale = True

        latest_prediction = queue[-1][1]
        return SmoothedPrediction(
            label=top_label,
            confidence=avg_conf,
            margin=avg_margin,
            stable=stable,
            display_label=display_label,
            instability_reason=reason,
            is_stale=stale,
            latest=latest_prediction,
            history_size=len(queue),
        )

    def confirmed(self) -> List[str]:
        return [lbl for lbl in self.last_confirmed]


class DetectionLogger:
    """Append detection telemetry to ``logs/detections.csv``."""

    HEADER = [
        "timestamp_iso",
        "region",
        "slot",
        "raw_label",
        "raw_confidence",
        "raw_margin",
        "smoothed_label",
        "smoothed_confidence",
        "smoothed_margin",
        "stable",
        "instability_reason",
        "corner",
    ]

    def __init__(self, path: Path = LOG_PATH) -> None:
        self.path = path
        _ensure_parent(self.path)
        if not self.path.exists():
            with self.path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.HEADER)

    def log(self, region: str, slot: int, smoothed: SmoothedPrediction) -> None:
        latest = smoothed.latest
        with self.path.open("a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    region,
                    slot,
                    latest.label,
                    f"{latest.confidence:.4f}",
                    f"{latest.margin:.4f}",
                    smoothed.display_label,
                    f"{smoothed.confidence:.4f}",
                    f"{smoothed.margin:.4f}",
                    int(smoothed.stable),
                    smoothed.instability_reason,
                    latest.corner,
                ]
            )


class EquityCalculator:
    """Background wrapper around the CLI simulator for live equity updates."""

    def __init__(self, players: int = 6, trials: int = 50_000) -> None:
        self.players = players
        self.trials = trials
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pending_key: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None
        self._result: Optional[Dict[str, object]] = None

    def request(self, hand: Sequence[str], board: Sequence[str]) -> None:
        key = (tuple(hand), tuple(board))
        with self._lock:
            if self._pending_key == key:
                return
            self._pending_key = key
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, args=(key,), daemon=True)
        self._thread.start()

    def latest(self) -> Optional[Dict[str, object]]:
        with self._lock:
            return self._result.copy() if self._result else None

    def _worker(self, key: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> None:
        hand, board = key
        try:
            treys_hand = to_treys(list(hand))
            treys_board = to_treys(list(board))
            win_p, tie_p, equity = simulate(self.players, treys_hand, treys_board, self.trials)
            kelly = kelly_even_money(equity)
            result = {
                "hand": list(hand),
                "board": list(board),
                "win": win_p,
                "tie": tie_p,
                "equity": equity,
                "kelly": kelly,
                "players": self.players,
                "trials": self.trials,
                "timestamp": time.time(),
            }
        except Exception as exc:  # pragma: no cover - defensive logging only
            result = {"error": str(exc), "hand": list(hand), "board": list(board), "timestamp": time.time()}
        with self._lock:
            if self._pending_key == key:
                self._result = result


class FrameController:
    """Utility to regulate the main loop frame rate."""

    def __init__(self, fps: float) -> None:
        self.target_fps = fps

    @property
    def interval(self) -> float:
        return 1.0 / self.target_fps if self.target_fps > 0 else 0.0

    def sleep(self, start_time: float) -> None:
        elapsed = time.time() - start_time
        remaining = self.interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def grab_rect(sct: mss, rect: Dict[str, int]) -> np.ndarray:
    return np.array(sct.grab(rect))[:, :, :3]


HELP_TEXT = """
PokerHelper Live Detection Controls
----------------------------------
 q / Esc  Quit
 p        Pause/resume detection
 v        Toggle the overlay preview window
 n        Toggle adaptive brightness normalisation (helps under glare)
 r        Reload ROI configuration
 [ / ]    Decrease / increase target FPS (2–4 fps recommended)
 ?        Show this help again

HUD legend:
  ♠ ♣ black suits, ♥ ♦ red suits.  A yellow ⚠ marker indicates the slot is
  still stabilising (either low confidence or changing quickly).
""".strip()


def build_overlay(
    hand_roi: Optional[np.ndarray],
    board_roi: Optional[np.ndarray],
    hand_predictions: Sequence[SmoothedPrediction],
    board_predictions: Sequence[SmoothedPrediction],
    equity: Optional[Dict[str, object]],
    status_text: str,
    fps_reading: float,
) -> np.ndarray:
    canvas = np.zeros((420, 900, 3), dtype=np.uint8)

    def _paste(img: np.ndarray, top_left: Tuple[int, int], size: Tuple[int, int]) -> None:
        if img is None or img.size == 0:
            return
        target_w, target_h = size
        resized = cv2.resize(img, (target_w, target_h))
        x, y = top_left
        canvas[y : y + target_h, x : x + target_w] = resized

    _paste(hand_roi, (20, 20), (280, 180))
    _paste(board_roi, (320, 20), (520, 180))

    cv2.putText(canvas, status_text, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Loop FPS: {fps_reading:.2f}", (680, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def _draw_predictions(preds: Sequence[SmoothedPrediction], origin: Tuple[int, int], title: str) -> None:
        x, y = origin
        cv2.putText(canvas, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for idx, pred in enumerate(preds):
            label = pred.display_label or "--"
            suit_symbol = ""
            suit_colour = (230, 230, 230)
            if len(label) >= 2 and label[-1] in SUIT_TO_SYMBOL:
                suit_symbol, suit_colour = SUIT_TO_SYMBOL[label[-1]]
            rank_part = label[:-1] if suit_symbol else label
            row_y = y + 35 + idx * 28
            text = f"{idx + 1}: {rank_part:>2} {suit_symbol}" if label != "--" else f"{idx + 1}: --"
            cv2.putText(canvas, text, (x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, suit_colour, 2)
            info = f"{pred.confidence*100:5.1f}% conf  margin {pred.margin:0.3f}"
            colour = (120, 230, 120) if pred.stable and pred.display_label else (60, 200, 255)
            cv2.putText(canvas, info, (x + 160, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
            if not pred.stable or pred.instability_reason:
                cv2.putText(canvas, "⚠", (x + 360, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 200, 255), 2)

    _draw_predictions(hand_predictions, (20, 260), "Hand")
    _draw_predictions(board_predictions, (20, 320), "Board")

    instructions = "Keys: q quit | p pause | v preview | n norm | r reload | [ ] fps"
    cv2.putText(canvas, instructions, (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, "Suits: ♠♣ black, ♥♦ red", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if equity and "equity" in equity:
        eq = equity.get("equity", 0.0)
        win = equity.get("win", 0.0)
        kelly = equity.get("kelly", 0.0)
        cv2.putText(
            canvas,
            f"Win% {win*100:5.2f}   Equity {eq*100:5.2f}%   Kelly {kelly*100:4.1f}%",
            (420, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 120),
            2,
        )
        cv2.putText(
            canvas,
            f"Players {equity.get('players', 0)} | Trials {equity.get('trials', 0):,}",
            (420, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 120),
            1,
        )

    return canvas


def print_intro() -> None:
    print(
        """
PokerHelper Live Detection
==========================
• Select your regions with `python -m poker_vision.ui.roi_select` if you have not already.
• Place the preview window away from the capture regions to avoid the "hall of mirrors" effect.
• Keys: q Quit | p Pause | v Preview | n Normalise brightness | r Reload ROI | [ ] FPS | ? Help
• The program writes results to output/state.json and logs/detections.csv.
""".strip()
    )
    print(HELP_TEXT)


def main() -> None:
    print_intro()

    classifier = TwoHead(str(MODEL_RANK), str(MODEL_SUIT), enable_adaptive_normalisation=True)
    blank_prediction = make_blank_prediction(classifier.height, classifier.width)

    sct = mss()
    config = load_config()
    frame_controller = FrameController(fps=3.0)
    smoother_hand = TemporalSmoother(slots=2)
    smoother_board = TemporalSmoother(slots=5)
    logger = DetectionLogger()
    equity = EquityCalculator()

    preview_enabled = False
    paused = False
    running = True
    last_equity_print_key: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None

    def _handle_sigint(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)

    while running:
        loop_start = time.time()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            paused = not paused
            print(f"Paused: {paused}")
        elif key == ord("v"):
            preview_enabled = not preview_enabled
            if preview_enabled:
                cv2.namedWindow("PokerVision", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("PokerVision", cv2.WND_PROP_TOPMOST, 1)
            else:
                cv2.destroyWindow("PokerVision")
        elif key == ord("r"):
            config = load_config()
            print("Reloaded ROI configuration.")
        elif key == ord("["):
            frame_controller.target_fps = max(1.5, frame_controller.target_fps - 0.5)
            print(f"Target FPS → {frame_controller.target_fps:.1f}")
        elif key == ord("]"):
            frame_controller.target_fps = min(4.5, frame_controller.target_fps + 0.5)
            print(f"Target FPS → {frame_controller.target_fps:.1f}")
        elif key == ord("n"):
            classifier.enable_adaptive_normalisation = not classifier.enable_adaptive_normalisation
            mode = "ON" if classifier.enable_adaptive_normalisation else "OFF"
            print(f"Adaptive brightness normalisation {mode}")
        elif key == ord("?"):
            print(HELP_TEXT)

        if paused:
            frame_controller.sleep(loop_start)
            continue

        state = {
            "my_cards": [],
            "board": [],
            "pot": None,
            "to_call": None,
            "stacks": {},
            "equity": None,
            "kelly": None,
        }

        hand_roi = None
        board_roi = None

        hand_predictions: List[SmoothedPrediction] = []
        board_predictions: List[SmoothedPrediction] = []

        if "my_hand_region" in config:
            hand_roi = grab_rect(sct, config["my_hand_region"])
            cards = find_cards(hand_roi, min_area=800)
            cards = sorted(cards, key=lambda c: _mean_x(c[1]))
            for idx in range(2):
                if idx < len(cards):
                    card_img = cards[idx][0]
                    prediction = classifier.predict_card(card_img)
                else:
                    prediction = blank_prediction
                smoothed = smoother_hand.update(idx, prediction)
                hand_predictions.append(smoothed)
                logger.log("hand", idx, smoothed)
        else:
            for idx in range(2):
                smoothed = smoother_hand.update(idx, blank_prediction)
                hand_predictions.append(smoothed)
                logger.log("hand", idx, smoothed)

        if "board_region" in config:
            board_roi = grab_rect(sct, config["board_region"])
            cards = find_cards(board_roi, min_area=700)
            cards = sorted(cards, key=lambda c: _mean_x(c[1]))
            for idx in range(5):
                if idx < len(cards):
                    card_img = cards[idx][0]
                    prediction = classifier.predict_card(card_img)
                else:
                    prediction = blank_prediction
                smoothed = smoother_board.update(idx, prediction)
                board_predictions.append(smoothed)
                logger.log("board", idx, smoothed)
        else:
            for idx in range(5):
                smoothed = smoother_board.update(idx, blank_prediction)
                board_predictions.append(smoothed)
                logger.log("board", idx, smoothed)

        hand_cards = [pred.display_label for pred in hand_predictions if pred.display_label]
        board_cards = [pred.display_label for pred in board_predictions if pred.display_label]

        state["my_cards"] = hand_cards
        state["board"] = board_cards

        board_unique = len(board_cards) == len(set(board_cards))
        if len(hand_cards) == 2 and len(set(hand_cards)) == 2 and board_unique:
            equity.request(hand_cards, board_cards)
        result = equity.latest()
        if result and (result.get("hand") != hand_cards or result.get("board") != board_cards):
            result = None
        if result and "error" not in result:
            state["equity"] = result.get("equity")
            state["kelly"] = result.get("kelly")
            key = (tuple(result.get("hand", [])), tuple(result.get("board", [])))
            if key != last_equity_print_key:
                last_equity_print_key = key
                print(
                    f"Win {result['win']*100:5.2f}% | Equity {result['equity']*100:5.2f}% | Kelly {result['kelly']*100:4.1f}%"
                )
        elif result and "error" in result:
            print(f"Equity calc error: {result['error']}")

        write_state(state)

        loop_dt = time.time() - loop_start
        fps_reading = 1.0 / loop_dt if loop_dt > 0 else 0.0

        status = (
            f"{'PAUSED' if paused else 'RUNNING'}  target {frame_controller.target_fps:.1f} fps  "
            "Keep overlay outside capture regions to avoid screen feedback."
        )

        if preview_enabled:
            overlay = build_overlay(hand_roi, board_roi, hand_predictions, board_predictions, result, status, fps_reading)
            cv2.imshow("PokerVision", overlay)

        frame_controller.sleep(loop_start)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
