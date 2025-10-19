"""Interactive ROI selector for the live Poker Vision pipeline.

This utility lets non-technical users quickly mark the on-screen regions that
contain their hole cards and the community board.  The configuration is stored
as JSON and reused by :mod:`poker_vision.main_site_agnostic`.

Controls
--------
* ``h`` – draw/update *my_hand_region*
* ``b`` – draw/update *board_region*
* ``a`` – draw/update *amounts_region* (optional, reserved for future use)
* ``s`` – save immediately
* ``m`` – cycle through available monitors (if you play on a secondary screen)
* ``q`` or ``Esc`` – quit (configuration is auto-saved)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mss import mss
import json

_CONFIG_FILENAME = "roi_config.json"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / _CONFIG_FILENAME

_BOX_COLOURS = {
    "my_hand_region": (0, 200, 255),
    "board_region": (120, 240, 80),
    "amounts_region": (255, 180, 60),
}


@dataclass
class Region:
    """Simple struct capturing an ROI in MSS coordinate space."""

    left: int
    top: int
    width: int
    height: int

    def as_dict(self) -> Dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class ROISelector:
    """Interactive tool to define rectangular capture regions."""

    def __init__(self, config_path: Optional[str] = None, monitor_index: int = 1) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self.monitor_index = monitor_index
        self._sct = mss()
        self._regions: Dict[str, Region] = {}
        self._active_name: Optional[str] = None
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_rect: Optional[Tuple[int, int, int, int]] = None
        self._monitor = self._pick_monitor(self.monitor_index)
        self._load_config()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                self._regions = {
                    key: Region(**value)
                    for key, value in data.items()
                    if {"left", "top", "width", "height"}.issubset(value)
                }
            except Exception:
                self._regions = {}
        else:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._regions = {}

    def _save_config(self) -> None:
        payload = {name: region.as_dict() for name, region in self._regions.items()}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Capture + drawing helpers
    # ------------------------------------------------------------------
    def _pick_monitor(self, index: int) -> Dict[str, int]:
        monitors = self._sct.monitors
        index = max(1, min(index, len(monitors) - 1))
        return monitors[index]

    def _to_monitor_space(self, x: int, y: int) -> Tuple[int, int]:
        return x + self._monitor["left"], y + self._monitor["top"]

    def _draw_regions(self, frame: np.ndarray) -> None:
        for name, region in self._regions.items():
            colour = _BOX_COLOURS.get(name, (255, 255, 0))
            x1 = region.left - self._monitor["left"]
            y1 = region.top - self._monitor["top"]
            x2 = x1 + region.width
            y2 = y1 + region.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                frame,
                name.replace("_", " "),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
            )

    def _draw_instructions(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        panel_h = 120
        cv2.rectangle(
            overlay,
            (0, frame.shape[0] - panel_h),
            (frame.shape[1], frame.shape[0]),
            (20, 20, 20),
            -1,
        )
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        lines = [
            "Keys: h=hand  b=board  a=amounts  s=save  m=monitor  q=quit",
            "Drag with the left mouse button to draw the highlighted region.",
            "Tip: keep the selection just around the cards for best accuracy.",
        ]
        for idx, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (16, frame.shape[0] - panel_h + 30 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        legend_y = frame.shape[0] - 20
        for name, colour in _BOX_COLOURS.items():
            label = name.replace("_", " ")
            cv2.circle(frame, (20, legend_y - 5), 8, colour, -1)
            cv2.putText(
                frame,
                label,
                (36, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            legend_y -= 22

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the OpenCV ROI selection loop."""

        print(
            "\nPokerHelper ROI selector\n"
            "-------------------------------------------\n"
            "1. Press h (hand) or b (board) to choose the next region.\n"
            "2. Click and drag to draw the rectangle.\n"
            "3. Press s at any time to save.\n"
            "Press m to switch monitors if needed, or q when you are done.\n"
        )

        window = "PokerHelper ROI"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

        def _mouse(event, x, y, _flags, _userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drag_start = (x, y)
                self._drag_rect = None
            elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
                x0, y0 = self._drag_start
                self._drag_rect = (
                    min(x0, x),
                    min(y0, y),
                    abs(x - x0),
                    abs(y - y0),
                )
            elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
                if self._active_name and self._drag_rect is not None:
                    dx, dy, w, h = self._drag_rect
                    if w > 10 and h > 10:
                        abs_x, abs_y = self._to_monitor_space(dx, dy)
                        self._regions[self._active_name] = Region(abs_x, abs_y, w, h)
                        print(f"Updated {self._active_name} → ({abs_x}, {abs_y}, {w}, {h})")
                self._drag_start = None
                self._drag_rect = None
                self._active_name = None

        cv2.setMouseCallback(window, _mouse)

        while True:
            frame = np.array(self._sct.grab(self._monitor))[:, :, :3]
            display = frame.copy()
            self._draw_regions(display)
            if self._drag_rect is not None:
                x, y, w, h = self._drag_rect
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self._draw_instructions(display)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self._save_config()
                print(f"Saved to {self.config_path}")
            elif key == ord("h"):
                self._active_name = "my_hand_region"
                print("Draw your HAND region…")
            elif key == ord("b"):
                self._active_name = "board_region"
                print("Draw the BOARD region…")
            elif key == ord("a"):
                self._active_name = "amounts_region"
                print("Draw the AMOUNTS region (optional)…")
            elif key == ord("m"):
                self.monitor_index += 1
                if self.monitor_index >= len(self._sct.monitors):
                    self.monitor_index = 1
                self._monitor = self._pick_monitor(self.monitor_index)
                print(f"Switched to monitor #{self.monitor_index}: {self._monitor['width']}x{self._monitor['height']}")

        self._save_config()
        cv2.destroyAllWindows()
        print(f"Configuration saved to {self.config_path}\n")


if __name__ == "__main__":
    ROISelector().run()
