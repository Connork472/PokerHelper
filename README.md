# PokerHelper

PokerHelper is a comprehensive toolkit for detecting Texas Hold'em cards from
the screen and calculating live win probability.  The project couples a
screen-capture pipeline ("Poker Vision") with a Monte Carlo simulator.  This
release focuses on stability and approachability for everyday players—no data
science background required.

## Key Features

- **Accurate live detection** – Rank and suit models run via ONNX Runtime with
  preprocessing that mirrors the synthetic training pipeline for consistent
  reads.
- **Temporal smoothing** – Rolling consensus per card slot suppresses one-frame
  glitches and avoids stale cards lingering on screen.
- **Built-in win probability** – Once your hole cards and board are known the
  simulator automatically computes win %, tie %, and Kelly fraction.
- **Actionable feedback** – Console + overlay instructions, suit legends,
  confidences, and stability warnings keep you informed.
- **Full telemetry** – Every frame is logged to `logs/detections.csv` (timestamp,
  label, confidence, stability) so you can audit performance afterwards.

## Installation

1. Create/activate a Python 3.11 environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   OpenCV requires GUI support (Tk/Qt) for window display.  On macOS install the
   official Python.org build or ensure Homebrew Tcl/Tk is configured (see
   Troubleshooting below).

## Quick Start

### 1. Select capture regions (one-time)

Run the ROI selector and drag rectangles around your hole cards and the board.

```bash
python -m poker_vision.ui.roi_select
```

Controls: `h` hand, `b` board, `a` optional amounts, `s` save, `m` cycle monitor,
`q` quit.  Keep rectangles tight around the cards for best accuracy.

### 2. Launch the live detector

```bash
python -m poker_vision.main_site_agnostic
```

What you get:

- Console help with all hotkeys and win probability updates.
- Optional overlay window showing cropped ROIs, card reads, confidences,
  stability (⚠ marks unstable slots), and suit colour legend.
- Continuous updates written to `output/state.json`:

  ```json
  {
    "my_cards": ["As", "Kh"],
    "board": ["7h", "2d", "2s"],
    "pot": null,
    "to_call": null,
    "stacks": {},
    "equity": 0.65,
    "kelly": 0.30
  }
  ```

### 3. Keys during live detection

| Key | Action |
| --- | ------ |
| `q` / `Esc` | Quit |
| `p` | Pause/resume capture |
| `v` | Toggle overlay preview window (place it away from the ROIs to avoid self-capture) |
| `n` | Toggle adaptive brightness normalisation (helps under glare or dim lighting) |
| `r` | Reload `config/roi_config.json` without restarting |
| `[` / `]` | Decrease/increase target FPS (2–4 fps recommended) |
| `?` | Reprint help text in the console |

### 4. Review detection logs

After a session inspect `logs/detections.csv` for per-frame diagnostics.  Columns
include raw label, combined confidence, probability margin, stabilised label,
and whether the slot was considered stable.

## Win Probability Integration

The detector automatically calls the local CLI simulator when:

- Both hole cards are confirmed (two different cards), and
- The community board contains only unique cards (0–5 cards).

Results appear in the console and overlay as soon as they are available and are
also stored in `state.json` (`equity` + `kelly`).  Default simulator settings:
6 players, 50,000 trials.  Adjust the constants in
`src/poker_vision/main_site_agnostic.py` if you prefer different defaults.

## Manual Entry Fallback

Need to sanity-check odds or your screen capture is unavailable?  Use the manual
CLI or GUI simulators:

```bash
python cli_main.py        # lightweight terminal workflow
python simulator_gui.py   # form-based desktop app
```

Both tools accept card codes like `As Kh` or `10c 9d` and display win %, tie %,
Kelly fraction, and confidence intervals.  You can keep the manual simulator
open alongside the live detector for quick corrections.

## Project Structure

```
PokerHelper/
├── README.md
├── requirements.txt
├── output/state.json
├── logs/detections.csv        # created on first run
├── config/roi_config.json     # ROI definition from the selector
├── rank.onnx / suit.onnx      # ONNX models
├── src/
│   ├── poker_vision/          # Detection pipeline
│   │   ├── main_site_agnostic.py
│   │   ├── ui/roi_select.py
│   │   ├── classify/infer_onnx_two.py
│   │   └── ...
│   └── simulator/
│       └── poker_cli_session.py
└── main.py / run.py / cli_main.py / simulator_gui.py
```

## Troubleshooting

- **Overlay captures itself (hall-of-mirrors):** Move the preview window to a
  different screen or outside the selected ROIs.  You can also toggle it off via
  `v` and rely on console output only.
- **Model misreads:** Ensure the ROI tightly encloses the cards and experiment
  with the `n` key (adaptive normalisation).  Check `logs/detections.csv` to
  spot systematically low-confidence slots.
- **`ModuleNotFoundError: _tkinter` on macOS:** Use the official Python.org
  installer which bundles Tcl/Tk, or install Tcl/Tk via Homebrew and rebuild
  Python (see the note at the end of the previous README revision).  You can
  always fall back to CLI modes (`python cli_main.py`).
- **Performance issues:** Use the `[` key to slow the capture loop (longer delay
  between grabs) or reduce monitor resolution.

Happy grinding!  If you capture an interesting log segment, share
`logs/detections.csv` along with the corresponding screenshots so the models can
be further tuned.
