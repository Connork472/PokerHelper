# PokerHelper

A comprehensive poker card detection and win probability calculation tool.

## Features

- **Poker Vision Detection**: Automatically detect cards from screen capture
- **Manual Simulator**: Enter cards manually for win probability calculation
- **Real-time Analysis**: Get equity and Kelly criterion calculations
- **Clean GUI**: Easy-to-use interface for all features

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application (choose your interface):
```bash
python run.py
```

Or run directly:
```bash
# GUI Interface (Recommended)
python main.py

# Command Line Interface
python cli_main.py
```

## Directory Structure

```
PokerHelper/
├── main.py                 # Main application entry point
├── poker_vision_detector.py  # GUI for card detection
├── simulator_gui.py        # GUI for manual simulation
├── src/                    # Source code
│   ├── poker_vision/      # Card detection modules
│   ├── simulator/         # Poker simulation modules
│   └── gui/              # GUI components
├── config/                # Configuration files
│   ├── rank.onnx         # Rank detection model
│   ├── suit.onnx         # Suit detection model
│   └── roi_config.json   # Region configuration
├── output/               # Output files
│   ├── state.json        # Current game state
│   ├── debug/           # Debug images
│   └── images/          # Saved images
└── requirements.txt     # Python dependencies
```

## Usage

### Poker Vision Detection
1. Click "🎯 Poker Vision Detection"
2. Press 'h' to select hand region
3. Press 'b' to select board region
4. Press 'd' to start detection
5. Watch real-time card detection

### Manual Simulator
1. Click "🎲 Manual Simulator"
2. Enter your hand (e.g., "As Kh")
3. Enter board cards (e.g., "7h 2d 2s")
4. Click "Calculate Win Probability"
5. View detailed analysis

## Controls

### Poker Vision
- **h**: Select hand region
- **b**: Select board region
- **d**: Start detection
- **r**: Restart
- **q**: Quit

### Manual Simulator
- Enter cards in standard format (e.g., "As Kh", "10c 9d")
- Adjust players (2-10) and trials (1000-100000)
- View equity, Kelly criterion, and confidence intervals

## Output

Results are saved to `output/state.json` with the following format:
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

## Requirements

- Python 3.11+
- OpenCV
- MSS (screen capture)
- ONNX Runtime
- Tkinter (GUI)
- Treys (poker evaluation)
- NumPy
- Pillow

## Troubleshooting

- **Detection issues**: Try adjusting region selection or enabling debug mode
- **Model errors**: Ensure rank.onnx and suit.onnx are in config/ directory
- **GUI issues**: Check that all dependencies are installed
- **Performance**: Reduce trial count for faster calculations

### Fix: ModuleNotFoundError: No module named '_tkinter' (macOS)

If running `python main.py` raises:
ModuleNotFoundError: No module named '_tkinter'

Quick workaround (no code changes needed)
- Run the CLI entrypoint instead of the GUI:
  - python cli_main.py
  - or python run.py
This will let you use PokerHelper immediately without Tkinter.

Recommended fixes
1) Use the official Python macOS installer (easiest)
- Download and install Python from https://www.python.org/downloads/mac-osx/
- This installer includes a working Tcl/Tk (Tkinter).

2) Install Tcl/Tk via Homebrew and (re)build or reinstall Python (for Homebrew/pyenv users)
- Install Tcl/Tk:
  - brew install tcl-tk
- Example with pyenv (Apple Silicon/Homebrew path shown):
  - export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
  - export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"
  - export PKG_CONFIG_PATH="/opt/homebrew/opt/tcl-tk/lib/pkgconfig"
  - env LDFLAGS="$LDFLAGS" CPPFLAGS="$CPPFLAGS" pyenv install 3.11.14
- After reinstall, recreate your virtualenv and reinstall requirements.

Notes
- If you installed Python via Homebrew, ensure your PATH picks up the Homebrew Python that links to tcl-tk.
- If you prefer not to reinstall Python, use the CLI fallback above until you can update your Python build.
