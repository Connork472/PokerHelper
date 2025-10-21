#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🧩 Setting up project environment..."

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Detected macOS"
  if ! brew list python-tk &>/dev/null; then
    echo "Installing Tkinter via Homebrew..."
    brew install python-tk || brew install tcl-tk
  fi
fi

# Ensure Python 3 is installed
if ! command -v python3 &>/dev/null; then
  echo "Python3 not found. Installing..."
  brew install python
fi

# Create virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
else
  echo ".venv already exists, skipping creation."
fi

# Activate environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
if [ -f "requirements.txt" ]; then
  echo "Installing Python packages..."
  pip install -r requirements.txt
else
  echo "⚠️ No requirements.txt found — skipping package installation."
fi

echo "✅ Environment setup complete!"
echo "To activate manually later, run:"
echo "source .venv/bin/activate"
